use either::Either;

use super::specification::try_check_offsets_bounds;
use super::{Array, GenericBinaryArray, Splitable};
use crate::array::iterator::NonNullValuesIter;
use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::offset::{Offset, Offsets, OffsetsBuffer};
use crate::trusted_len::TrustedLen;

mod ffi;
pub(super) mod fmt;
mod iterator;
pub use iterator::*;
mod from;
mod mutable_values;
pub use mutable_values::*;
mod mutable;
pub use mutable::*;
use polars_error::{polars_bail, PolarsResult};

#[cfg(feature = "arrow_rs")]
mod data;

/// A [`BinaryArray`] is Arrow's semantically equivalent of an immutable `Vec<Option<Vec<u8>>>`.
/// It implements [`Array`].
///
/// The size of this struct is `O(1)`, as all data is stored behind an [`std::sync::Arc`].
/// # Example
/// ```
/// use polars_arrow::array::BinaryArray;
/// use polars_arrow::bitmap::Bitmap;
/// use polars_arrow::buffer::Buffer;
///
/// let array = BinaryArray::<i32>::from([Some([1, 2].as_ref()), None, Some([3].as_ref())]);
/// assert_eq!(array.value(0), &[1, 2]);
/// assert_eq!(array.iter().collect::<Vec<_>>(), vec![Some([1, 2].as_ref()), None, Some([3].as_ref())]);
/// assert_eq!(array.values_iter().collect::<Vec<_>>(), vec![[1, 2].as_ref(), &[], &[3]]);
/// // the underlying representation:
/// assert_eq!(array.values(), &Buffer::from(vec![1, 2, 3]));
/// assert_eq!(array.offsets().buffer(), &Buffer::from(vec![0, 2, 2, 3]));
/// assert_eq!(array.validity(), Some(&Bitmap::from([true, false, true])));
/// ```
///
/// # Generic parameter
/// The generic parameter [`Offset`] can only be `i32` or `i64` and tradeoffs maximum array length with
/// memory usage:
/// * the sum of lengths of all elements cannot exceed `Offset::MAX`
/// * the total size of the underlying data is `array.len() * size_of::<Offset>() + sum of lengths of all elements`
///
/// # Safety
/// The following invariants hold:
/// * Two consecutives `offsets` casted (`as`) to `usize` are valid slices of `values`.
/// * `len` is equal to `validity.len()`, when defined.
#[derive(Clone)]
pub struct BinaryArray<O: Offset> {
    data_type: ArrowDataType,
    offsets: OffsetsBuffer<O>,
    values: Buffer<u8>,
    validity: Option<Bitmap>,
}

impl<O: Offset> BinaryArray<O> {
    /// Returns a [`BinaryArray`] created from its internal representation.
    ///
    /// # Errors
    /// This function returns an error iff:
    /// * The last offset is not equal to the values' length.
    /// * the validity's length is not equal to `offsets.len()`.
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to either `Binary` or `LargeBinary`.
    /// # Implementation
    /// This function is `O(1)`
    pub fn try_new(
        data_type: ArrowDataType,
        offsets: OffsetsBuffer<O>,
        values: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        try_check_offsets_bounds(&offsets, values.len())?;

        if validity
            .as_ref()
            .map_or(false, |validity| validity.len() != offsets.len_proxy())
        {
            polars_bail!(ComputeError: "validity mask length must match the number of values")
        }

        if data_type.to_physical_type() != Self::default_data_type().to_physical_type() {
            polars_bail!(ComputeError: "BinaryArray can only be initialized with DataType::Binary or DataType::LargeBinary")
        }

        Ok(Self {
            data_type,
            offsets,
            values,
            validity,
        })
    }

    /// Creates a new [`BinaryArray`] without checking invariants.
    ///
    /// # Safety
    ///
    /// The invariants must be valid (see try_new).
    pub unsafe fn new_unchecked(
        data_type: ArrowDataType,
        offsets: OffsetsBuffer<O>,
        values: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> Self {
        Self {
            data_type,
            offsets,
            values,
            validity,
        }
    }

    /// Creates a new [`BinaryArray`] from slices of `&[u8]`.
    pub fn from_slice<T: AsRef<[u8]>, P: AsRef<[T]>>(slice: P) -> Self {
        Self::from_trusted_len_values_iter(slice.as_ref().iter())
    }

    /// Creates a new [`BinaryArray`] from a slice of optional `&[u8]`.
    // Note: this can't be `impl From` because Rust does not allow double `AsRef` on it.
    pub fn from<T: AsRef<[u8]>, P: AsRef<[Option<T>]>>(slice: P) -> Self {
        MutableBinaryArray::<O>::from(slice).into()
    }

    /// Returns an iterator of `Option<&[u8]>` over every element of this array.
    pub fn iter(&self) -> ZipValidity<&[u8], BinaryValueIter<O>, BitmapIter> {
        ZipValidity::new_with_validity(self.values_iter(), self.validity.as_ref())
    }

    /// Returns an iterator of `&[u8]` over every element of this array, ignoring the validity
    pub fn values_iter(&self) -> BinaryValueIter<O> {
        BinaryValueIter::new(self)
    }

    /// Returns an iterator of the non-null values.
    #[inline]
    pub fn non_null_values_iter(&self) -> NonNullValuesIter<'_, BinaryArray<O>> {
        NonNullValuesIter::new(self, self.validity())
    }

    /// Returns the length of this array
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    /// Returns the element at index `i`
    /// # Panics
    /// iff `i >= self.len()`
    #[inline]
    pub fn value(&self, i: usize) -> &[u8] {
        assert!(i < self.len());
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the element at index `i`
    ///
    /// # Safety
    /// Assumes that the `i < self.len`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        // soundness: the invariant of the function
        let (start, end) = self.offsets.start_end_unchecked(i);

        // soundness: the invariant of the struct
        self.values.get_unchecked(start..end)
    }

    /// Returns the element at index `i` or `None` if it is null
    /// # Panics
    /// iff `i >= self.len()`
    #[inline]
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        if !self.is_null(i) {
            // soundness: Array::is_null panics if i >= self.len
            unsafe { Some(self.value_unchecked(i)) }
        } else {
            None
        }
    }

    /// Returns the [`ArrowDataType`] of this array.
    #[inline]
    pub fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    /// Returns the values of this [`BinaryArray`].
    #[inline]
    pub fn values(&self) -> &Buffer<u8> {
        &self.values
    }

    /// Returns the offsets of this [`BinaryArray`].
    #[inline]
    pub fn offsets(&self) -> &OffsetsBuffer<O> {
        &self.offsets
    }

    /// The optional validity.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Slices this [`BinaryArray`].
    /// # Implementation
    /// This function is `O(1)`.
    /// # Panics
    /// iff `offset + length > self.len()`.
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new Buffer cannot exceed the existing length"
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Slices this [`BinaryArray`].
    /// # Implementation
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// The caller must ensure that `offset + length <= self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.validity = self
            .validity
            .take()
            .map(|bitmap| bitmap.sliced_unchecked(offset, length))
            .filter(|bitmap| bitmap.unset_bits() > 0);
        self.offsets.slice_unchecked(offset, length + 1);
    }

    impl_sliced!();
    impl_mut_validity!();
    impl_into_array!();

    /// Returns its internal representation
    #[must_use]
    pub fn into_inner(self) -> (ArrowDataType, OffsetsBuffer<O>, Buffer<u8>, Option<Bitmap>) {
        let Self {
            data_type,
            offsets,
            values,
            validity,
        } = self;
        (data_type, offsets, values, validity)
    }

    /// Try to convert this `BinaryArray` to a `MutableBinaryArray`
    #[must_use]
    pub fn into_mut(self) -> Either<Self, MutableBinaryArray<O>> {
        use Either::*;
        if let Some(bitmap) = self.validity {
            match bitmap.into_mut() {
                // SAFETY: invariants are preserved
                Left(bitmap) => Left(BinaryArray::new(
                    self.data_type,
                    self.offsets,
                    self.values,
                    Some(bitmap),
                )),
                Right(mutable_bitmap) => match (self.values.into_mut(), self.offsets.into_mut()) {
                    (Left(values), Left(offsets)) => Left(BinaryArray::new(
                        self.data_type,
                        offsets,
                        values,
                        Some(mutable_bitmap.into()),
                    )),
                    (Left(values), Right(offsets)) => Left(BinaryArray::new(
                        self.data_type,
                        offsets.into(),
                        values,
                        Some(mutable_bitmap.into()),
                    )),
                    (Right(values), Left(offsets)) => Left(BinaryArray::new(
                        self.data_type,
                        offsets,
                        values.into(),
                        Some(mutable_bitmap.into()),
                    )),
                    (Right(values), Right(offsets)) => Right(
                        MutableBinaryArray::try_new(
                            self.data_type,
                            offsets,
                            values,
                            Some(mutable_bitmap),
                        )
                        .unwrap(),
                    ),
                },
            }
        } else {
            match (self.values.into_mut(), self.offsets.into_mut()) {
                (Left(values), Left(offsets)) => {
                    Left(BinaryArray::new(self.data_type, offsets, values, None))
                },
                (Left(values), Right(offsets)) => Left(BinaryArray::new(
                    self.data_type,
                    offsets.into(),
                    values,
                    None,
                )),
                (Right(values), Left(offsets)) => Left(BinaryArray::new(
                    self.data_type,
                    offsets,
                    values.into(),
                    None,
                )),
                (Right(values), Right(offsets)) => Right(
                    MutableBinaryArray::try_new(self.data_type, offsets, values, None).unwrap(),
                ),
            }
        }
    }

    /// Creates an empty [`BinaryArray`], i.e. whose `.len` is zero.
    pub fn new_empty(data_type: ArrowDataType) -> Self {
        Self::new(data_type, OffsetsBuffer::new(), Buffer::new(), None)
    }

    /// Creates an null [`BinaryArray`], i.e. whose `.null_count() == .len()`.
    #[inline]
    pub fn new_null(data_type: ArrowDataType, length: usize) -> Self {
        unsafe {
            Self::new_unchecked(
                data_type,
                Offsets::new_zeroed(length).into(),
                Buffer::new(),
                Some(Bitmap::new_zeroed(length)),
            )
        }
    }

    /// Returns the default [`ArrowDataType`], `DataType::Binary` or `DataType::LargeBinary`
    pub fn default_data_type() -> ArrowDataType {
        if O::IS_LARGE {
            ArrowDataType::LargeBinary
        } else {
            ArrowDataType::Binary
        }
    }

    /// Alias for unwrapping [`Self::try_new`]
    pub fn new(
        data_type: ArrowDataType,
        offsets: OffsetsBuffer<O>,
        values: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new(data_type, offsets, values, validity).unwrap()
    }

    /// Returns a [`BinaryArray`] from an iterator of trusted length.
    ///
    /// The [`BinaryArray`] is guaranteed to not have a validity
    #[inline]
    pub fn from_trusted_len_values_iter<T: AsRef<[u8]>, I: TrustedLen<Item = T>>(
        iterator: I,
    ) -> Self {
        MutableBinaryArray::<O>::from_trusted_len_values_iter(iterator).into()
    }

    /// Returns a new [`BinaryArray`] from a [`Iterator`] of `&[u8]`.
    ///
    /// The [`BinaryArray`] is guaranteed to not have a validity
    pub fn from_iter_values<T: AsRef<[u8]>, I: Iterator<Item = T>>(iterator: I) -> Self {
        MutableBinaryArray::<O>::from_iter_values(iterator).into()
    }

    /// Creates a [`BinaryArray`] from an iterator of trusted length.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn from_trusted_len_iter_unchecked<I, P>(iterator: I) -> Self
    where
        P: AsRef<[u8]>,
        I: Iterator<Item = Option<P>>,
    {
        MutableBinaryArray::<O>::from_trusted_len_iter_unchecked(iterator).into()
    }

    /// Creates a [`BinaryArray`] from a [`TrustedLen`]
    #[inline]
    pub fn from_trusted_len_iter<I, P>(iterator: I) -> Self
    where
        P: AsRef<[u8]>,
        I: TrustedLen<Item = Option<P>>,
    {
        // soundness: I is `TrustedLen`
        unsafe { Self::from_trusted_len_iter_unchecked(iterator) }
    }

    /// Creates a [`BinaryArray`] from an falible iterator of trusted length.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn try_from_trusted_len_iter_unchecked<E, I, P>(iterator: I) -> Result<Self, E>
    where
        P: AsRef<[u8]>,
        I: IntoIterator<Item = Result<Option<P>, E>>,
    {
        MutableBinaryArray::<O>::try_from_trusted_len_iter_unchecked(iterator).map(|x| x.into())
    }

    /// Creates a [`BinaryArray`] from an fallible iterator of trusted length.
    #[inline]
    pub fn try_from_trusted_len_iter<E, I, P>(iter: I) -> Result<Self, E>
    where
        P: AsRef<[u8]>,
        I: TrustedLen<Item = Result<Option<P>, E>>,
    {
        // soundness: I: TrustedLen
        unsafe { Self::try_from_trusted_len_iter_unchecked(iter) }
    }
}

impl<O: Offset> Array for BinaryArray<O> {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    #[inline]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.clone().with_validity(validity))
    }
}

unsafe impl<O: Offset> GenericBinaryArray<O> for BinaryArray<O> {
    #[inline]
    fn values(&self) -> &[u8] {
        self.values()
    }

    #[inline]
    fn offsets(&self) -> &[O] {
        self.offsets().buffer()
    }
}

impl<O: Offset> Splitable for BinaryArray<O> {
    #[inline(always)]
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs_offsets, rhs_offsets) = unsafe { self.offsets.split_at_unchecked(offset) };
        let (lhs_validity, rhs_validity) = unsafe { self.validity.split_at_unchecked(offset) };

        (
            Self {
                data_type: self.data_type.clone(),
                offsets: lhs_offsets,
                values: self.values.clone(),
                validity: lhs_validity,
            },
            Self {
                data_type: self.data_type.clone(),
                offsets: rhs_offsets,
                values: self.values.clone(),
                validity: rhs_validity,
            },
        )
    }
}
