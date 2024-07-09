use either::Either;

use super::specification::try_check_utf8;
use super::{Array, GenericBinaryArray, Splitable};
use crate::array::iterator::NonNullValuesIter;
use crate::array::BinaryArray;
use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::offset::{Offset, Offsets, OffsetsBuffer};
use crate::trusted_len::TrustedLen;

#[cfg(feature = "arrow_rs")]
mod data;
mod ffi;
pub(super) mod fmt;
mod from;
mod iterator;
mod mutable;
mod mutable_values;
pub use iterator::*;
pub use mutable::*;
pub use mutable_values::MutableUtf8ValuesArray;
use polars_error::*;

// Auxiliary struct to allow presenting &str as [u8] to a generic function
pub(super) struct StrAsBytes<P>(P);
impl<T: AsRef<str>> AsRef<[u8]> for StrAsBytes<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref().as_bytes()
    }
}

/// A [`Utf8Array`] is arrow's semantic equivalent of an immutable `Vec<Option<String>>`.
/// Cloning and slicing this struct is `O(1)`.
/// # Example
/// ```
/// use polars_arrow::bitmap::Bitmap;
/// use polars_arrow::buffer::Buffer;
/// use polars_arrow::array::Utf8Array;
/// # fn main() {
/// let array = Utf8Array::<i32>::from([Some("hi"), None, Some("there")]);
/// assert_eq!(array.value(0), "hi");
/// assert_eq!(array.iter().collect::<Vec<_>>(), vec![Some("hi"), None, Some("there")]);
/// assert_eq!(array.values_iter().collect::<Vec<_>>(), vec!["hi", "", "there"]);
/// // the underlying representation
/// assert_eq!(array.validity(), Some(&Bitmap::from([true, false, true])));
/// assert_eq!(array.values(), &Buffer::from(b"hithere".to_vec()));
/// assert_eq!(array.offsets().buffer(), &Buffer::from(vec![0, 2, 2, 2 + 5]));
/// # }
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
/// * A slice of `values` taken from two consecutives `offsets` is valid `utf8`.
/// * `len` is equal to `validity.len()`, when defined.
#[derive(Clone)]
pub struct Utf8Array<O: Offset> {
    data_type: ArrowDataType,
    offsets: OffsetsBuffer<O>,
    values: Buffer<u8>,
    validity: Option<Bitmap>,
}

// constructors
impl<O: Offset> Utf8Array<O> {
    /// Returns a [`Utf8Array`] created from its internal representation.
    ///
    /// # Errors
    /// This function returns an error iff:
    /// * The last offset is not equal to the values' length.
    /// * the validity's length is not equal to `offsets.len()`.
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to either `Utf8` or `LargeUtf8`.
    /// * The `values` between two consecutive `offsets` are not valid utf8
    /// # Implementation
    /// This function is `O(N)` - checking utf8 is `O(N)`
    pub fn try_new(
        data_type: ArrowDataType,
        offsets: OffsetsBuffer<O>,
        values: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        try_check_utf8(&offsets, &values)?;
        if validity
            .as_ref()
            .map_or(false, |validity| validity.len() != offsets.len_proxy())
        {
            polars_bail!(ComputeError: "validity mask length must match the number of values");
        }

        if data_type.to_physical_type() != Self::default_data_type().to_physical_type() {
            polars_bail!(ComputeError: "Utf8Array can only be initialized with DataType::Utf8 or DataType::LargeUtf8")
        }

        Ok(Self {
            data_type,
            offsets,
            values,
            validity,
        })
    }

    /// Returns a [`Utf8Array`] from a slice of `&str`.
    ///
    /// A convenience method that uses [`Self::from_trusted_len_values_iter`].
    pub fn from_slice<T: AsRef<str>, P: AsRef<[T]>>(slice: P) -> Self {
        Self::from_trusted_len_values_iter(slice.as_ref().iter())
    }

    /// Returns a new [`Utf8Array`] from a slice of `&str`.
    ///
    /// A convenience method that uses [`Self::from_trusted_len_iter`].
    // Note: this can't be `impl From` because Rust does not allow double `AsRef` on it.
    pub fn from<T: AsRef<str>, P: AsRef<[Option<T>]>>(slice: P) -> Self {
        MutableUtf8Array::<O>::from(slice).into()
    }

    /// Returns an iterator of `Option<&str>`
    pub fn iter(&self) -> ZipValidity<&str, Utf8ValuesIter<O>, BitmapIter> {
        ZipValidity::new_with_validity(self.values_iter(), self.validity())
    }

    /// Returns an iterator of `&str`
    pub fn values_iter(&self) -> Utf8ValuesIter<O> {
        Utf8ValuesIter::new(self)
    }

    /// Returns an iterator of the non-null values `&str.
    #[inline]
    pub fn non_null_values_iter(&self) -> NonNullValuesIter<'_, Utf8Array<O>> {
        NonNullValuesIter::new(self, self.validity())
    }

    /// Returns the length of this array
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    /// Returns the value of the element at index `i`, ignoring the array's validity.
    /// # Panic
    /// This function panics iff `i >= self.len`.
    #[inline]
    pub fn value(&self, i: usize) -> &str {
        assert!(i < self.len());
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the value of the element at index `i`, ignoring the array's validity.
    ///
    /// # Safety
    /// This function is safe iff `i < self.len`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &str {
        // soundness: the invariant of the function
        let (start, end) = self.offsets.start_end_unchecked(i);

        // soundness: the invariant of the struct
        let slice = self.values.get_unchecked(start..end);

        // soundness: the invariant of the struct
        std::str::from_utf8_unchecked(slice)
    }

    /// Returns the element at index `i` or `None` if it is null
    /// # Panics
    /// iff `i >= self.len()`
    #[inline]
    pub fn get(&self, i: usize) -> Option<&str> {
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

    /// Returns the values of this [`Utf8Array`].
    #[inline]
    pub fn values(&self) -> &Buffer<u8> {
        &self.values
    }

    /// Returns the offsets of this [`Utf8Array`].
    #[inline]
    pub fn offsets(&self) -> &OffsetsBuffer<O> {
        &self.offsets
    }

    /// The optional validity.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Slices this [`Utf8Array`].
    /// # Implementation
    /// This function is `O(1)`.
    /// # Panics
    /// iff `offset + length > self.len()`.
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new array cannot exceed the arrays' length"
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Slices this [`Utf8Array`].
    /// # Implementation
    /// This function is `O(1)`
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

    /// Try to convert this `Utf8Array` to a `MutableUtf8Array`
    #[must_use]
    pub fn into_mut(self) -> Either<Self, MutableUtf8Array<O>> {
        use Either::*;
        if let Some(bitmap) = self.validity {
            match bitmap.into_mut() {
                // SAFETY: invariants are preserved
                Left(bitmap) => Left(unsafe {
                    Utf8Array::new_unchecked(
                        self.data_type,
                        self.offsets,
                        self.values,
                        Some(bitmap),
                    )
                }),
                Right(mutable_bitmap) => match (self.values.into_mut(), self.offsets.into_mut()) {
                    (Left(values), Left(offsets)) => {
                        // SAFETY: invariants are preserved
                        Left(unsafe {
                            Utf8Array::new_unchecked(
                                self.data_type,
                                offsets,
                                values,
                                Some(mutable_bitmap.into()),
                            )
                        })
                    },
                    (Left(values), Right(offsets)) => {
                        // SAFETY: invariants are preserved
                        Left(unsafe {
                            Utf8Array::new_unchecked(
                                self.data_type,
                                offsets.into(),
                                values,
                                Some(mutable_bitmap.into()),
                            )
                        })
                    },
                    (Right(values), Left(offsets)) => {
                        // SAFETY: invariants are preserved
                        Left(unsafe {
                            Utf8Array::new_unchecked(
                                self.data_type,
                                offsets,
                                values.into(),
                                Some(mutable_bitmap.into()),
                            )
                        })
                    },
                    (Right(values), Right(offsets)) => Right(unsafe {
                        MutableUtf8Array::new_unchecked(
                            self.data_type,
                            offsets,
                            values,
                            Some(mutable_bitmap),
                        )
                    }),
                },
            }
        } else {
            match (self.values.into_mut(), self.offsets.into_mut()) {
                (Left(values), Left(offsets)) => {
                    Left(unsafe { Utf8Array::new_unchecked(self.data_type, offsets, values, None) })
                },
                (Left(values), Right(offsets)) => Left(unsafe {
                    Utf8Array::new_unchecked(self.data_type, offsets.into(), values, None)
                }),
                (Right(values), Left(offsets)) => Left(unsafe {
                    Utf8Array::new_unchecked(self.data_type, offsets, values.into(), None)
                }),
                (Right(values), Right(offsets)) => Right(unsafe {
                    MutableUtf8Array::new_unchecked(self.data_type, offsets, values, None)
                }),
            }
        }
    }

    /// Returns a new empty [`Utf8Array`].
    ///
    /// The array is guaranteed to have no elements nor validity.
    #[inline]
    pub fn new_empty(data_type: ArrowDataType) -> Self {
        unsafe { Self::new_unchecked(data_type, OffsetsBuffer::new(), Buffer::new(), None) }
    }

    /// Returns a new [`Utf8Array`] whose all slots are null / `None`.
    #[inline]
    pub fn new_null(data_type: ArrowDataType, length: usize) -> Self {
        Self::new(
            data_type,
            Offsets::new_zeroed(length).into(),
            Buffer::new(),
            Some(Bitmap::new_zeroed(length)),
        )
    }

    /// Returns a default [`ArrowDataType`] of this array, which depends on the generic parameter `O`: `DataType::Utf8` or `DataType::LargeUtf8`
    pub fn default_data_type() -> ArrowDataType {
        if O::IS_LARGE {
            ArrowDataType::LargeUtf8
        } else {
            ArrowDataType::Utf8
        }
    }

    /// Creates a new [`Utf8Array`] without checking for offsets monotinicity nor utf8-validity
    ///
    /// # Panic
    /// This function panics (in debug mode only) iff:
    /// * The last offset is not equal to the values' length.
    /// * the validity's length is not equal to `offsets.len()`.
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to either `Utf8` or `LargeUtf8`.
    ///
    /// # Safety
    /// This function is unsound iff:
    /// * The `values` between two consecutive `offsets` are not valid utf8
    /// # Implementation
    /// This function is `O(1)`
    pub unsafe fn new_unchecked(
        data_type: ArrowDataType,
        offsets: OffsetsBuffer<O>,
        values: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> Self {
        debug_assert!(
            offsets.last().to_usize() <= values.len(),
            "offsets must not exceed the values length"
        );
        debug_assert!(
            validity
                .as_ref()
                .map_or(true, |validity| validity.len() == offsets.len_proxy()),
            "validity mask length must match the number of values"
        );
        debug_assert!(
            data_type.to_physical_type() == Self::default_data_type().to_physical_type(),
            "Utf8Array can only be initialized with DataType::Utf8 or DataType::LargeUtf8"
        );

        Self {
            data_type,
            offsets,
            values,
            validity,
        }
    }

    /// Creates a new [`Utf8Array`].
    /// # Panics
    /// This function panics iff:
    /// * The last offset is not equal to the values' length.
    /// * the validity's length is not equal to `offsets.len()`.
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to either `Utf8` or `LargeUtf8`.
    /// * The `values` between two consecutive `offsets` are not valid utf8
    /// # Implementation
    /// This function is `O(N)` - checking utf8 is `O(N)`
    pub fn new(
        data_type: ArrowDataType,
        offsets: OffsetsBuffer<O>,
        values: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new(data_type, offsets, values, validity).unwrap()
    }

    /// Returns a (non-null) [`Utf8Array`] created from a [`TrustedLen`] of `&str`.
    /// # Implementation
    /// This function is `O(N)`
    #[inline]
    pub fn from_trusted_len_values_iter<T: AsRef<str>, I: TrustedLen<Item = T>>(
        iterator: I,
    ) -> Self {
        MutableUtf8Array::<O>::from_trusted_len_values_iter(iterator).into()
    }

    /// Creates a new [`Utf8Array`] from a [`Iterator`] of `&str`.
    pub fn from_iter_values<T: AsRef<str>, I: Iterator<Item = T>>(iterator: I) -> Self {
        MutableUtf8Array::<O>::from_iter_values(iterator).into()
    }

    /// Creates a [`Utf8Array`] from an iterator of trusted length.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn from_trusted_len_iter_unchecked<I, P>(iterator: I) -> Self
    where
        P: AsRef<str>,
        I: Iterator<Item = Option<P>>,
    {
        MutableUtf8Array::<O>::from_trusted_len_iter_unchecked(iterator).into()
    }

    /// Creates a [`Utf8Array`] from an iterator of trusted length.
    #[inline]
    pub fn from_trusted_len_iter<I, P>(iterator: I) -> Self
    where
        P: AsRef<str>,
        I: TrustedLen<Item = Option<P>>,
    {
        MutableUtf8Array::<O>::from_trusted_len_iter(iterator).into()
    }

    /// Creates a [`Utf8Array`] from an falible iterator of trusted length.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    #[inline]
    pub unsafe fn try_from_trusted_len_iter_unchecked<E, I, P>(
        iterator: I,
    ) -> std::result::Result<Self, E>
    where
        P: AsRef<str>,
        I: IntoIterator<Item = std::result::Result<Option<P>, E>>,
    {
        MutableUtf8Array::<O>::try_from_trusted_len_iter_unchecked(iterator).map(|x| x.into())
    }

    /// Creates a [`Utf8Array`] from an fallible iterator of trusted length.
    #[inline]
    pub fn try_from_trusted_len_iter<E, I, P>(iter: I) -> std::result::Result<Self, E>
    where
        P: AsRef<str>,
        I: TrustedLen<Item = std::result::Result<Option<P>, E>>,
    {
        MutableUtf8Array::<O>::try_from_trusted_len_iter(iter).map(|x| x.into())
    }

    /// Applies a function `f` to the validity of this array.
    ///
    /// This is an API to leverage clone-on-write
    /// # Panics
    /// This function panics if the function `f` modifies the length of the [`Bitmap`].
    pub fn apply_validity<F: FnOnce(Bitmap) -> Bitmap>(&mut self, f: F) {
        if let Some(validity) = std::mem::take(&mut self.validity) {
            self.set_validity(Some(f(validity)))
        }
    }

    // Convert this [`Utf8Array`] to a [`BinaryArray`].
    pub fn to_binary(&self) -> BinaryArray<O> {
        unsafe {
            BinaryArray::new_unchecked(
                BinaryArray::<O>::default_data_type(),
                self.offsets.clone(),
                self.values.clone(),
                self.validity.clone(),
            )
        }
    }
}

impl<O: Offset> Splitable for Utf8Array<O> {
    #[inline(always)]
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs_validity, rhs_validity) = unsafe { self.validity.split_at_unchecked(offset) };
        let (lhs_offsets, rhs_offsets) = unsafe { self.offsets.split_at_unchecked(offset) };

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

impl<O: Offset> Array for Utf8Array<O> {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    #[inline]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.clone().with_validity(validity))
    }
}

unsafe impl<O: Offset> GenericBinaryArray<O> for Utf8Array<O> {
    #[inline]
    fn values(&self) -> &[u8] {
        self.values()
    }

    #[inline]
    fn offsets(&self) -> &[O] {
        self.offsets().buffer()
    }
}

impl<O: Offset> Default for Utf8Array<O> {
    fn default() -> Self {
        let data_type = if O::IS_LARGE {
            ArrowDataType::LargeUtf8
        } else {
            ArrowDataType::Utf8
        };
        Utf8Array::new(data_type, Default::default(), Default::default(), None)
    }
}
