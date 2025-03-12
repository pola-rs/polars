use super::{Array, Splitable};
use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;

mod builder;
mod ffi;
pub(super) mod fmt;
mod iterator;
pub use builder::*;
mod mutable;
pub use mutable::*;
use polars_error::{PolarsResult, polars_bail, polars_ensure};

/// The Arrow's equivalent to an immutable `Vec<Option<[u8; size]>>`.
/// Cloning and slicing this struct is `O(1)`.
#[derive(Clone)]
pub struct FixedSizeBinaryArray {
    size: usize, // this is redundant with `dtype`, but useful to not have to deconstruct the dtype.
    dtype: ArrowDataType,
    values: Buffer<u8>,
    validity: Option<Bitmap>,
}

impl FixedSizeBinaryArray {
    /// Creates a new [`FixedSizeBinaryArray`].
    ///
    /// # Errors
    /// This function returns an error iff:
    /// * The `dtype`'s physical type is not [`crate::datatypes::PhysicalType::FixedSizeBinary`]
    /// * The length of `values` is not a multiple of `size` in `dtype`
    /// * the validity's length is not equal to `values.len() / size`.
    pub fn try_new(
        dtype: ArrowDataType,
        values: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        let size = Self::maybe_get_size(&dtype)?;

        if values.len() % size != 0 {
            polars_bail!(ComputeError:
                "values (of len {}) must be a multiple of size ({}) in FixedSizeBinaryArray.",
                values.len(),
                size
            )
        }
        let len = values.len() / size;

        if validity
            .as_ref()
            .is_some_and(|validity| validity.len() != len)
        {
            polars_bail!(ComputeError: "validity mask length must be equal to the number of values divided by size")
        }

        Ok(Self {
            size,
            dtype,
            values,
            validity,
        })
    }

    /// Creates a new [`FixedSizeBinaryArray`].
    /// # Panics
    /// This function panics iff:
    /// * The `dtype`'s physical type is not [`crate::datatypes::PhysicalType::FixedSizeBinary`]
    /// * The length of `values` is not a multiple of `size` in `dtype`
    /// * the validity's length is not equal to `values.len() / size`.
    pub fn new(dtype: ArrowDataType, values: Buffer<u8>, validity: Option<Bitmap>) -> Self {
        Self::try_new(dtype, values, validity).unwrap()
    }

    /// Returns a new empty [`FixedSizeBinaryArray`].
    pub fn new_empty(dtype: ArrowDataType) -> Self {
        Self::new(dtype, Buffer::new(), None)
    }

    /// Returns a new null [`FixedSizeBinaryArray`].
    pub fn new_null(dtype: ArrowDataType, length: usize) -> Self {
        let size = Self::maybe_get_size(&dtype).unwrap();
        Self::new(
            dtype,
            vec![0u8; length * size].into(),
            Some(Bitmap::new_zeroed(length)),
        )
    }
}

// must use
impl FixedSizeBinaryArray {
    /// Slices this [`FixedSizeBinaryArray`].
    /// # Implementation
    /// This operation is `O(1)`.
    /// # Panics
    /// panics iff `offset + length > self.len()`
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new Buffer cannot exceed the existing length"
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Slices this [`FixedSizeBinaryArray`].
    /// # Implementation
    /// This operation is `O(1)`.
    ///
    /// # Safety
    /// The caller must ensure that `offset + length <= self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.validity = self
            .validity
            .take()
            .map(|bitmap| bitmap.sliced_unchecked(offset, length))
            .filter(|bitmap| bitmap.unset_bits() > 0);
        self.values
            .slice_unchecked(offset * self.size, length * self.size);
    }

    impl_sliced!();
    impl_mut_validity!();
    impl_into_array!();
}

// accessors
impl FixedSizeBinaryArray {
    /// Returns the length of this array
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len() / self.size
    }

    /// The optional validity.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Returns the values allocated on this [`FixedSizeBinaryArray`].
    pub fn values(&self) -> &Buffer<u8> {
        &self.values
    }

    /// Returns value at position `i`.
    /// # Panic
    /// Panics iff `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> &[u8] {
        assert!(i < self.len());
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the element at index `i` as &str
    ///
    /// # Safety
    /// Assumes that the `i < self.len`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        // soundness: invariant of the function.
        self.values
            .get_unchecked(i * self.size..(i + 1) * self.size)
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

    /// Returns a new [`FixedSizeBinaryArray`] with a different logical type.
    /// This is `O(1)`.
    /// # Panics
    /// Panics iff the dtype is not supported for the physical type.
    #[inline]
    pub fn to(self, dtype: ArrowDataType) -> Self {
        match (dtype.to_logical_type(), self.dtype().to_logical_type()) {
            (ArrowDataType::FixedSizeBinary(size_a), ArrowDataType::FixedSizeBinary(size_b))
                if size_a == size_b => {},
            _ => panic!("Wrong DataType"),
        }

        Self {
            size: self.size,
            dtype,
            values: self.values,
            validity: self.validity,
        }
    }

    /// Returns the size
    pub fn size(&self) -> usize {
        self.size
    }
}

impl FixedSizeBinaryArray {
    pub(crate) fn maybe_get_size(dtype: &ArrowDataType) -> PolarsResult<usize> {
        match dtype.to_logical_type() {
            ArrowDataType::FixedSizeBinary(size) => {
                polars_ensure!(*size != 0, ComputeError: "FixedSizeBinaryArray expects a positive size");
                Ok(*size)
            },
            other => {
                polars_bail!(ComputeError: "FixedSizeBinaryArray expects DataType::FixedSizeBinary. found {other:?}")
            },
        }
    }

    pub fn get_size(dtype: &ArrowDataType) -> usize {
        Self::maybe_get_size(dtype).unwrap()
    }
}

impl Array for FixedSizeBinaryArray {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    #[inline]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.clone().with_validity(validity))
    }
}

impl Splitable for FixedSizeBinaryArray {
    fn check_bound(&self, offset: usize) -> bool {
        offset < self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs_values, rhs_values) = unsafe { self.values.split_at_unchecked(offset) };
        let (lhs_validity, rhs_validity) = unsafe { self.validity.split_at_unchecked(offset) };

        let size = self.size;

        (
            Self {
                dtype: self.dtype.clone(),
                values: lhs_values,
                validity: lhs_validity,
                size,
            },
            Self {
                dtype: self.dtype.clone(),
                values: rhs_values,
                validity: rhs_validity,
                size,
            },
        )
    }
}

impl FixedSizeBinaryArray {
    /// Creates a [`FixedSizeBinaryArray`] from an fallible iterator of optional `[u8]`.
    pub fn try_from_iter<P: AsRef<[u8]>, I: IntoIterator<Item = Option<P>>>(
        iter: I,
        size: usize,
    ) -> PolarsResult<Self> {
        MutableFixedSizeBinaryArray::try_from_iter(iter, size).map(|x| x.into())
    }

    /// Creates a [`FixedSizeBinaryArray`] from an iterator of optional `[u8]`.
    pub fn from_iter<P: AsRef<[u8]>, I: IntoIterator<Item = Option<P>>>(
        iter: I,
        size: usize,
    ) -> Self {
        MutableFixedSizeBinaryArray::try_from_iter(iter, size)
            .unwrap()
            .into()
    }

    /// Creates a [`FixedSizeBinaryArray`] from a slice of arrays of bytes
    pub fn from_slice<const N: usize, P: AsRef<[[u8; N]]>>(a: P) -> Self {
        let values = a.as_ref().iter().flatten().copied().collect::<Vec<_>>();
        Self::new(ArrowDataType::FixedSizeBinary(N), values.into(), None)
    }

    /// Creates a new [`FixedSizeBinaryArray`] from a slice of optional `[u8]`.
    // Note: this can't be `impl From` because Rust does not allow double `AsRef` on it.
    pub fn from<const N: usize, P: AsRef<[Option<[u8; N]>]>>(slice: P) -> Self {
        MutableFixedSizeBinaryArray::from(slice).into()
    }
}
