use super::specification::try_check_offsets_bounds;
use super::{new_empty_array, Array, Splitable};
use crate::bitmap::Bitmap;
use crate::datatypes::{ArrowDataType, Field};
use crate::offset::OffsetsBuffer;

mod ffi;
pub(super) mod fmt;
mod iterator;

use polars_error::{polars_bail, PolarsResult};

/// An array representing a (key, value), both of arbitrary logical types.
#[derive(Clone)]
pub struct MapArray {
    dtype: ArrowDataType,
    // invariant: field.len() == offsets.len()
    offsets: OffsetsBuffer<i32>,
    field: Box<dyn Array>,
    // invariant: offsets.len() - 1 == Bitmap::len()
    validity: Option<Bitmap>,
}

impl MapArray {
    /// Returns a new [`MapArray`].
    /// # Errors
    /// This function errors iff:
    /// * `offsets.last()` is greater than `field.len()`
    /// * The `dtype`'s physical type is not [`crate::datatypes::PhysicalType::Map`]
    /// * The fields' `dtype` is not equal to the inner field of `dtype`
    /// * The validity is not `None` and its length is different from `offsets.len() - 1`.
    pub fn try_new(
        dtype: ArrowDataType,
        offsets: OffsetsBuffer<i32>,
        field: Box<dyn Array>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        try_check_offsets_bounds(&offsets, field.len())?;

        let inner_field = Self::try_get_field(&dtype)?;
        if let ArrowDataType::Struct(inner) = inner_field.dtype() {
            if inner.len() != 2 {
                polars_bail!(ComputeError: "MapArray's inner `Struct` must have 2 fields (keys and maps)")
            }
        } else {
            polars_bail!(ComputeError: "MapArray expects `DataType::Struct` as its inner logical type")
        }
        if field.dtype() != inner_field.dtype() {
            polars_bail!(ComputeError: "MapArray expects `field.dtype` to match its inner DataType")
        }

        if validity
            .as_ref()
            .is_some_and(|validity| validity.len() != offsets.len_proxy())
        {
            polars_bail!(ComputeError: "validity mask length must match the number of values")
        }

        Ok(Self {
            dtype,
            field,
            offsets,
            validity,
        })
    }

    /// Creates a new [`MapArray`].
    /// # Panics
    /// * `offsets.last()` is greater than `field.len()`.
    /// * The `dtype`'s physical type is not [`crate::datatypes::PhysicalType::Map`],
    /// * The validity is not `None` and its length is different from `offsets.len() - 1`.
    pub fn new(
        dtype: ArrowDataType,
        offsets: OffsetsBuffer<i32>,
        field: Box<dyn Array>,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new(dtype, offsets, field, validity).unwrap()
    }

    /// Returns a new null [`MapArray`] of `length`.
    pub fn new_null(dtype: ArrowDataType, length: usize) -> Self {
        let field = new_empty_array(Self::get_field(&dtype).dtype().clone());
        Self::new(
            dtype,
            vec![0i32; 1 + length].try_into().unwrap(),
            field,
            Some(Bitmap::new_zeroed(length)),
        )
    }

    /// Returns a new empty [`MapArray`].
    pub fn new_empty(dtype: ArrowDataType) -> Self {
        let field = new_empty_array(Self::get_field(&dtype).dtype().clone());
        Self::new(dtype, OffsetsBuffer::default(), field, None)
    }
}

impl MapArray {
    /// Returns a slice of this [`MapArray`].
    /// # Panics
    /// panics iff `offset + length > self.len()`
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new Buffer cannot exceed the existing length"
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Returns a slice of this [`MapArray`].
    ///
    /// # Safety
    /// The caller must ensure that `offset + length < self.len()`.
    #[inline]
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

    pub(crate) fn try_get_field(dtype: &ArrowDataType) -> PolarsResult<&Field> {
        if let ArrowDataType::Map(field, _) = dtype.to_logical_type() {
            Ok(field.as_ref())
        } else {
            polars_bail!(ComputeError: "The dtype's logical type must be DataType::Map")
        }
    }

    pub(crate) fn get_field(dtype: &ArrowDataType) -> &Field {
        Self::try_get_field(dtype).unwrap()
    }
}

// Accessors
impl MapArray {
    /// Returns the length of this array
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    /// returns the offsets
    #[inline]
    pub fn offsets(&self) -> &OffsetsBuffer<i32> {
        &self.offsets
    }

    /// Returns the field (guaranteed to be a `Struct`)
    #[inline]
    pub fn field(&self) -> &Box<dyn Array> {
        &self.field
    }

    /// Returns the element at index `i`.
    #[inline]
    pub fn value(&self, i: usize) -> Box<dyn Array> {
        assert!(i < self.len());
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the element at index `i`.
    ///
    /// # Safety
    /// Assumes that the `i < self.len`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> Box<dyn Array> {
        // soundness: the invariant of the function
        let (start, end) = self.offsets.start_end_unchecked(i);
        let length = end - start;

        // soundness: the invariant of the struct
        self.field.sliced_unchecked(start, length)
    }
}

impl Array for MapArray {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    #[inline]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.clone().with_validity(validity))
    }
}

impl Splitable for MapArray {
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs_offsets, rhs_offsets) = unsafe { self.offsets.split_at_unchecked(offset) };
        let (lhs_validity, rhs_validity) = unsafe { self.validity.split_at_unchecked(offset) };

        (
            Self {
                dtype: self.dtype.clone(),
                offsets: lhs_offsets,
                field: self.field.clone(),
                validity: lhs_validity,
            },
            Self {
                dtype: self.dtype.clone(),
                offsets: rhs_offsets,
                field: self.field.clone(),
                validity: rhs_validity,
            },
        )
    }
}
