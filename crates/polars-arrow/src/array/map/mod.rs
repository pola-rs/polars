use super::specification::try_check_offsets_bounds;
use super::{new_empty_array, Array, Splitable};
use crate::bitmap::Bitmap;
use crate::datatypes::{ArrowDataType, Field};
use crate::offset::OffsetsBuffer;

#[cfg(feature = "arrow_rs")]
mod data;
mod ffi;
pub(super) mod fmt;
mod iterator;

use polars_error::{polars_bail, PolarsResult};

/// An array representing a (key, value), both of arbitrary logical types.
#[derive(Clone)]
pub struct MapArray {
    data_type: ArrowDataType,
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
    /// * The last offset is not equal to the field' length
    /// * The `data_type`'s physical type is not [`crate::datatypes::PhysicalType::Map`]
    /// * The fields' `data_type` is not equal to the inner field of `data_type`
    /// * The validity is not `None` and its length is different from `offsets.len() - 1`.
    pub fn try_new(
        data_type: ArrowDataType,
        offsets: OffsetsBuffer<i32>,
        field: Box<dyn Array>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        try_check_offsets_bounds(&offsets, field.len())?;

        let inner_field = Self::try_get_field(&data_type)?;
        if let ArrowDataType::Struct(inner) = inner_field.data_type() {
            if inner.len() != 2 {
                polars_bail!(ComputeError: "MapArray's inner `Struct` must have 2 fields (keys and maps)")
            }
        } else {
            polars_bail!(ComputeError: "MapArray expects `DataType::Struct` as its inner logical type")
        }
        if field.data_type() != inner_field.data_type() {
            polars_bail!(ComputeError: "MapArray expects `field.data_type` to match its inner DataType")
        }

        if validity
            .as_ref()
            .map_or(false, |validity| validity.len() != offsets.len_proxy())
        {
            polars_bail!(ComputeError: "validity mask length must match the number of values")
        }

        Ok(Self {
            data_type,
            field,
            offsets,
            validity,
        })
    }

    /// Creates a new [`MapArray`].
    /// # Panics
    /// * The last offset is not equal to the field' length.
    /// * The `data_type`'s physical type is not [`crate::datatypes::PhysicalType::Map`],
    /// * The validity is not `None` and its length is different from `offsets.len() - 1`.
    pub fn new(
        data_type: ArrowDataType,
        offsets: OffsetsBuffer<i32>,
        field: Box<dyn Array>,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new(data_type, offsets, field, validity).unwrap()
    }

    /// Returns a new null [`MapArray`] of `length`.
    pub fn new_null(data_type: ArrowDataType, length: usize) -> Self {
        let field = new_empty_array(Self::get_field(&data_type).data_type().clone());
        Self::new(
            data_type,
            vec![0i32; 1 + length].try_into().unwrap(),
            field,
            Some(Bitmap::new_zeroed(length)),
        )
    }

    /// Returns a new empty [`MapArray`].
    pub fn new_empty(data_type: ArrowDataType) -> Self {
        let field = new_empty_array(Self::get_field(&data_type).data_type().clone());
        Self::new(data_type, OffsetsBuffer::default(), field, None)
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

    pub(crate) fn try_get_field(data_type: &ArrowDataType) -> PolarsResult<&Field> {
        if let ArrowDataType::Map(field, _) = data_type.to_logical_type() {
            Ok(field.as_ref())
        } else {
            polars_bail!(ComputeError: "The data_type's logical type must be DataType::Map")
        }
    }

    pub(crate) fn get_field(data_type: &ArrowDataType) -> &Field {
        Self::try_get_field(data_type).unwrap()
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
                data_type: self.data_type.clone(),
                offsets: lhs_offsets,
                field: self.field.clone(),
                validity: lhs_validity,
            },
            Self {
                data_type: self.data_type.clone(),
                offsets: rhs_offsets,
                field: self.field.clone(),
                validity: rhs_validity,
            },
        )
    }
}
