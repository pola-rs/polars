use super::{new_empty_array, new_null_array, Array, Splitable};
use crate::bitmap::Bitmap;
use crate::datatypes::{ArrowDataType, Field};

#[cfg(feature = "arrow_rs")]
mod data;
mod ffi;
pub(super) mod fmt;
mod iterator;

mod mutable;
pub use mutable::*;
use polars_error::{polars_bail, PolarsResult};

/// The Arrow's equivalent to an immutable `Vec<Option<[T; size]>>` where `T` is an Arrow type.
/// Cloning and slicing this struct is `O(1)`.
#[derive(Clone)]
pub struct FixedSizeListArray {
    size: usize, // this is redundant with `data_type`, but useful to not have to deconstruct the data_type.
    data_type: ArrowDataType,
    values: Box<dyn Array>,
    validity: Option<Bitmap>,
}

impl FixedSizeListArray {
    /// Creates a new [`FixedSizeListArray`].
    ///
    /// # Errors
    /// This function returns an error iff:
    /// * The `data_type`'s physical type is not [`crate::datatypes::PhysicalType::FixedSizeList`]
    /// * The `data_type`'s inner field's data type is not equal to `values.data_type`.
    /// * The length of `values` is not a multiple of `size` in `data_type`
    /// * the validity's length is not equal to `values.len() / size`.
    pub fn try_new(
        data_type: ArrowDataType,
        values: Box<dyn Array>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        let (child, size) = Self::try_child_and_size(&data_type)?;

        let child_data_type = &child.data_type;
        let values_data_type = values.data_type();
        if child_data_type != values_data_type {
            polars_bail!(ComputeError: "FixedSizeListArray's child's DataType must match. However, the expected DataType is {child_data_type:?} while it got {values_data_type:?}.")
        }

        if values.len() % size != 0 {
            polars_bail!(ComputeError:
                "values (of len {}) must be a multiple of size ({}) in FixedSizeListArray.",
                values.len(),
                size
            )
        }
        let len = values.len() / size;

        if validity
            .as_ref()
            .map_or(false, |validity| validity.len() != len)
        {
            polars_bail!(ComputeError: "validity mask length must be equal to the number of values divided by size")
        }

        Ok(Self {
            size,
            data_type,
            values,
            validity,
        })
    }

    /// Alias to `Self::try_new(...).unwrap()`
    #[track_caller]
    pub fn new(data_type: ArrowDataType, values: Box<dyn Array>, validity: Option<Bitmap>) -> Self {
        Self::try_new(data_type, values, validity).unwrap()
    }

    /// Returns the size (number of elements per slot) of this [`FixedSizeListArray`].
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Returns a new empty [`FixedSizeListArray`].
    pub fn new_empty(data_type: ArrowDataType) -> Self {
        let values = new_empty_array(Self::get_child_and_size(&data_type).0.data_type().clone());
        Self::new(data_type, values, None)
    }

    /// Returns a new null [`FixedSizeListArray`].
    pub fn new_null(data_type: ArrowDataType, length: usize) -> Self {
        let (field, size) = Self::get_child_and_size(&data_type);

        let values = new_null_array(field.data_type().clone(), length * size);
        Self::new(data_type, values, Some(Bitmap::new_zeroed(length)))
    }
}

// must use
impl FixedSizeListArray {
    /// Slices this [`FixedSizeListArray`].
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

    /// Slices this [`FixedSizeListArray`].
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
impl FixedSizeListArray {
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

    /// Returns the inner array.
    pub fn values(&self) -> &Box<dyn Array> {
        &self.values
    }

    /// Returns the `Vec<T>` at position `i`.
    /// # Panic:
    /// panics iff `i >= self.len()`
    #[inline]
    pub fn value(&self, i: usize) -> Box<dyn Array> {
        self.values.sliced(i * self.size, self.size)
    }

    /// Returns the `Vec<T>` at position `i`.
    ///
    /// # Safety
    /// Caller must ensure that `i < self.len()`
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> Box<dyn Array> {
        self.values.sliced_unchecked(i * self.size, self.size)
    }

    /// Returns the element at index `i` or `None` if it is null
    /// # Panics
    /// iff `i >= self.len()`
    #[inline]
    pub fn get(&self, i: usize) -> Option<Box<dyn Array>> {
        if !self.is_null(i) {
            // soundness: Array::is_null panics if i >= self.len
            unsafe { Some(self.value_unchecked(i)) }
        } else {
            None
        }
    }
}

impl FixedSizeListArray {
    pub(crate) fn try_child_and_size(data_type: &ArrowDataType) -> PolarsResult<(&Field, usize)> {
        match data_type.to_logical_type() {
            ArrowDataType::FixedSizeList(child, size) => {
                if *size == 0 {
                    polars_bail!(ComputeError: "FixedSizeBinaryArray expects a positive size")
                }
                Ok((child.as_ref(), *size))
            },
            _ => polars_bail!(ComputeError: "FixedSizeListArray expects DataType::FixedSizeList"),
        }
    }

    pub(crate) fn get_child_and_size(data_type: &ArrowDataType) -> (&Field, usize) {
        Self::try_child_and_size(data_type).unwrap()
    }

    /// Returns a [`ArrowDataType`] consistent with [`FixedSizeListArray`].
    pub fn default_datatype(data_type: ArrowDataType, size: usize) -> ArrowDataType {
        let field = Box::new(Field::new("item", data_type, true));
        ArrowDataType::FixedSizeList(field, size)
    }
}

impl Array for FixedSizeListArray {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    #[inline]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.clone().with_validity(validity))
    }
}

impl Splitable for FixedSizeListArray {
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs_values, rhs_values) =
            unsafe { self.values.split_at_boxed_unchecked(offset * self.size) };
        let (lhs_validity, rhs_validity) =
            unsafe { self.validity.split_at_unchecked(offset * self.size) };

        let size = self.size;

        (
            Self {
                data_type: self.data_type.clone(),
                values: lhs_values,
                validity: lhs_validity,
                size,
            },
            Self {
                data_type: self.data_type.clone(),
                values: rhs_values,
                validity: rhs_validity,
                size,
            },
        )
    }
}
