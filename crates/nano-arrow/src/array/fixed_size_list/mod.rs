use super::{new_empty_array, new_null_array, Array};
use crate::bitmap::Bitmap;
use crate::datatypes::{DataType, Field};
use crate::error::Error;

#[cfg(feature = "arrow")]
mod data;
mod ffi;
pub(super) mod fmt;
mod iterator;
pub use iterator::*;
mod mutable;
pub use mutable::*;

/// The Arrow's equivalent to an immutable `Vec<Option<[T; size]>>` where `T` is an Arrow type.
/// Cloning and slicing this struct is `O(1)`.
#[derive(Clone)]
pub struct FixedSizeListArray {
    size: usize, // this is redundant with `data_type`, but useful to not have to deconstruct the data_type.
    data_type: DataType,
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
        data_type: DataType,
        values: Box<dyn Array>,
        validity: Option<Bitmap>,
    ) -> Result<Self, Error> {
        let (child, size) = Self::try_child_and_size(&data_type)?;

        let child_data_type = &child.data_type;
        let values_data_type = values.data_type();
        if child_data_type != values_data_type {
            return Err(Error::oos(
                format!("FixedSizeListArray's child's DataType must match. However, the expected DataType is {child_data_type:?} while it got {values_data_type:?}."),
            ));
        }

        if values.len() % size != 0 {
            return Err(Error::oos(format!(
                "values (of len {}) must be a multiple of size ({}) in FixedSizeListArray.",
                values.len(),
                size
            )));
        }
        let len = values.len() / size;

        if validity
            .as_ref()
            .map_or(false, |validity| validity.len() != len)
        {
            return Err(Error::oos(
                "validity mask length must be equal to the number of values divided by size",
            ));
        }

        Ok(Self {
            size,
            data_type,
            values,
            validity,
        })
    }

    /// Alias to `Self::try_new(...).unwrap()`
    pub fn new(data_type: DataType, values: Box<dyn Array>, validity: Option<Bitmap>) -> Self {
        Self::try_new(data_type, values, validity).unwrap()
    }

    /// Returns the size (number of elements per slot) of this [`FixedSizeListArray`].
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Returns a new empty [`FixedSizeListArray`].
    pub fn new_empty(data_type: DataType) -> Self {
        let values = new_empty_array(Self::get_child_and_size(&data_type).0.data_type().clone());
        Self::new(data_type, values, None)
    }

    /// Returns a new null [`FixedSizeListArray`].
    pub fn new_null(data_type: DataType, length: usize) -> Self {
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
    /// # Safety
    /// The caller must ensure that `offset + length <= self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.validity.as_mut().and_then(|bitmap| {
            bitmap.slice_unchecked(offset, length);
            (bitmap.unset_bits() > 0).then(|| bitmap)
        });
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
    pub(crate) fn try_child_and_size(data_type: &DataType) -> Result<(&Field, usize), Error> {
        match data_type.to_logical_type() {
            DataType::FixedSizeList(child, size) => {
                if *size == 0 {
                    return Err(Error::oos("FixedSizeBinaryArray expects a positive size"));
                }
                Ok((child.as_ref(), *size))
            },
            _ => Err(Error::oos(
                "FixedSizeListArray expects DataType::FixedSizeList",
            )),
        }
    }

    pub(crate) fn get_child_and_size(data_type: &DataType) -> (&Field, usize) {
        Self::try_child_and_size(data_type).unwrap()
    }

    /// Returns a [`DataType`] consistent with [`FixedSizeListArray`].
    pub fn default_datatype(data_type: DataType, size: usize) -> DataType {
        let field = Box::new(Field::new("item", data_type, true));
        DataType::FixedSizeList(field, size)
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
