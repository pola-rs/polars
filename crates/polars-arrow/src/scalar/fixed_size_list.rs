use std::any::Any;

use super::Scalar;
use crate::array::*;
use crate::datatypes::ArrowDataType;

/// The scalar equivalent of [`FixedSizeListArray`]. Like [`FixedSizeListArray`], this struct holds a dynamically-typed
/// [`Array`]. The only difference is that this has only one element.
#[derive(Debug, Clone)]
pub struct FixedSizeListScalar {
    values: Option<Box<dyn Array>>,
    dtype: ArrowDataType,
}

impl PartialEq for FixedSizeListScalar {
    fn eq(&self, other: &Self) -> bool {
        (self.dtype == other.dtype)
            && (self.values.is_some() == other.values.is_some())
            && ((self.values.is_none()) | (self.values.as_ref() == other.values.as_ref()))
    }
}

impl FixedSizeListScalar {
    /// returns a new [`FixedSizeListScalar`]
    /// # Panics
    /// iff
    /// * the `dtype` is not `FixedSizeList`
    /// * the child of the `dtype` is not equal to the `values`
    /// * the size of child array is not equal
    #[inline]
    pub fn new(dtype: ArrowDataType, values: Option<Box<dyn Array>>) -> Self {
        let (field, size) = FixedSizeListArray::get_child_and_size(&dtype);
        let inner_dtype = field.dtype();
        let values = values.inspect(|x| {
            assert_eq!(inner_dtype, x.dtype());
            assert_eq!(size, x.len());
        });
        Self { values, dtype }
    }

    /// The values of the [`FixedSizeListScalar`]
    pub fn values(&self) -> Option<&Box<dyn Array>> {
        self.values.as_ref()
    }
}

impl Scalar for FixedSizeListScalar {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_valid(&self) -> bool {
        self.values.is_some()
    }

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }
}
