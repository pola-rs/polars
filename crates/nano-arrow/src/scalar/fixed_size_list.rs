use std::any::Any;

use crate::{array::*, datatypes::DataType};

use super::Scalar;

/// The scalar equivalent of [`FixedSizeListArray`]. Like [`FixedSizeListArray`], this struct holds a dynamically-typed
/// [`Array`]. The only difference is that this has only one element.
#[derive(Debug, Clone)]
pub struct FixedSizeListScalar {
    values: Option<Box<dyn Array>>,
    data_type: DataType,
}

impl PartialEq for FixedSizeListScalar {
    fn eq(&self, other: &Self) -> bool {
        (self.data_type == other.data_type)
            && (self.values.is_some() == other.values.is_some())
            && ((self.values.is_none()) | (self.values.as_ref() == other.values.as_ref()))
    }
}

impl FixedSizeListScalar {
    /// returns a new [`FixedSizeListScalar`]
    /// # Panics
    /// iff
    /// * the `data_type` is not `FixedSizeList`
    /// * the child of the `data_type` is not equal to the `values`
    /// * the size of child array is not equal
    #[inline]
    pub fn new(data_type: DataType, values: Option<Box<dyn Array>>) -> Self {
        let (field, size) = FixedSizeListArray::get_child_and_size(&data_type);
        let inner_data_type = field.data_type();
        let values = values.map(|x| {
            assert_eq!(inner_data_type, x.data_type());
            assert_eq!(size, x.len());
            x
        });
        Self { values, data_type }
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

    fn data_type(&self) -> &DataType {
        &self.data_type
    }
}
