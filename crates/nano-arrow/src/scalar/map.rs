use std::any::Any;

use crate::{array::*, datatypes::DataType};

use super::Scalar;

/// The scalar equivalent of [`MapArray`]. Like [`MapArray`], this struct holds a dynamically-typed
/// [`Array`]. The only difference is that this has only one element.
#[derive(Debug, Clone)]
pub struct MapScalar {
    values: Box<dyn Array>,
    is_valid: bool,
    data_type: DataType,
}

impl PartialEq for MapScalar {
    fn eq(&self, other: &Self) -> bool {
        (self.data_type == other.data_type)
            && (self.is_valid == other.is_valid)
            && ((!self.is_valid) | (self.values.as_ref() == other.values.as_ref()))
    }
}

impl MapScalar {
    /// returns a new [`MapScalar`]
    /// # Panics
    /// iff
    /// * the `data_type` is not `Map`
    /// * the child of the `data_type` is not equal to the `values`
    #[inline]
    pub fn new(data_type: DataType, values: Option<Box<dyn Array>>) -> Self {
        let inner_field = MapArray::try_get_field(&data_type).unwrap();
        let inner_data_type = inner_field.data_type();
        let (is_valid, values) = match values {
            Some(values) => {
                assert_eq!(inner_data_type, values.data_type());
                (true, values)
            }
            None => (false, new_empty_array(inner_data_type.clone())),
        };
        Self {
            values,
            is_valid,
            data_type,
        }
    }

    /// The values of the [`MapScalar`]
    pub fn values(&self) -> &Box<dyn Array> {
        &self.values
    }
}

impl Scalar for MapScalar {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_valid(&self) -> bool {
        self.is_valid
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }
}
