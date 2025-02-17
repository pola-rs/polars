use std::any::Any;

use super::Scalar;
use crate::array::*;
use crate::datatypes::ArrowDataType;

/// The scalar equivalent of [`MapArray`]. Like [`MapArray`], this struct holds a dynamically-typed
/// [`Array`]. The only difference is that this has only one element.
#[derive(Debug, Clone)]
pub struct MapScalar {
    values: Box<dyn Array>,
    is_valid: bool,
    dtype: ArrowDataType,
}

impl PartialEq for MapScalar {
    fn eq(&self, other: &Self) -> bool {
        (self.dtype == other.dtype)
            && (self.is_valid == other.is_valid)
            && ((!self.is_valid) | (self.values.as_ref() == other.values.as_ref()))
    }
}

impl MapScalar {
    /// returns a new [`MapScalar`]
    /// # Panics
    /// iff
    /// * the `dtype` is not `Map`
    /// * the child of the `dtype` is not equal to the `values`
    #[inline]
    pub fn new(dtype: ArrowDataType, values: Option<Box<dyn Array>>) -> Self {
        let inner_field = MapArray::try_get_field(&dtype).unwrap();
        let inner_dtype = inner_field.dtype();
        let (is_valid, values) = match values {
            Some(values) => {
                assert_eq!(inner_dtype, values.dtype());
                (true, values)
            },
            None => (false, new_empty_array(inner_dtype.clone())),
        };
        Self {
            values,
            is_valid,
            dtype,
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

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }
}
