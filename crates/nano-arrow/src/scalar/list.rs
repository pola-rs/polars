use std::any::Any;

use crate::{array::*, datatypes::DataType, offset::Offset};

use super::Scalar;

/// The scalar equivalent of [`ListArray`]. Like [`ListArray`], this struct holds a dynamically-typed
/// [`Array`]. The only difference is that this has only one element.
#[derive(Debug, Clone)]
pub struct ListScalar<O: Offset> {
    values: Box<dyn Array>,
    is_valid: bool,
    phantom: std::marker::PhantomData<O>,
    data_type: DataType,
}

impl<O: Offset> PartialEq for ListScalar<O> {
    fn eq(&self, other: &Self) -> bool {
        (self.data_type == other.data_type)
            && (self.is_valid == other.is_valid)
            && ((!self.is_valid) | (self.values.as_ref() == other.values.as_ref()))
    }
}

impl<O: Offset> ListScalar<O> {
    /// returns a new [`ListScalar`]
    /// # Panics
    /// iff
    /// * the `data_type` is not `List` or `LargeList` (depending on this scalar's offset `O`)
    /// * the child of the `data_type` is not equal to the `values`
    #[inline]
    pub fn new(data_type: DataType, values: Option<Box<dyn Array>>) -> Self {
        let inner_data_type = ListArray::<O>::get_child_type(&data_type);
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
            phantom: std::marker::PhantomData,
            data_type,
        }
    }

    /// The values of the [`ListScalar`]
    pub fn values(&self) -> &Box<dyn Array> {
        &self.values
    }
}

impl<O: Offset> Scalar for ListScalar<O> {
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
