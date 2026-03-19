use std::any::Any;

use super::Scalar;
use crate::array::*;
use crate::datatypes::ArrowDataType;
use crate::offset::Offset;

/// The scalar equivalent of [`ListArray`]. Like [`ListArray`], this struct holds a dynamically-typed
/// [`Array`]. The only difference is that this has only one element.
#[derive(Debug, Clone)]
pub struct ListScalar<O: Offset> {
    values: Box<dyn Array>,
    is_valid: bool,
    phantom: std::marker::PhantomData<O>,
    dtype: ArrowDataType,
}

impl<O: Offset> PartialEq for ListScalar<O> {
    fn eq(&self, other: &Self) -> bool {
        (self.dtype == other.dtype)
            && (self.is_valid == other.is_valid)
            && ((!self.is_valid) | (self.values.as_ref() == other.values.as_ref()))
    }
}

impl<O: Offset> ListScalar<O> {
    /// returns a new [`ListScalar`]
    /// # Panics
    /// iff
    /// * the `dtype` is not `List` or `LargeList` (depending on this scalar's offset `O`)
    /// * the child of the `dtype` is not equal to the `values`
    #[inline]
    pub fn new(dtype: ArrowDataType, values: Option<Box<dyn Array>>) -> Self {
        let inner_dtype = ListArray::<O>::get_child_type(&dtype);
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
            phantom: std::marker::PhantomData,
            dtype,
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

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }
}
