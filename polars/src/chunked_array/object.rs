pub use crate::prelude::*;
use arrow::array::{Array, ArrayDataRef, ArrayEqual, ArrayRef, JsonEqual};
use serde_json::Value;
use std::any::{Any, TypeId};
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct ObjectArray<T>
where
    T: Any + Debug + Clone + Send + Sync,
{
    inner: Vec<Option<T>>,
    null_count: usize,
    offset: usize,
    len: usize,
}

impl<T> ArrayEqual for ObjectArray<T>
where
    T: Any + Debug + Clone + Send + Sync,
{
    fn equals(&self, _other: &dyn Array) -> bool {
        false
    }

    fn range_equals(
        &self,
        _other: &dyn Array,
        _start_idx: usize,
        _end_idx: usize,
        _other_start_idx: usize,
    ) -> bool {
        false
    }
}

impl<T> JsonEqual for ObjectArray<T>
where
    T: Any + Debug + Clone + Send + Sync,
{
    fn equals_json(&self, _json: &[&Value]) -> bool {
        false
    }
}

impl<T> Array for ObjectArray<T>
where
    T: Any + Debug + Clone + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data(&self) -> ArrayDataRef {
        unimplemented!()
    }

    fn data_ref(&self) -> &ArrayDataRef {
        unimplemented!()
    }

    fn data_type(&self) -> &ArrowDataType {
        &ArrowDataType::Binary
    }

    fn slice(&self, _offset: usize, _length: usize) -> ArrayRef {
        unimplemented!()
    }

    fn len(&self) -> usize {
        unimplemented!()
    }

    fn is_empty(&self) -> bool {
        unimplemented!()
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn is_null(&self, index: usize) -> bool {
        if index > self.len {
            true
        } else {
            self.inner[index + self.offset].is_none()
        }
    }

    fn is_valid(&self, index: usize) -> bool {
        if index > self.len() {
            true
        } else {
            self.inner[index + self.offset].is_some()
        }
    }

    fn null_count(&self) -> usize {
        self.null_count
    }

    fn get_buffer_memory_size(&self) -> usize {
        unimplemented!()
    }

    fn get_array_memory_size(&self) -> usize {
        unimplemented!()
    }
}

pub trait Object: Array {
    /// For downcasting at runtime.
    fn as_any(&self) -> &dyn Any;

    fn type_name(&self) -> &'static str;

    fn type_id(&self) -> TypeId;
}

impl<T> Object for ObjectArray<T>
where
    T: Any + Debug + Clone + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }

    fn type_id(&self) -> TypeId {
        Object::as_any(self).type_id()
    }
}
