pub mod builder;
mod iterator;

pub use crate::prelude::*;
use crate::utils::arrow::array::ArrayData;
use arrow::array::{Array, ArrayRef, BooleanBufferBuilder, JsonEqual};
use arrow::bitmap::Bitmap;
use serde_json::Value;
use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ObjectArray<T>
where
    T: PolarsObject,
{
    pub(crate) values: Arc<Vec<T>>,
    pub(crate) null_bitmap: Option<Arc<Bitmap>>,
    pub(crate) null_count: usize,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

pub trait PolarsObject: Any + Debug + Clone + Send + Sync + Default + Display {
    fn type_name() -> &'static str;
}

impl<T> ObjectArray<T>
where
    T: PolarsObject,
{
    /// Get a reference to the underlying data
    pub fn values(&self) -> &Arc<Vec<T>> {
        &self.values
    }

    /// Get a value at a certain index location
    pub fn value(&self, index: usize) -> &T {
        &self.values[self.offset + index]
    }

    /// Get a value at a certain index location
    ///
    /// # Safety
    ///
    /// This does not any bound checks. The caller needs to ensure the index is within
    /// the size of the array.
    pub unsafe fn value_unchecked(&self, index: usize) -> &T {
        self.values.get_unchecked(index)
    }
}

impl<T> JsonEqual for ObjectArray<T>
where
    T: PolarsObject,
{
    fn equals_json(&self, _json: &[&Value]) -> bool {
        false
    }
}

impl<T> Array for ObjectArray<T>
where
    T: PolarsObject,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data(&self) -> &ArrayData {
        unimplemented!()
    }

    fn data_ref(&self) -> &ArrayData {
        unimplemented!()
    }

    fn data_type(&self) -> &ArrowDataType {
        unimplemented!()
    }

    fn slice(&self, offset: usize, length: usize) -> ArrayRef {
        let mut new = self.clone();
        let len = std::cmp::min(new.len - offset, length);

        new.len = length;
        new.offset = offset;
        new.null_count = if let Some(bitmap) = &new.null_bitmap {
            let no_null_count = bitmap.buffer_ref().count_set_bits_offset(offset, length);
            len.checked_sub(no_null_count).unwrap();
            0
        } else {
            0
        };
        Arc::new(new)
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn is_null(&self, index: usize) -> bool {
        match &self.null_bitmap {
            Some(b) => !b.is_set(index),
            None => false,
        }
    }

    fn is_valid(&self, index: usize) -> bool {
        match &self.null_bitmap {
            Some(b) => b.is_set(index),
            None => true,
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

impl<T> ObjectChunked<T>
where
    T: PolarsObject,
{
    ///
    /// # Safety
    ///
    /// No bounds checks
    pub unsafe fn get_as_any(&self, index: usize) -> &dyn Any {
        let chunks = self.downcast_chunks();
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        let arr = chunks.get_unchecked(chunk_idx);
        arr.value(idx)
    }
}
