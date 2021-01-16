pub mod builder;
use super::kernels::utils::{count_set_bits, count_set_bits_offset};
pub use crate::prelude::*;
use arrow::array::{Array, ArrayDataRef, ArrayRef, BooleanBufferBuilder, JsonEqual};
use arrow::bitmap::Bitmap;
use serde_json::Value;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ObjectArray<T>
where
    T: Any + Debug + Clone + Send + Sync,
{
    values: Arc<Vec<T>>,
    null_bitmap: Option<Arc<Bitmap>>,
    null_count: usize,
    offset: usize,
    len: usize,
}

impl<T> ObjectArray<T>
where
    T: Any + Debug + Clone + Send + Sync,
{
    pub fn value(&self, index: usize) -> &T {
        &self.values[self.offset + index]
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
        // todo! we hijack the binary type for this. If we actually implement binary we need to find
        // another solution
        &ArrowDataType::Binary
    }

    fn slice(&self, offset: usize, length: usize) -> ArrayRef {
        let mut new = self.clone();
        let len = std::cmp::min(new.len - offset, length);

        new.len = length;
        new.offset = offset;
        new.null_count = if let Some(bitmap) = &new.null_bitmap {
            let valid_bits = bitmap.buffer_ref().as_slice();
            len.checked_sub(count_set_bits_offset(valid_bits, offset, length))
                .unwrap();
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
    T: Any + Debug + Clone + Send + Sync + Default,
{
    pub fn get_as_any(&self, index: usize) -> &dyn Any {
        let chunks = self.downcast_chunks();
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        let arr = unsafe { *chunks.get_unchecked(chunk_idx) };
        arr.value(idx)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn object_series() {
        let s = ObjectChunked::new_from_opt_slice("foo", &[Some(1), None, Some(3)]);
        assert_eq!(
            Vec::from(s.is_null()),
            &[Some(false), Some(true), Some(false)]
        )
    }
}
