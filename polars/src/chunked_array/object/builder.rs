use super::*;
use crate::prelude::*;
use arrow::bitmap::Bitmap;
use arrow::util::bit_util::count_set_bits;
use std::marker::PhantomData;
use std::sync::Arc;

impl<T> ObjectChunkedBuilder<T>
where
    T: Any + Debug + Clone + Send + Sync + Default,
{
    pub fn new(name: &str, capacity: usize) -> Self {
        ObjectChunkedBuilder {
            field: Field::new(name, ArrowDataType::Binary, true),
            values: Vec::with_capacity(capacity),
            bitmask_builder: BooleanBufferBuilder::new(capacity),
        }
    }

    /// Appends a value of type `T` into the builder
    pub fn append_value(&mut self, v: T) {
        self.values.push(v);
        self.bitmask_builder.append(true).unwrap();
    }

    /// Appends a null slot into the builder
    pub fn append_null(&mut self) {
        self.values.push(T::default());
        self.bitmask_builder.append(false).unwrap();
    }

    pub fn append_option(&mut self, opt: Option<T>) {
        match opt {
            Some(s) => self.append_value(s),
            None => self.append_null(),
        }
    }

    pub fn finish(mut self) -> ObjectChunked {
        let null_bit_buffer = self.bitmask_builder.finish();
        let null_count = count_set_bits(null_bit_buffer.data());
        let null_bitmap = Bitmap::from(null_bit_buffer);
        let null_bitmap = match null_count {
            0 => None,
            _ => Some(null_bitmap),
        };

        let len = self.values.len();

        let arr = Arc::new(ObjectArray {
            values: Arc::new(self.values),
            null_bitmap: Arc::new(null_bitmap),
            null_count,
            offset: 0,
            len,
        });
        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
        }
    }
}
