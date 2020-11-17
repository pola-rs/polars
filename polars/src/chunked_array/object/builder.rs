use super::*;
use crate::prelude::*;
use crate::utils::get_iter_capacity;
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

    pub fn append_value_from_any(&mut self, v: &dyn Any) -> Result<()> {
        match v.downcast_ref::<T>() {
            None => Err(PolarsError::DataTypeMisMatch(
                "cannot downcast any in ObjectBuilder".into(),
            )),
            Some(v) => {
                self.append_value(v.clone());
                Ok(())
            }
        }
    }

    pub fn append_option(&mut self, opt: Option<T>) {
        match opt {
            Some(s) => self.append_value(s),
            None => self.append_null(),
        }
    }

    pub fn finish(mut self) -> ObjectChunked<T> {
        let null_bit_buffer = self.bitmask_builder.finish();
        let null_count = count_set_bits(null_bit_buffer.data());
        let null_bitmap = Bitmap::from(null_bit_buffer);
        let null_bitmap = match null_count {
            0 => None,
            _ => Some(null_bitmap),
        };

        let len = self.values.len();

        let arr = Arc::new(ObjectArrayTyped {
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

impl<T> Default for ObjectChunkedBuilder<T>
where
    T: Any + Debug + Clone + Send + Sync + Default,
{
    fn default() -> Self {
        ObjectChunkedBuilder::new("", 0)
    }
}

// pub trait AnyObjectBuilder {
//     fn append_null(&mut self);
//
//     fn append_value(&mut self, v: &dyn Any) -> Result<()>;
//
//     // trait object cannot take self by value
//     fn finish(&mut self) -> ObjectChunked<T>;
// }
//
// impl<T> AnyObjectBuilder for ObjectChunkedBuilder<T>
// where
//     T: Any + Debug + Clone + Send + Sync + Default,
// {
//     fn append_null(&mut self) {
//         ObjectChunkedBuilder::append_null(self)
//     }
//
//     fn append_value(&mut self, v: &dyn Any) -> Result<()> {
//         self.append_value_from_any(v)
//     }
//
//     fn finish(&mut self) -> ObjectChunked<T> {
//         let builder = std::mem::take(self);
//         ObjectChunkedBuilder::finish(builder)
//     }
// }

impl<T>
    NewChunkedArray<
        ObjectType<T>, // pub trait AnyObjectBuilder {
        //     fn append_null(&mut self);
        //
        //     fn append_value(&mut self, v: &dyn Any) -> Result<()>;
        //
        //     // trait object cannot take self by value
        //     fn finish(&mut self) -> ObjectChunked<T>;
        // }
        //
        // impl<T> AnyObjectBuilder for ObjectChunkedBuilder<T>
        // where
        //     T: Any + Debug + Clone + Send + Sync + Default,
        // {
        //     fn append_null(&mut self) {
        //         ObjectChunkedBuilder::append_null(self)
        //     }
        //
        //     fn append_value(&mut self, v: &dyn Any) -> Result<()> {
        //         self.append_value_from_any(v)
        //     }
        //
        //     fn finish(&mut self) -> ObjectChunked<T> {
        //         let builder = std::mem::take(self);
        //         ObjectChunkedBuilder::finish(builder)
        //     }
        // }
        T,
    > for ObjectChunked<T>
where
    T: Any + Debug + Clone + Send + Sync + Default,
{
    fn new_from_slice(name: &str, v: &[T]) -> Self {
        Self::new_from_iter(name, v.iter().cloned())
    }

    fn new_from_opt_slice(name: &str, opt_v: &[Option<T>]) -> Self {
        let mut builder = ObjectChunkedBuilder::<T>::new(name, opt_v.len());
        opt_v
            .iter()
            .cloned()
            .for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    fn new_from_opt_iter(name: &str, it: impl Iterator<Item = Option<T>>) -> ObjectChunked<T> {
        let mut builder = ObjectChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn new_from_iter(name: &str, it: impl Iterator<Item = T>) -> ObjectChunked<T> {
        let mut builder = ObjectChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|v| builder.append_value(v));
        builder.finish()
    }
}
