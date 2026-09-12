use std::marker::PhantomData;

use polars_array::{PlBinaryViewArrayBuilder, PlUtf8ViewArray};

use super::*;

pub struct BinViewChunkedBuilder<T: ViewType + ?Sized> {
    /// The bytes, whatever they stand for; which of the two the chunk reads as is settled once.
    chunk_builder: PlBinaryViewArrayBuilder,
    pub(crate) field: FieldRef,
    _type: PhantomData<fn() -> Box<T>>,
}

impl<T: ViewType + ?Sized> Clone for BinViewChunkedBuilder<T> {
    fn clone(&self) -> Self {
        Self {
            chunk_builder: self.chunk_builder.clone(),
            field: self.field.clone(),
            _type: PhantomData,
        }
    }
}

pub type StringChunkedBuilder = BinViewChunkedBuilder<str>;
pub type BinaryChunkedBuilder = BinViewChunkedBuilder<[u8]>;

impl<T: ViewType + ?Sized> BinViewChunkedBuilder<T> {
    /// Create a new BinViewChunkedBuilder
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of string elements in the final array.
    pub fn new(name: PlSmallStr, capacity: usize) -> Self {
        Self {
            chunk_builder: PlBinaryViewArrayBuilder::with_capacity(capacity),
            field: Arc::new(Field::new(name, DataType::from_arrow_dtype(&T::DATA_TYPE))),
            _type: PhantomData,
        }
    }

    /// Appends a value of type `T` into the builder
    #[inline]
    pub fn append_value<S: AsRef<T>>(&mut self, v: S) {
        self.chunk_builder.push_value(v.as_ref().to_bytes());
    }

    /// Appends a value of type `T`, leaving the validity mask untouched.
    ///
    /// A builder every value is pushed onto this way keeps no mask at all, and the array it
    /// freezes into holds none — which is the right answer for a loop that produces no null. See
    /// [`PlBinaryViewArrayBuilder::push_value_ignore_validity`].
    ///
    /// A builder must not see both this and [`append_value`](Self::append_value) or
    /// [`append_null`](Self::append_null): the bits would then stand for some of the elements and
    /// not others.
    #[inline]
    pub fn append_value_ignore_validity<S: AsRef<T>>(&mut self, v: S) {
        self.chunk_builder
            .push_value_ignore_validity(v.as_ref().to_bytes());
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.chunk_builder.push_null()
    }

    #[inline]
    pub fn append_option<S: AsRef<T>>(&mut self, opt: Option<S>) {
        match opt {
            Some(v) => self.append_value(v),
            None => self.append_null(),
        }
    }
}

impl StringChunkedBuilder {
    pub fn finish(self) -> StringChunked {
        // SAFETY: every value went in through `AsRef<str>`, so the bytes are valid UTF-8.
        let arr = unsafe { PlUtf8ViewArray::from_binview_unchecked(self.chunk_builder.freeze()) };
        ChunkedArray::new_with_compute_len(self.field, vec![arr.into_boxed()])
    }
}
impl BinaryChunkedBuilder {
    pub fn finish(self) -> BinaryChunked {
        let arr = self.chunk_builder.freeze();
        ChunkedArray::new_with_compute_len(self.field, vec![arr.into_boxed()])
    }
}
