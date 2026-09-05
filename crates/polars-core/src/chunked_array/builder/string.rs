use std::marker::PhantomData;

use polars_array::{PlBinaryViewArrayBuilder, PlUtf8ViewArray};

use super::*;

pub struct BinViewChunkedBuilder<T: ViewType + ?Sized> {
    /// The bytes, whatever they stand for: a `str` is its own bytes, and which of the two the
    /// chunk reads as is settled once, in `finish`.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Values shorter than a view's inline limit live in the view itself and longer ones in a
    /// buffer, so both sides of that boundary have to come back intact.
    #[test]
    fn strings_round_trip_either_side_of_the_inline_limit() {
        let mut builder = StringChunkedBuilder::new("s".into(), 4);
        builder.append_value("short");
        builder.append_value("a value far longer than the twelve byte inline limit");
        builder.append_null();
        builder.append_option(Some("é—non-ascii"));
        builder.append_option(None::<&str>);

        let ca = builder.finish();
        assert_eq!(ca.dtype(), &DataType::String);
        assert_eq!(
            ca.iter().collect::<Vec<_>>(),
            [
                Some("short"),
                Some("a value far longer than the twelve byte inline limit"),
                None,
                Some("é—non-ascii"),
                None,
            ],
        );
    }

    #[test]
    fn binary_round_trips_and_keeps_its_dtype() {
        let mut builder = BinaryChunkedBuilder::new("b".into(), 3);
        builder.append_value(b"ab".as_slice());
        builder.append_null();
        builder.append_value(b"a payload well past the inline limit".as_slice());

        let ca = builder.finish();
        assert_eq!(ca.dtype(), &DataType::Binary);
        assert_eq!(
            ca.iter().collect::<Vec<_>>(),
            [
                Some(b"ab".as_slice()),
                None,
                Some(b"a payload well past the inline limit".as_slice()),
            ],
        );
    }
}
