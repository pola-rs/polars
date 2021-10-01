use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use crate::utils::NoNull;

impl<T> ChunkFull<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn full(name: &str, value: T::Native, length: usize) -> Self {
        let mut ca = (0..length)
            .map(|_| value)
            .collect::<NoNull<ChunkedArray<T>>>()
            .into_inner();
        ca.rename(name);
        ca
    }
}

impl<T> ChunkFullNull for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn full_null(name: &str, length: usize) -> Self {
        let mut ca = (0..length).map(|_| None).collect::<Self>();
        ca.rename(name);
        ca
    }
}
impl ChunkFull<bool> for BooleanChunked {
    fn full(name: &str, value: bool, length: usize) -> Self {
        let mut ca = (0..length).map(|_| value).collect::<BooleanChunked>();
        ca.rename(name);
        ca
    }
}

impl ChunkFullNull for BooleanChunked {
    fn full_null(name: &str, length: usize) -> Self {
        let mut ca = (0..length).map(|_| None).collect::<Self>();
        ca.rename(name);
        ca
    }
}

impl<'a> ChunkFull<&'a str> for Utf8Chunked {
    fn full(name: &str, value: &'a str, length: usize) -> Self {
        let mut builder = Utf8ChunkedBuilder::new(name, length, length * value.len());

        for _ in 0..length {
            builder.append_value(value);
        }
        builder.finish()
    }
}

impl ChunkFullNull for Utf8Chunked {
    fn full_null(name: &str, length: usize) -> Self {
        let mut ca = (0..length)
            .map::<Option<String>, _>(|_| None)
            .collect::<Self>();
        ca.rename(name);
        ca
    }
}

impl ChunkFull<&Series> for ListChunked {
    fn full(name: &str, value: &Series, length: usize) -> ListChunked {
        let mut builder = get_list_builder(value.dtype(), value.len() * length, length, name);
        for _ in 0..length {
            builder.append_series(value)
        }
        builder.finish()
    }
}

impl ChunkFullNull for ListChunked {
    fn full_null(name: &str, length: usize) -> ListChunked {
        let mut builder = ListBooleanChunkedBuilder::new(name, length, 0);
        for _ in 0..length {
            builder.append_null();
        }
        builder.finish()
    }
}

#[cfg(feature = "dtype-categorical")]
impl ChunkFullNull for CategoricalChunked {
    fn full_null(name: &str, length: usize) -> CategoricalChunked {
        use crate::chunked_array::categorical::CategoricalChunkedBuilder;
        let mut builder = CategoricalChunkedBuilder::new(name, length);
        let iter = (0..length).map(|_| None);
        builder.from_iter(iter);
        builder.finish()
    }
}

impl ListChunked {
    pub(crate) fn full_null_with_dtype(name: &str, length: usize, dt: &DataType) -> ListChunked {
        let mut builder = get_list_builder(dt, 0, length, name);
        for _ in 0..length {
            builder.append_null();
        }
        builder.finish()
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkFullNull for ObjectChunked<T> {
    fn full_null(name: &str, length: usize) -> ObjectChunked<T> {
        let mut ca: Self = (0..length).map(|_| None).collect();
        ca.rename(name);
        ca
    }
}
