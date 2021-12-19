use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use crate::utils::NoNull;
use arrow::bitmap::MutableBitmap;
use polars_arrow::array::default_arrays::FromData;

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
        let arr = new_null_array(T::get_dtype().to_arrow(), length).into();
        ChunkedArray::new_from_chunks(name, vec![arr])
    }
}
impl ChunkFull<bool> for BooleanChunked {
    fn full(name: &str, value: bool, length: usize) -> Self {
        let mut bits = MutableBitmap::with_capacity(length);
        bits.extend_constant(length, value);
        (name, BooleanArray::from_data_default(bits.into(), None)).into()
    }
}

impl ChunkFullNull for BooleanChunked {
    fn full_null(name: &str, length: usize) -> Self {
        let arr = new_null_array(DataType::Boolean.to_arrow(), length).into();
        BooleanChunked::new_from_chunks(name, vec![arr])
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
        let arr = new_null_array(DataType::Utf8.to_arrow(), length).into();
        Utf8Chunked::new_from_chunks(name, vec![arr])
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
        ListChunked::full_null_with_dtype(name, length, &DataType::Boolean)
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
    pub(crate) fn full_null_with_dtype(
        name: &str,
        length: usize,
        inner_dtype: &DataType,
    ) -> ListChunked {
        let arr = new_null_array(
            ArrowDataType::LargeList(Box::new(ArrowField::new(
                "item",
                inner_dtype.to_arrow(),
                true,
            ))),
            length,
        )
        .into();
        ListChunked::new_from_chunks(name, vec![arr])
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkFull<T> for ObjectChunked<T> {
    fn full(name: &str, value: T, length: usize) -> Self
    where
        Self: Sized,
    {
        let mut ca: Self = (0..length).map(|_| Some(value.clone())).collect();
        ca.rename(name);
        ca
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
