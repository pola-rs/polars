use arrow::bitmap::MutableBitmap;
use polars_arrow::array::default_arrays::FromData;

use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use crate::series::IsSorted;

impl<T> ChunkFull<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn full(name: &str, value: T::Native, length: usize) -> Self {
        let data = vec![value; length];
        let mut out = ChunkedArray::from_vec(name, data);
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl<T> ChunkFullNull for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn full_null(name: &str, length: usize) -> Self {
        let arr = new_null_array(T::get_dtype().to_arrow(), length);
        unsafe { ChunkedArray::from_chunks(name, vec![arr]) }
    }
}
impl ChunkFull<bool> for BooleanChunked {
    fn full(name: &str, value: bool, length: usize) -> Self {
        let mut bits = MutableBitmap::with_capacity(length);
        bits.extend_constant(length, value);
        let mut out: BooleanChunked =
            (name, BooleanArray::from_data_default(bits.into(), None)).into();
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for BooleanChunked {
    fn full_null(name: &str, length: usize) -> Self {
        let arr = new_null_array(DataType::Boolean.to_arrow(), length);
        unsafe { BooleanChunked::from_chunks(name, vec![arr]) }
    }
}

impl<'a> ChunkFull<&'a str> for Utf8Chunked {
    fn full(name: &str, value: &'a str, length: usize) -> Self {
        let mut builder = Utf8ChunkedBuilder::new(name, length, length * value.len());

        for _ in 0..length {
            builder.append_value(value);
        }
        let mut out = builder.finish();
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for Utf8Chunked {
    fn full_null(name: &str, length: usize) -> Self {
        let arr = new_null_array(DataType::Utf8.to_arrow(), length);
        unsafe { Utf8Chunked::from_chunks(name, vec![arr]) }
    }
}

#[cfg(feature = "dtype-binary")]
impl<'a> ChunkFull<&'a [u8]> for BinaryChunked {
    fn full(name: &str, value: &'a [u8], length: usize) -> Self {
        let mut builder = BinaryChunkedBuilder::new(name, length, length * value.len());

        for _ in 0..length {
            builder.append_value(value);
        }
        let mut out = builder.finish();
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkFullNull for BinaryChunked {
    fn full_null(name: &str, length: usize) -> Self {
        let arr = new_null_array(DataType::Binary.to_arrow(), length);
        unsafe { BinaryChunked::from_chunks(name, vec![arr]) }
    }
}

impl ChunkFull<&Series> for ListChunked {
    fn full(name: &str, value: &Series, length: usize) -> ListChunked {
        let mut builder =
            get_list_builder(value.dtype(), value.len() * length, length, name).unwrap();
        for _ in 0..length {
            builder.append_series(value)
        }
        builder.finish()
    }
}

impl ChunkFullNull for ListChunked {
    fn full_null(name: &str, length: usize) -> ListChunked {
        ListChunked::full_null_with_dtype(name, length, &DataType::Null)
    }
}

impl ListChunked {
    pub fn full_null_with_dtype(name: &str, length: usize, inner_dtype: &DataType) -> ListChunked {
        let arr = new_null_array(
            ArrowDataType::LargeList(Box::new(ArrowField::new(
                "item",
                inner_dtype.to_arrow(),
                true,
            ))),
            length,
        );
        unsafe { ListChunked::from_chunks(name, vec![arr]) }
    }
}
#[cfg(feature = "dtype-struct")]
impl ChunkFullNull for StructChunked {
    fn full_null(name: &str, length: usize) -> StructChunked {
        let s = vec![Series::full_null("", length, &DataType::Null)];
        StructChunked::new_unchecked(name, &s)
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
