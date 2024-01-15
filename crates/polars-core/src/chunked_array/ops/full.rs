use arrow::bitmap::MutableBitmap;
use arrow::legacy::array::default_arrays::FromData;

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
        let arr = PrimitiveArray::new_null(T::get_dtype().to_arrow(true), length);
        ChunkedArray::with_chunk(name, arr)
    }
}
impl ChunkFull<bool> for BooleanChunked {
    fn full(name: &str, value: bool, length: usize) -> Self {
        let mut bits = MutableBitmap::with_capacity(length);
        bits.extend_constant(length, value);
        let arr = BooleanArray::from_data_default(bits.into(), None);
        let mut out = BooleanChunked::with_chunk(name, arr);
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for BooleanChunked {
    fn full_null(name: &str, length: usize) -> Self {
        let arr = BooleanArray::new_null(ArrowDataType::Boolean, length);
        ChunkedArray::with_chunk(name, arr)
    }
}

impl<'a> ChunkFull<&'a str> for StringChunked {
    fn full(name: &str, value: &'a str, length: usize) -> Self {
        let mut builder = StringChunkedBuilder::new(name, length);
        builder.chunk_builder.extend_constant(length, Some(value));
        let mut out = builder.finish();
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for StringChunked {
    fn full_null(name: &str, length: usize) -> Self {
        let arr = Utf8ViewArray::new_null(DataType::String.to_arrow(true), length);
        ChunkedArray::with_chunk(name, arr)
    }
}

impl<'a> ChunkFull<&'a [u8]> for BinaryChunked {
    fn full(name: &str, value: &'a [u8], length: usize) -> Self {

        let mut builder = BinaryChunkedBuilder::new(name, length);
        builder.chunk_builder.extend_constant(length, Some(value));
        let mut out = builder.finish();
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for BinaryChunked {
    fn full_null(name: &str, length: usize) -> Self {
        let arr = BinaryViewArray::new_null(DataType::Binary.to_arrow(true), length);
        ChunkedArray::with_chunk(name, arr)
    }
}

impl ChunkFull<&Series> for ListChunked {
    fn full(name: &str, value: &Series, length: usize) -> ListChunked {
        let mut builder =
            get_list_builder(value.dtype(), value.len() * length, length, name).unwrap();
        for _ in 0..length {
            builder.append_series(value).unwrap();
        }
        builder.finish()
    }
}

impl ChunkFullNull for ListChunked {
    fn full_null(name: &str, length: usize) -> ListChunked {
        ListChunked::full_null_with_dtype(name, length, &DataType::Null)
    }
}

#[cfg(feature = "dtype-array")]
impl ArrayChunked {
    pub fn full_null_with_dtype(
        name: &str,
        length: usize,
        inner_dtype: &DataType,
        width: usize,
    ) -> ArrayChunked {
        let arr = FixedSizeListArray::new_null(
            ArrowDataType::FixedSizeList(
                Box::new(ArrowField::new("item", inner_dtype.to_arrow(true), true)),
                width,
            ),
            length,
        );
        ChunkedArray::with_chunk(name, arr)
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkFull<&Series> for ArrayChunked {
    fn full(name: &str, value: &Series, length: usize) -> ArrayChunked {
        if !value.dtype().is_numeric() {
            todo!("Array only supports numeric data types");
        };
        let width = value.len();
        let values = value.tile(length);
        let values = values.chunks()[0].clone();
        let data_type = ArrowDataType::FixedSizeList(
            Box::new(ArrowField::new("item", values.data_type().clone(), true)),
            width,
        );
        let arr = FixedSizeListArray::new(data_type, values, None);
        ChunkedArray::with_chunk(name, arr)
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkFullNull for ArrayChunked {
    fn full_null(name: &str, length: usize) -> ArrayChunked {
        ArrayChunked::full_null_with_dtype(name, length, &DataType::Null, 0)
    }
}

impl ListChunked {
    pub fn full_null_with_dtype(name: &str, length: usize, inner_dtype: &DataType) -> ListChunked {
        let arr: ListArray<i64> = ListArray::new_null(
            ArrowDataType::LargeList(Box::new(ArrowField::new(
                "item",
                inner_dtype.to_physical().to_arrow(true),
                true,
            ))),
            length,
        );
        // SAFETY: physical type matches the logical.
        unsafe {
            ChunkedArray::from_chunks_and_dtype(
                name,
                vec![Box::new(arr)],
                DataType::List(Box::new(inner_dtype.clone())),
            )
        }
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
