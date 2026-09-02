use arrow::bitmap::Bitmap;
use polars_array::builder::full_null_like;

use crate::chunked_array::builder::get_list_builder;
use crate::chunked_array::new_empty_chunk;
use crate::prelude::*;
use crate::series::IsSorted;

// A `ChunkedArray` of one value repeated is the scalar representation of `polars-array`: every
// `full` here is `O(1)` in both time and memory, however long the result is.

impl<T> ChunkFull<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn full(name: PlSmallStr, value: T::Native, length: usize) -> Self {
        let mut out = ChunkedArray::with_chunk(name, PlPrimitiveArray::new_scalar(value, length));
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl<T> ChunkFullNull for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn full_null(name: PlSmallStr, length: usize) -> Self {
        ChunkedArray::with_chunk(name, T::full_null_array(length))
    }
}

impl ChunkFull<bool> for BooleanChunked {
    fn full(name: PlSmallStr, value: bool, length: usize) -> Self {
        let mut out = BooleanChunked::with_chunk(name, PlBooleanArray::new_scalar(value, length));
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for BooleanChunked {
    fn full_null(name: PlSmallStr, length: usize) -> Self {
        ChunkedArray::with_chunk(name, BooleanType::full_null_array(length))
    }
}

impl<'a> ChunkFull<&'a str> for StringChunked {
    fn full(name: PlSmallStr, value: &'a str, length: usize) -> Self {
        let mut out = StringChunked::with_chunk(name, PlUtf8ViewArray::new_scalar(value, length));
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for StringChunked {
    fn full_null(name: PlSmallStr, length: usize) -> Self {
        ChunkedArray::with_chunk(name, StringType::full_null_array(length))
    }
}

impl<'a> ChunkFull<&'a [u8]> for BinaryChunked {
    fn full(name: PlSmallStr, value: &'a [u8], length: usize) -> Self {
        let mut out = BinaryChunked::with_chunk(name, PlBinaryViewArray::new_scalar(value, length));
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for BinaryChunked {
    fn full_null(name: PlSmallStr, length: usize) -> Self {
        ChunkedArray::with_chunk(name, BinaryType::full_null_array(length))
    }
}

impl<'a> ChunkFull<&'a [u8]> for BinaryOffsetChunked {
    fn full(name: PlSmallStr, value: &'a [u8], length: usize) -> Self {
        let mut out =
            BinaryOffsetChunked::with_chunk(name, PlBinaryArray::new_scalar(value, length));
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for BinaryOffsetChunked {
    fn full_null(name: PlSmallStr, length: usize) -> Self {
        ChunkedArray::with_chunk(name, BinaryOffsetType::full_null_array(length))
    }
}

impl ChunkFull<&Series> for ListChunked {
    fn full(name: PlSmallStr, value: &Series, length: usize) -> ListChunked {
        if value.len() == 1 && !value.dtype().is_nested() {
            let out = value
                .new_from_index(0, length)
                .reshape_list(&[
                    ReshapeDimension::Infer,
                    ReshapeDimension::Specified(Dimension::new(1)),
                ])
                .unwrap();
            return out.list().unwrap().clone();
        }

        let mut builder = get_list_builder(value.dtype(), value.len() * length, length, name);
        for _ in 0..length {
            builder.append_series(value).unwrap();
        }
        builder.finish()
    }
}

impl ChunkFullNull for ListChunked {
    fn full_null(name: PlSmallStr, length: usize) -> ListChunked {
        ListChunked::full_null_with_dtype(name, length, &DataType::Null)
    }
}

#[cfg(feature = "dtype-array")]
impl ArrayChunked {
    pub fn full_null_with_dtype(
        name: PlSmallStr,
        length: usize,
        inner_dtype: &DataType,
        width: usize,
    ) -> ArrayChunked {
        // An element of a null list is as wide as any other, so the one row the values stand for
        // is `width` nulls of the inner type.
        let values = full_null_like(&*new_empty_chunk(inner_dtype), width);
        let arr = PlFixedSizeListArray::new_full_null(values, length);

        // SAFETY: physical type matches the logical.
        unsafe {
            ChunkedArray::from_chunks_and_dtype(
                name,
                vec![Box::new(arr)],
                DataType::Array(Box::new(inner_dtype.clone()), width),
            )
        }
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkFull<&Series> for ArrayChunked {
    fn full(name: PlSmallStr, value: &Series, length: usize) -> ArrayChunked {
        let width = value.len();
        let dtype = value.dtype();
        let values = value.rechunk().chunks()[0].clone();
        let arr = PlFixedSizeListArray::new_scalar(values, length);

        // SAFETY: physical type matches the logical.
        unsafe {
            ChunkedArray::from_chunks_and_dtype(
                name,
                vec![Box::new(arr)],
                DataType::Array(Box::new(dtype.clone()), width),
            )
        }
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkFullNull for ArrayChunked {
    fn full_null(name: PlSmallStr, length: usize) -> ArrayChunked {
        ArrayChunked::full_null_with_dtype(name, length, &DataType::Null, 0)
    }
}

impl ListChunked {
    pub fn full_null_with_dtype(
        name: PlSmallStr,
        length: usize,
        inner_dtype: &DataType,
    ) -> ListChunked {
        // Every element is an empty list, so the values are only there to carry the inner shape.
        let arr = PlListArray::new_full_null(new_empty_chunk(inner_dtype), length);

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
    fn full_null(name: PlSmallStr, length: usize) -> StructChunked {
        StructChunked::from_series(name, length, [].iter())
            .unwrap()
            .with_outer_validity(Some(Bitmap::new_zeroed(length)))
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkFull<T> for ObjectChunked<T> {
    fn full(name: PlSmallStr, value: T, length: usize) -> Self
    where
        Self: Sized,
    {
        use crate::chunked_array::object::registry::run_with_gil;

        run_with_gil(|| {
            let mut ca: Self = (0..length).map(|_| Some(value.clone())).collect();
            ca.rename(name);
            ca
        })
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkFullNull for ObjectChunked<T> {
    fn full_null(name: PlSmallStr, length: usize) -> ObjectChunked<T> {
        use crate::chunked_array::object::registry::run_with_gil;

        run_with_gil(|| {
            let mut ca: Self = (0..length).map(|_| None).collect();
            ca.rename(name);
            ca
        })
    }
}
