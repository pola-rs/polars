use arrow::bitmap::Bitmap;

use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use crate::series::IsSorted;

impl<T> ChunkFull<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn full(name: PlSmallStr, value: T::Native, length: usize) -> Self {
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
    fn full_null(name: PlSmallStr, length: usize) -> Self {
        let arr = PrimitiveArray::new_null(
            T::get_static_dtype().to_arrow(CompatLevel::newest()),
            length,
        );
        ChunkedArray::with_chunk(name, arr)
    }
}
impl ChunkFull<bool> for BooleanChunked {
    fn full(name: PlSmallStr, value: bool, length: usize) -> Self {
        let bits = Bitmap::new_with_value(value, length);
        let arr = BooleanArray::from_data_default(bits, None);
        let mut out = BooleanChunked::with_chunk(name, arr);
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for BooleanChunked {
    fn full_null(name: PlSmallStr, length: usize) -> Self {
        let arr = BooleanArray::new_null(ArrowDataType::Boolean, length);
        ChunkedArray::with_chunk(name, arr)
    }
}

impl<'a> ChunkFull<&'a str> for StringChunked {
    fn full(name: PlSmallStr, value: &'a str, length: usize) -> Self {
        let mut builder = StringChunkedBuilder::new(name, length);
        builder.chunk_builder.extend_constant(length, Some(value));
        let mut out = builder.finish();
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for StringChunked {
    fn full_null(name: PlSmallStr, length: usize) -> Self {
        let arr = Utf8ViewArray::new_null(DataType::String.to_arrow(CompatLevel::newest()), length);
        ChunkedArray::with_chunk(name, arr)
    }
}

impl<'a> ChunkFull<&'a [u8]> for BinaryChunked {
    fn full(name: PlSmallStr, value: &'a [u8], length: usize) -> Self {
        let mut builder = BinaryChunkedBuilder::new(name, length);
        builder.chunk_builder.extend_constant(length, Some(value));
        let mut out = builder.finish();
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for BinaryChunked {
    fn full_null(name: PlSmallStr, length: usize) -> Self {
        let arr =
            BinaryViewArray::new_null(DataType::Binary.to_arrow(CompatLevel::newest()), length);
        ChunkedArray::with_chunk(name, arr)
    }
}

impl<'a> ChunkFull<&'a [u8]> for BinaryOffsetChunked {
    fn full(name: PlSmallStr, value: &'a [u8], length: usize) -> Self {
        let mut mutable = MutableBinaryArray::with_capacities(length, length * value.len());
        mutable.extend_values(std::iter::repeat_n(value, length));
        let arr: BinaryArray<i64> = mutable.into();
        let mut out = ChunkedArray::with_chunk(name, arr);
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkFullNull for BinaryOffsetChunked {
    fn full_null(name: PlSmallStr, length: usize) -> Self {
        let arr = BinaryArray::<i64>::new_null(
            DataType::BinaryOffset.to_arrow(CompatLevel::newest()),
            length,
        );
        ChunkedArray::with_chunk(name, arr)
    }
}

impl ChunkFull<&Series> for ListChunked {
    fn full(name: PlSmallStr, value: &Series, length: usize) -> ListChunked {
        // Fast path: single-element primitive numeric series — build values buffer + offsets directly.
        if value.len() == 1 && value.dtype().is_primitive_numeric() || value.dtype().is_bool() {
            use arrow::datatypes::PhysicalType;
            use arrow::offset::{Offsets, OffsetsBuffer};
            use arrow::with_match_primitive_type;

            let chunk = value.rechunk();
            let arr = chunk.chunks()[0].as_ref();
            let arrow_dtype = arr.dtype().clone();

            let values_arr = match arrow_dtype.to_physical_type() {
                PhysicalType::Primitive(primitive) => {
                    with_match_primitive_type!(primitive, |$T| {
                        if arr.null_count() > 0 {
                            PrimitiveArray::<$T>::new_null(arrow_dtype.clone(), length).boxed()
                        } else {
                            let prim_arr =
                                arr.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                            let val = prim_arr.value(0);
                            PrimitiveArray::<$T>::from_vec(vec![val; length]).boxed()
                        }
                    })
                },
                PhysicalType::Boolean => {
                    if arr.null_count() > 0 {
                        BooleanArray::new_null(arrow_dtype.clone(), length).boxed()
                    } else {
                        let prim_arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
                        let val = prim_arr.value(0);
                        BooleanArray::full(length, val, arrow_dtype.clone()).boxed()
                    }
                },
                _ => unreachable!(),
            };

            let offsets: OffsetsBuffer<i64> =
                Offsets::try_from_lengths(std::iter::repeat_n(1usize, length))
                    .unwrap()
                    .into();

            let list_dtype = ArrowDataType::LargeList(Box::new(ArrowField::new(
                LIST_VALUES_NAME,
                arrow_dtype,
                true,
            )));
            let list_arr = LargeListArray::new(list_dtype, offsets, values_arr, None);

            // SAFETY: physical type matches the logical.
            return unsafe {
                ChunkedArray::from_chunks_and_dtype(
                    name,
                    vec![Box::new(list_arr)],
                    DataType::List(Box::new(value.dtype().clone())),
                )
            };
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
        let arr = FixedSizeListArray::new_null(
            ArrowDataType::FixedSizeList(
                Box::new(ArrowField::new(
                    LIST_VALUES_NAME,
                    inner_dtype.to_physical().to_arrow(CompatLevel::newest()),
                    true,
                )),
                width,
            ),
            length,
        );
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
        let arrow_dtype = ArrowDataType::FixedSizeList(
            Box::new(ArrowField::new(
                LIST_VALUES_NAME,
                dtype.to_physical().to_arrow(CompatLevel::newest()),
                true,
            )),
            width,
        );
        let value = value.rechunk().chunks()[0].clone();
        let arr = FixedSizeListArray::full(length, value, arrow_dtype);

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
        let arr: ListArray<i64> = ListArray::new_null(
            ArrowDataType::LargeList(Box::new(ArrowField::new(
                LIST_VALUES_NAME,
                inner_dtype.to_physical().to_arrow(CompatLevel::newest()),
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
