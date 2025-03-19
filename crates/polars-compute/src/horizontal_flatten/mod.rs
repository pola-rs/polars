#![allow(unsafe_op_in_unsafe_fn)]
use arrow::array::{
    Array, ArrayCollectIterExt, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeListArray,
    ListArray, NullArray, PrimitiveArray, StaticArray, StructArray, Utf8ViewArray,
};
use arrow::bitmap::Bitmap;
use arrow::datatypes::{ArrowDataType, PhysicalType};
use arrow::with_match_primitive_type_full;
use strength_reduce::StrengthReducedUsize;
mod struct_;

/// Low-level operation used by `concat_arr`. This should be called with the inner values array of
/// every FixedSizeList array.
///
/// # Safety
/// * `arrays` is non-empty
/// * `arrays` and `widths` have equal length
/// * All widths in `widths` are non-zero
/// * Every array `arrays[i]` has a length of either
///   * `widths[i] * output_height`
///   * `widths[i]` (this would be broadcasted)
/// * All arrays in `arrays` have the same type
pub unsafe fn horizontal_flatten_unchecked(
    arrays: &[Box<dyn Array>],
    widths: &[usize],
    output_height: usize,
) -> Box<dyn Array> {
    use PhysicalType::*;

    let dtype = arrays[0].dtype();

    match dtype.to_physical_type() {
        Null => Box::new(NullArray::new(
            dtype.clone(),
            output_height * widths.iter().copied().sum::<usize>(),
        )),
        Boolean => Box::new(horizontal_flatten_unchecked_impl_generic(
            &arrays
                .iter()
                .map(|x| x.as_any().downcast_ref::<BooleanArray>().unwrap().clone())
                .collect::<Vec<_>>(),
            widths,
            output_height,
            dtype,
        )),
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            Box::new(horizontal_flatten_unchecked_impl_generic(
                &arrays
                    .iter()
                    .map(|x| x.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap().clone())
                    .collect::<Vec<_>>(),
                widths,
                output_height,
                dtype
            ))
        }),
        LargeBinary => Box::new(horizontal_flatten_unchecked_impl_generic(
            &arrays
                .iter()
                .map(|x| {
                    x.as_any()
                        .downcast_ref::<BinaryArray<i64>>()
                        .unwrap()
                        .clone()
                })
                .collect::<Vec<_>>(),
            widths,
            output_height,
            dtype,
        )),
        Struct => Box::new(struct_::horizontal_flatten_unchecked(
            &arrays
                .iter()
                .map(|x| x.as_any().downcast_ref::<StructArray>().unwrap().clone())
                .collect::<Vec<_>>(),
            widths,
            output_height,
        )),
        LargeList => Box::new(horizontal_flatten_unchecked_impl_generic(
            &arrays
                .iter()
                .map(|x| x.as_any().downcast_ref::<ListArray<i64>>().unwrap().clone())
                .collect::<Vec<_>>(),
            widths,
            output_height,
            dtype,
        )),
        FixedSizeList => Box::new(horizontal_flatten_unchecked_impl_generic(
            &arrays
                .iter()
                .map(|x| {
                    x.as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap()
                        .clone()
                })
                .collect::<Vec<_>>(),
            widths,
            output_height,
            dtype,
        )),
        BinaryView => Box::new(horizontal_flatten_unchecked_impl_generic(
            &arrays
                .iter()
                .map(|x| {
                    x.as_any()
                        .downcast_ref::<BinaryViewArray>()
                        .unwrap()
                        .clone()
                })
                .collect::<Vec<_>>(),
            widths,
            output_height,
            dtype,
        )),
        Utf8View => Box::new(horizontal_flatten_unchecked_impl_generic(
            &arrays
                .iter()
                .map(|x| x.as_any().downcast_ref::<Utf8ViewArray>().unwrap().clone())
                .collect::<Vec<_>>(),
            widths,
            output_height,
            dtype,
        )),
        t => unimplemented!("horizontal_flatten not supported for data type {:?}", t),
    }
}

unsafe fn horizontal_flatten_unchecked_impl_generic<T>(
    arrays: &[T],
    widths: &[usize],
    output_height: usize,
    dtype: &ArrowDataType,
) -> T
where
    T: StaticArray,
{
    assert!(!arrays.is_empty());
    assert_eq!(widths.len(), arrays.len());

    debug_assert!(widths.iter().all(|x| *x > 0));
    debug_assert!(
        arrays
            .iter()
            .zip(widths)
            .all(|(arr, width)| arr.len() == output_height * *width || arr.len() == *width)
    );

    // We modulo the array length to support broadcasting.
    let lengths = arrays
        .iter()
        .map(|x| StrengthReducedUsize::new(x.len()))
        .collect::<Vec<_>>();
    let out_row_width: usize = widths.iter().cloned().sum();
    let out_len = out_row_width.checked_mul(output_height).unwrap();

    let mut col_idx = 0;
    let mut row_idx = 0;
    let mut until = widths[0];
    let mut outer_row_idx = 0;

    // We do `0..out_len` to get an `ExactSizeIterator`.
    (0..out_len)
        .map(|_| {
            let arr = arrays.get_unchecked(col_idx);
            let out = arr.get_unchecked(row_idx % *lengths.get_unchecked(col_idx));

            row_idx += 1;

            if row_idx == until {
                // Safety: All widths are non-zero so we only need to increment once.
                col_idx = if 1 + col_idx == widths.len() {
                    outer_row_idx += 1;
                    0
                } else {
                    1 + col_idx
                };
                row_idx = outer_row_idx * *widths.get_unchecked(col_idx);
                until = (1 + outer_row_idx) * *widths.get_unchecked(col_idx)
            }

            out
        })
        .collect_arr_trusted_with_dtype(dtype.clone())
}
