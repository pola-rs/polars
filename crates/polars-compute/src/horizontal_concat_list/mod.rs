#![allow(unsafe_op_in_unsafe_fn)]
use arrow::array::{
    Array, ArrayCollectIterExt, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeListArray,
    ListArray, NullArray, PrimitiveArray, StaticArray, Utf8ViewArray,
};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::{ArrowDataType, PhysicalType};
use arrow::offset::{Offsets, OffsetsBuffer};
use arrow::trusted_len::TrustMyLength;
use arrow::with_match_primitive_type_full;

mod struct_;

/// A contiguous run of inner values taken from a single input column: values
/// `[start, end)` of column `col`. Runs are stored in row-major output order so
/// they can be replayed cheaply (e.g. once per struct field).
type Span = (usize, usize, usize);

/// Horizontally concatenate variable-length list columns row-wise.
///
/// For every output row `r`, the resulting list is the concatenation of
/// `arrays[0][r] ++ arrays[1][r] ++ .. ++ arrays[n-1][r]`. If any input list at
/// row `r` is null, the whole output row is null (null propagation).
///
/// This is the list-analogue of [`crate::horizontal_flatten`]: it operates
/// directly on the underlying Arrow arrays with monomorphized per-type kernels
/// instead of going through `amortized_iter` / `Series::append`.
///
/// # Safety
/// * `arrays` is non-empty.
/// * All arrays in `arrays` have the same inner values type.
/// * Every array has a length of either `output_height` (the maximum length) or
///   `1` (which is broadcast across all output rows).
pub unsafe fn horizontal_concat_list_unchecked(arrays: &[ListArray<i64>]) -> ListArray<i64> {
    assert!(!arrays.is_empty());

    let n_cols = arrays.len();
    // The output height is the common (broadcast) length: the length of the
    // non-unit columns, or 1 when every column has unit length. Using the
    // non-unit max (rather than the plain max) keeps empty inputs empty even
    // when unit-length columns are broadcast alongside them.
    let output_height = arrays
        .iter()
        .map(|a| a.len())
        .filter(|&l| l != 1)
        .max()
        .unwrap_or(1);

    let offsets: Vec<&OffsetsBuffer<i64>> = arrays.iter().map(|a| a.offsets()).collect();
    let validities: Vec<Option<&Bitmap>> = arrays.iter().map(|a| a.validity()).collect();
    // A unit-length column is broadcast (unless the output itself has height 1).
    let is_broadcast: Vec<bool> = arrays
        .iter()
        .map(|a| a.len() == 1 && output_height != 1)
        .collect();

    let has_validity = validities.iter().any(|v| v.is_some());
    let mut out_validity = has_validity.then(|| MutableBitmap::with_capacity(output_height));

    let mut row_lengths: Vec<usize> = Vec::with_capacity(output_height);
    let mut spans: Vec<Span> = Vec::new();
    let mut out_values_len = 0usize;

    for r in 0..output_height {
        // Null propagation: the output row is valid only if every input row is valid.
        let mut valid = true;
        for j in 0..n_cols {
            let row = if is_broadcast[j] { 0 } else { r };
            if let Some(v) = validities[j] {
                if !v.get_bit_unchecked(row) {
                    valid = false;
                    break;
                }
            }
        }

        if let Some(mb) = out_validity.as_mut() {
            mb.push(valid);
        }

        if !valid {
            row_lengths.push(0);
            continue;
        }

        let mut row_len = 0usize;
        for j in 0..n_cols {
            let row = if is_broadcast[j] { 0 } else { r };
            let (start, end) = offsets[j].start_end(row);
            if end > start {
                spans.push((j, start, end));
                row_len += end - start;
            }
        }
        row_lengths.push(row_len);
        out_values_len += row_len;
    }

    let inner_arrays: Vec<Box<dyn Array>> = arrays.iter().map(|a| a.values().clone()).collect();
    let out_inner = build_inner_values(&inner_arrays, &spans, out_values_len);

    let out_offsets: OffsetsBuffer<i64> = Offsets::try_from_lengths(row_lengths.into_iter())
        .expect("offset overflow in concat_list")
        .into();

    ListArray::<i64>::new(
        arrays[0].dtype().clone(),
        out_offsets,
        out_inner,
        out_validity.map(|mb| mb.freeze()),
    )
}

fn downcast_all<T: Array + Clone + 'static>(values: &[Box<dyn Array>]) -> Vec<T> {
    values
        .iter()
        .map(|x| x.as_any().downcast_ref::<T>().unwrap().clone())
        .collect()
}

/// Build the concatenated inner values array by dispatching on the physical type.
/// Reused recursively for the fields of a struct inner type.
pub(super) unsafe fn build_inner_values(
    values: &[Box<dyn Array>],
    spans: &[Span],
    out_values_len: usize,
) -> Box<dyn Array> {
    use PhysicalType::*;

    let dtype = values[0].dtype();

    match dtype.to_physical_type() {
        Null => Box::new(NullArray::new(dtype.clone(), out_values_len)),
        Boolean => Box::new(build_leaf::<BooleanArray>(
            &downcast_all(values),
            spans,
            out_values_len,
            dtype.clone(),
        )),
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            Box::new(build_leaf::<PrimitiveArray<$T>>(
                &downcast_all(values),
                spans,
                out_values_len,
                dtype.clone(),
            ))
        }),
        LargeBinary => Box::new(build_leaf::<BinaryArray<i64>>(
            &downcast_all(values),
            spans,
            out_values_len,
            dtype.clone(),
        )),
        LargeList => Box::new(build_leaf::<ListArray<i64>>(
            &downcast_all(values),
            spans,
            out_values_len,
            dtype.clone(),
        )),
        FixedSizeList => Box::new(build_leaf::<FixedSizeListArray>(
            &downcast_all(values),
            spans,
            out_values_len,
            dtype.clone(),
        )),
        BinaryView => Box::new(build_leaf::<BinaryViewArray>(
            &downcast_all(values),
            spans,
            out_values_len,
            dtype.clone(),
        )),
        Utf8View => Box::new(build_leaf::<Utf8ViewArray>(
            &downcast_all(values),
            spans,
            out_values_len,
            dtype.clone(),
        )),
        Struct => Box::new(struct_::build_struct_values(
            &downcast_all(values),
            spans,
            out_values_len,
        )),
        t => unimplemented!("concat_list not supported for data type {:?}", t),
    }
}

/// Monomorphized leaf builder: replays `spans` over the per-column values arrays,
/// gathering each element (preserving inner-element nulls, since `get_unchecked`
/// yields `Option<ValueT>`).
unsafe fn build_leaf<T: StaticArray>(
    values: &[T],
    spans: &[Span],
    out_values_len: usize,
    dtype: ArrowDataType,
) -> T {
    let iter = spans.iter().flat_map(|&(col, start, end)| {
        let arr = &values[col];
        (start..end).map(move |k| arr.get_unchecked(k))
    });
    TrustMyLength::new(iter, out_values_len).collect_arr_trusted_with_dtype(dtype)
}

/// Build the element-level validity of a concatenated inner array from `spans`,
/// combining the per-column validities in row-major order. Returns `None` when
/// no input column carries a validity bitmap.
pub(super) unsafe fn build_values_validity(
    per_col_validity: &[Option<&Bitmap>],
    spans: &[Span],
    out_values_len: usize,
) -> Option<Bitmap> {
    if per_col_validity.iter().all(|v| v.is_none()) {
        return None;
    }

    let mut mb = MutableBitmap::with_capacity(out_values_len);
    for &(col, start, end) in spans {
        match per_col_validity[col] {
            Some(v) => {
                for k in start..end {
                    mb.push(v.get_bit_unchecked(k));
                }
            },
            None => mb.extend_constant(end - start, true),
        }
    }
    Some(mb.freeze())
}
