#![allow(unsafe_op_in_unsafe_fn)]
use arrow::array::{
    Array, ArrayCollectIterExt, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeListArray,
    ListArray, NullArray, PrimitiveArray, StaticArray, Utf8ViewArray,
};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::{ArrowDataType, PhysicalType};
use arrow::offset::{Offsets, OffsetsBuffer};
use arrow::trusted_len::TrustMyLength;
use arrow::types::NativeType;
use arrow::with_match_primitive_type_full;
use polars_error::PolarsResult;

mod struct_;

/// A contiguous run of inner values taken from a single input column: values
/// `[start, end)` of column `col`.
#[derive(Clone, Copy)]
struct Span {
    col: usize,
    start: usize,
    end: usize,
}

impl Span {
    fn len(&self) -> usize {
        self.end - self.start
    }
}

/// Describes which runs of inner values make up the concatenated output, in
/// row-major output order. Spans are generated lazily by [`SpanPlan::iter`]
/// rather than materialized, so replaying the plan (e.g. once per struct
/// field) costs no `O(rows * columns)` allocation.
struct SpanPlan<'a> {
    offsets: Vec<&'a OffsetsBuffer<i64>>,
    validities: Vec<Option<&'a Bitmap>>,
    is_broadcast: Vec<bool>,
    output_height: usize,
    /// Total number of inner values in the output (the sum of all span lengths).
    out_values_len: usize,
}

impl SpanPlan<'_> {
    /// Whether output row `row` is valid, i.e. whether every input list at
    /// that row is valid (null propagation).
    ///
    /// # Safety
    /// `row < self.output_height` and the preconditions of
    /// [`horizontal_concat_list_unchecked`] hold.
    unsafe fn row_is_valid(&self, row: usize) -> bool {
        self.validities
            .iter()
            .zip(&self.is_broadcast)
            .all(|(validity, &broadcast)| {
                let row = if broadcast { 0 } else { row };
                validity.is_none_or(|v| v.get_bit_unchecked(row))
            })
    }

    /// Lazily yields the non-empty spans of every valid output row, in
    /// row-major output order.
    ///
    /// # Safety
    /// The preconditions of [`horizontal_concat_list_unchecked`] must hold
    /// while the returned iterator is consumed.
    unsafe fn iter(&self) -> impl Iterator<Item = Span> + '_ {
        (0..self.output_height)
            .filter(|&row| self.row_is_valid(row))
            .flat_map(move |row| {
                self.offsets
                    .iter()
                    .zip(&self.is_broadcast)
                    .enumerate()
                    .filter_map(move |(col, (offsets, &broadcast))| {
                        let row = if broadcast { 0 } else { row };
                        let (start, end) = offsets.start_end(row);
                        (end > start).then_some(Span { col, start, end })
                    })
            })
    }
}

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
/// * All arrays with non-unit length have the same length. Unit-length arrays
///   are broadcast to that common length.
///
/// # Errors
/// Errors if the total number of inner values overflows `i64`.
pub unsafe fn horizontal_concat_list_unchecked(
    arrays: &[ListArray<i64>],
) -> PolarsResult<ListArray<i64>> {
    assert!(!arrays.is_empty());
    debug_assert!(
        arrays
            .iter()
            .all(|a| a.values().dtype() == arrays[0].values().dtype())
    );

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
    debug_assert!(
        arrays
            .iter()
            .all(|a| a.len() == output_height || a.len() == 1)
    );

    let mut plan = SpanPlan {
        offsets: arrays.iter().map(|a| a.offsets()).collect(),
        validities: arrays.iter().map(|a| a.validity()).collect(),
        // A unit-length column is broadcast (unless the output itself has height 1).
        is_broadcast: arrays
            .iter()
            .map(|a| a.len() == 1 && output_height != 1)
            .collect(),
        output_height,
        out_values_len: 0,
    };

    let has_validity = plan.validities.iter().any(|v| v.is_some());
    let mut out_validity = has_validity.then(|| BitmapBuilder::with_capacity(output_height));

    let mut row_lengths: Vec<usize> = Vec::with_capacity(output_height);
    let mut out_values_len = 0usize;
    for row in 0..output_height {
        let valid = plan.row_is_valid(row);
        if let Some(b) = out_validity.as_mut() {
            b.push(valid);
        }
        if !valid {
            row_lengths.push(0);
            continue;
        }

        let row_len: usize = plan
            .offsets
            .iter()
            .zip(&plan.is_broadcast)
            .map(|(offsets, &broadcast)| {
                let row = if broadcast { 0 } else { row };
                let (start, end) = offsets.start_end(row);
                end - start
            })
            .sum();
        row_lengths.push(row_len);
        out_values_len += row_len;
    }
    plan.out_values_len = out_values_len;

    let inner_arrays: Vec<Box<dyn Array>> = arrays.iter().map(|a| a.values().clone()).collect();
    let out_inner = build_inner_values(&inner_arrays, &plan);

    let out_offsets: OffsetsBuffer<i64> =
        Offsets::try_from_lengths(row_lengths.into_iter())?.into();

    Ok(ListArray::<i64>::new(
        arrays[0].dtype().clone(),
        out_offsets,
        out_inner,
        out_validity.and_then(|b| b.into_opt_validity()),
    ))
}

fn downcast_all<T: Array + Clone + 'static>(values: &[Box<dyn Array>]) -> Vec<T> {
    values
        .iter()
        .map(|x| x.as_any().downcast_ref::<T>().unwrap().clone())
        .collect()
}

/// Build the concatenated inner values array by dispatching on the physical type.
/// Reused recursively for the fields of a struct inner type.
unsafe fn build_inner_values(values: &[Box<dyn Array>], plan: &SpanPlan<'_>) -> Box<dyn Array> {
    use PhysicalType::*;

    let dtype = values[0].dtype();

    match dtype.to_physical_type() {
        Null => Box::new(NullArray::new(dtype.clone(), plan.out_values_len)),
        Boolean => Box::new(build_leaf::<BooleanArray>(
            &downcast_all(values),
            plan,
            dtype.clone(),
        )),
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            Box::new(build_primitive::<$T>(
                &downcast_all(values),
                plan,
                dtype.clone(),
            ))
        }),
        LargeBinary => Box::new(build_leaf::<BinaryArray<i64>>(
            &downcast_all(values),
            plan,
            dtype.clone(),
        )),
        LargeList => Box::new(build_leaf::<ListArray<i64>>(
            &downcast_all(values),
            plan,
            dtype.clone(),
        )),
        FixedSizeList => Box::new(build_leaf::<FixedSizeListArray>(
            &downcast_all(values),
            plan,
            dtype.clone(),
        )),
        BinaryView => Box::new(build_leaf::<BinaryViewArray>(
            &downcast_all(values),
            plan,
            dtype.clone(),
        )),
        Utf8View => Box::new(build_leaf::<Utf8ViewArray>(
            &downcast_all(values),
            plan,
            dtype.clone(),
        )),
        Struct => Box::new(struct_::build_struct_values(&downcast_all(values), plan)),
        t => unimplemented!("concat_list not supported for data type {:?}", t),
    }
}

/// Monomorphized primitive builder: bulk-copies each span from the values
/// buffer, and its validity with bulk bitmap copies.
unsafe fn build_primitive<T: NativeType>(
    values: &[PrimitiveArray<T>],
    plan: &SpanPlan<'_>,
    dtype: ArrowDataType,
) -> PrimitiveArray<T> {
    let mut out = Vec::with_capacity(plan.out_values_len);
    let mut validity = values
        .iter()
        .any(|a| a.validity().is_some())
        .then(|| BitmapBuilder::with_capacity(plan.out_values_len));

    for span in plan.iter() {
        let arr = &values[span.col];
        out.extend_from_slice(&arr.values()[span.start..span.end]);
        if let Some(b) = validity.as_mut() {
            b.subslice_extend_from_opt_validity(arr.validity(), span.start, span.len());
        }
    }

    PrimitiveArray::new(
        dtype,
        out.into(),
        validity.and_then(|b| b.into_opt_validity()),
    )
}

/// Monomorphized generic leaf builder: replays the span plan over the
/// per-column values arrays, gathering each element (preserving inner-element
/// nulls, since `get_unchecked` yields `Option<ValueT>`).
unsafe fn build_leaf<T: StaticArray>(values: &[T], plan: &SpanPlan<'_>, dtype: ArrowDataType) -> T {
    let iter = plan.iter().flat_map(|span| {
        let arr = &values[span.col];
        (span.start..span.end).map(move |k| arr.get_unchecked(k))
    });
    TrustMyLength::new(iter, plan.out_values_len).collect_arr_trusted_with_dtype(dtype)
}

/// Build the element-level validity of a concatenated inner array, combining
/// the per-column validities in row-major order. Returns `None` when no input
/// column carries a validity bitmap, or when the result contains no nulls.
unsafe fn build_values_validity(
    per_col_validity: &[Option<&Bitmap>],
    plan: &SpanPlan<'_>,
) -> Option<Bitmap> {
    if per_col_validity.iter().all(|v| v.is_none()) {
        return None;
    }

    let mut b = BitmapBuilder::with_capacity(plan.out_values_len);
    for span in plan.iter() {
        b.subslice_extend_from_opt_validity(per_col_validity[span.col], span.start, span.len());
    }
    b.into_opt_validity()
}
