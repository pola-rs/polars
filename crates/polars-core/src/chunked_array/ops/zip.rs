use arrow::bitmap::Bitmap;
use arrow::compute::utils::{combine_validities_and, combine_validities_and_not};
use polars_compute::if_then_else::IfThenElseKernel;

use crate::chunked_array::object::ObjectArray;
use crate::prelude::*;
use crate::utils::{align_chunks_binary, align_chunks_ternary};

const SHAPE_MISMATCH_STR: &'static str =
    "shapes of `self`, `mask` and `other` are not suitable for `zip_with` operation";

fn if_then_else_broadcast_mask<T: PolarsDataType>(
    mask: bool,
    if_true: &ChunkedArray<T>,
    if_false: &ChunkedArray<T>,
) -> PolarsResult<ChunkedArray<T>>
where
    ChunkedArray<T>: ChunkExpandAtIndex<T>,
{
    let src = if mask { if_true } else { if_false };
    let other = if mask { if_false } else { if_true };
    let ret = match (src.len(), other.len()) {
        (a, b) if a == b => src.clone(),
        (_, 1) => src.clone(),
        (1, other_len) => src.new_from_index(0, other_len),
        _ => polars_bail!(ShapeMismatch: SHAPE_MISMATCH_STR),
    };
    Ok(ret.with_name(if_true.name()))
}

fn bool_null_to_false(mask: &BooleanArray) -> Bitmap {
    if mask.null_count() == 0 {
        mask.values().clone()
    } else {
        mask.values() & mask.validity().unwrap()
    }
}

/// Combines the validities of ca with the bits in mask using the given combiner.
///
/// If the mask itself has validity, those null bits are converted to false.
fn combine_validities_chunked<
    T: PolarsDataType,
    F: Fn(Option<&Bitmap>, Option<&Bitmap>) -> Option<Bitmap>,
>(
    ca: &ChunkedArray<T>,
    mask: &BooleanChunked,
    combiner: F,
) -> ChunkedArray<T> {
    let (ca_al, mask_al) = align_chunks_binary(ca, mask);
    let chunks = ca_al
        .downcast_iter()
        .zip(mask_al.downcast_iter())
        .map(|(a, m)| {
            let bm = bool_null_to_false(m);
            let validity = combiner(a.validity(), Some(&bm));
            a.clone().with_validity_typed(validity)
        });
    ChunkedArray::from_chunk_iter_like(ca, chunks)
}

fn if_then_else_dispatch<T: PolarsDataType, K, KBT, KBF, KBB>(
    mask: &BooleanChunked,
    if_true: &ChunkedArray<T>,
    if_false: &ChunkedArray<T>,
    kernel: K,
    kernel_broadcast_true: KBT,
    kernel_broadcast_false: KBF,
    kernel_broadcast_both: KBB,
) -> PolarsResult<ChunkedArray<T>>
where
    ChunkedArray<T>: ChunkExpandAtIndex<T>,
    K: Fn(&Bitmap, &T::Array, &T::Array) -> T::Array,
    KBT: Fn(&Bitmap, T::Physical<'_>, &T::Array) -> T::Array,
    KBF: Fn(&Bitmap, &T::Array, T::Physical<'_>) -> T::Array,
    KBB: Fn(ArrowDataType, &Bitmap, T::Physical<'_>, T::Physical<'_>) -> T::Array,
{
    // Broadcast mask.
    if mask.len() == 1 {
        return if_then_else_broadcast_mask(mask.get(0).unwrap_or(false), if_true, if_false);
    }

    // Broadcast both.
    let ret = if if_true.len() == 1 && if_false.len() == 1 {
        match (if_true.get(0), if_false.get(0)) {
            (None, None) => ChunkedArray::full_null_like(if_true, mask.len()),
            (None, Some(_)) => combine_validities_chunked(
                &if_false.new_from_index(0, mask.len()),
                mask,
                combine_validities_and_not,
            ),
            (Some(_), None) => combine_validities_chunked(
                &if_true.new_from_index(0, mask.len()),
                mask,
                combine_validities_and,
            ),
            (Some(t), Some(f)) => {
                let dtype = if_true.downcast_iter().next().unwrap().data_type();
                let chunks = mask.downcast_iter().map(|m| {
                    let bm = bool_null_to_false(m);
                    kernel_broadcast_both(dtype.clone(), &bm, t.clone(), f.clone())
                });
                ChunkedArray::from_chunk_iter_like(if_true, chunks)
            },
        }

    // Broadcast neither.
    } else if if_true.len() == if_false.len() {
        polars_ensure!(mask.len() == if_true.len(), ShapeMismatch: SHAPE_MISMATCH_STR);
        let (mask_al, if_true_al, if_false_al) = align_chunks_ternary(mask, if_true, if_false);
        let chunks = mask_al
            .downcast_iter()
            .zip(if_true_al.downcast_iter())
            .zip(if_false_al.downcast_iter())
            .map(|((m, t), f)| kernel(&bool_null_to_false(m), t, f));
        ChunkedArray::from_chunk_iter_like(if_true, chunks)

    // Broadcast true value.
    } else if if_true.len() == 1 {
        polars_ensure!(mask.len() == if_false.len(), ShapeMismatch: SHAPE_MISMATCH_STR);
        if let Some(true_scalar) = if_true.get(0) {
            let (mask_al, if_false_al) = align_chunks_binary(mask, if_false);
            let chunks = mask_al
                .downcast_iter()
                .zip(if_false_al.downcast_iter())
                .map(|(m, f)| {
                    let bm = bool_null_to_false(m);
                    kernel_broadcast_true(&bm, true_scalar.clone(), f)
                });
            ChunkedArray::from_chunk_iter_like(if_true, chunks)
        } else {
            combine_validities_chunked(if_false, mask, combine_validities_and_not)
        }

    // Broadcast false value.
    } else if if_false.len() == 1 {
        polars_ensure!(mask.len() == if_true.len(), ShapeMismatch: SHAPE_MISMATCH_STR);
        let false_scalar = if_false.get(0);
        let (mask_al, if_true_al) = align_chunks_binary(mask, if_true);
        let chunks = mask_al
            .downcast_iter()
            .zip(if_true_al.downcast_iter())
            .map(|(m, t)| {
                let bm = bool_null_to_false(m);
                if let Some(false_val) = false_scalar.clone() {
                    kernel_broadcast_false(&bm, t, false_val)
                } else {
                    let validity = combine_validities_and(t.validity(), Some(&bm));
                    t.clone().with_validity_typed(validity)
                }
            });
        ChunkedArray::from_chunk_iter_like(if_true, chunks)
    } else {
        polars_bail!(ShapeMismatch: SHAPE_MISMATCH_STR)
    };

    Ok(ret.with_name(if_true.name()))
}

impl<T> ChunkZip<T> for ChunkedArray<T>
where
    T: PolarsDataType,
    T::Array: for<'a> IfThenElseKernel<Scalar<'a>=T::Physical<'a>>,
    ChunkedArray<T>: ChunkExpandAtIndex<T>,
{
    fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &ChunkedArray<T>,
    ) -> PolarsResult<ChunkedArray<T>> {
        if_then_else_dispatch(
            mask,
            self,
            other,
            IfThenElseKernel::if_then_else,
            |m, t, f| IfThenElseKernel::if_then_else_broadcast_true(m, t, f),
            |m, t, f| IfThenElseKernel::if_then_else_broadcast_false(m, t, f),
            |dt, m, t, f| IfThenElseKernel::if_then_else_broadcast_both(dt, m, t, f),
        )
    }
}

// Basic implementation for ObjectArray.
impl<T: PolarsObject> IfThenElseKernel for ObjectArray<T> {
    type Scalar<'a> = &'a T;

    fn if_then_else(mask: &Bitmap, if_true: &Self, if_false: &Self) -> Self {
        mask.iter()
            .zip(if_true.iter())
            .zip(if_false.iter())
            .map(|((m, t), f)| if m { t } else { f })
            .collect_arr()
    }

    fn if_then_else_broadcast_true(
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: &Self,
    ) -> Self {
        mask.iter()
            .zip(if_false.iter())
            .map(|(m, f)| if m { Some(if_true) } else { f })
            .collect_arr()
    }

    fn if_then_else_broadcast_false(
        mask: &Bitmap,
        if_true: &Self,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        mask.iter()
            .zip(if_true.iter())
            .map(|(m, t)| if m { t } else { Some(if_false) })
            .collect_arr()
    }

    fn if_then_else_broadcast_both(
        _dtype: ArrowDataType,
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        mask.iter()
            .map(|m| if m { if_true } else { if_false })
            .collect_arr()
    }
}
