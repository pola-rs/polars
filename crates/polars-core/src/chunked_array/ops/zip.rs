use arrow::bitmap::Bitmap;
use arrow::compute::utils::{combine_validities_and, combine_validities_and_not};
use polars_compute::if_then_else::{if_then_else_validity, IfThenElseKernel};

#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
use crate::prelude::*;
use crate::utils::{align_chunks_binary, align_chunks_ternary};

const SHAPE_MISMATCH_STR: &str =
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

impl<T> ChunkZip<T> for ChunkedArray<T>
where
    T: PolarsDataType<IsStruct = FalseT>,
    T::Array: for<'a> IfThenElseKernel<Scalar<'a> = T::Physical<'a>>,
    ChunkedArray<T>: ChunkExpandAtIndex<T>,
{
    fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &ChunkedArray<T>,
    ) -> PolarsResult<ChunkedArray<T>> {
        let if_true = self;
        let if_false = other;

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
                        let t = t.clone();
                        let f = f.clone();
                        IfThenElseKernel::if_then_else_broadcast_both(dtype.clone(), &bm, t, f)
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
                .map(|((m, t), f)| IfThenElseKernel::if_then_else(&bool_null_to_false(m), t, f));
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
                        let t = true_scalar.clone();
                        IfThenElseKernel::if_then_else_broadcast_true(&bm, t, f)
                    });
                ChunkedArray::from_chunk_iter_like(if_true, chunks)
            } else {
                combine_validities_chunked(if_false, mask, combine_validities_and_not)
            }

        // Broadcast false value.
        } else if if_false.len() == 1 {
            polars_ensure!(mask.len() == if_true.len(), ShapeMismatch: SHAPE_MISMATCH_STR);
            if let Some(false_scalar) = if_false.get(0) {
                let (mask_al, if_true_al) = align_chunks_binary(mask, if_true);
                let chunks =
                    mask_al
                        .downcast_iter()
                        .zip(if_true_al.downcast_iter())
                        .map(|(m, t)| {
                            let bm = bool_null_to_false(m);
                            let f = false_scalar.clone();
                            IfThenElseKernel::if_then_else_broadcast_false(&bm, t, f)
                        });
                ChunkedArray::from_chunk_iter_like(if_false, chunks)
            } else {
                combine_validities_chunked(if_true, mask, combine_validities_and)
            }
        } else {
            polars_bail!(ShapeMismatch: SHAPE_MISMATCH_STR)
        };

        Ok(ret.with_name(if_true.name()))
    }
}

// Basic implementation for ObjectArray.
#[cfg(feature = "object")]
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

#[cfg(feature = "dtype-struct")]
impl ChunkZip<StructType> for StructChunked {
    fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &ChunkedArray<StructType>,
    ) -> PolarsResult<ChunkedArray<StructType>> {
        let (l, r, mask) = align_chunks_ternary(self, other, mask);

        // Prepare the boolean arrays such that Null maps to false.
        // This prevents every field doing that.
        // # SAFETY
        // We don't modify the length and update the null count.
        let mut mask = mask.into_owned();
        unsafe {
            for arr in mask.downcast_iter_mut() {
                let bm = bool_null_to_false(arr);
                *arr = BooleanArray::from_data_default(bm, None);
            }
            mask.set_null_count(0);
        }

        // Zip all the fields.
        let fields = l
            .fields_as_series()
            .iter()
            .zip(r.fields_as_series())
            .map(|(lhs, rhs)| lhs.zip_with_same_type(&mask, &rhs))
            .collect::<PolarsResult<Vec<_>>>()?;

        let mut out = StructChunked::from_series(self.name(), &fields)?;

        // Zip the validities.
        if (l.null_count + r.null_count) > 0 {
            let validities = l
                .chunks()
                .iter()
                .zip(r.chunks())
                .map(|(l, r)| (l.validity(), r.validity()));

            fn broadcast(v: Option<&Bitmap>, arr: &ArrayRef) -> Bitmap {
                if v.unwrap().get(0).unwrap() {
                    Bitmap::new_with_value(true, arr.len())
                } else {
                    Bitmap::new_zeroed(arr.len())
                }
            }

            // # SAFETY
            // We don't modify the length and update the null count.
            unsafe {
                for ((arr, (lv, rv)), mask) in out
                    .chunks_mut()
                    .iter_mut()
                    .zip(validities)
                    .zip(mask.downcast_iter())
                {
                    // TODO! we can optimize this and use a kernel that is able to broadcast wo/ allocating.
                    let (lv, rv) = match (lv.map(|b| b.len()), rv.map(|b| b.len())) {
                        (Some(1), Some(1)) if arr.len() != 1 => {
                            let lv = broadcast(lv, arr);
                            let rv = broadcast(rv, arr);
                            (Some(lv), Some(rv))
                        },
                        (Some(a), Some(b)) if a == b => (lv.cloned(), rv.cloned()),
                        (Some(1), _) => {
                            let lv = broadcast(lv, arr);
                            (Some(lv), rv.cloned())
                        },
                        (_, Some(1)) => {
                            let rv = broadcast(rv, arr);
                            (lv.cloned(), Some(rv))
                        },
                        (None, Some(_)) | (Some(_), None) | (None, None) => {
                            (lv.cloned(), rv.cloned())
                        },
                        (Some(a), Some(b)) => {
                            polars_bail!(InvalidOperation: "got different sizes in 'zip' operation, got length: {a} and {b}")
                        },
                    };

                    // broadcast mask
                    let validity = if mask.len() != arr.len() && mask.len() == 1 {
                        if mask.get(0).unwrap() {
                            lv
                        } else {
                            rv
                        }
                    } else {
                        if_then_else_validity(mask.values(), lv.as_ref(), rv.as_ref())
                    };

                    *arr = arr.with_validity(validity);
                }
            }
            out.compute_len();
        }
        Ok(out)
    }
}
