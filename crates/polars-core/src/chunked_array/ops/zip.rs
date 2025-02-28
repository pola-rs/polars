use std::borrow::Cow;

use arrow::bitmap::{Bitmap, BitmapBuilder};
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
    Ok(ret.with_name(if_true.name().clone()))
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
                    let dtype = if_true.downcast_iter().next().unwrap().dtype();
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

        Ok(ret.with_name(if_true.name().clone()))
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
        let min_length = self.length.min(mask.length).min(other.length);
        let max_length = self.length.max(mask.length).max(other.length);

        let length = if min_length == 0 { 0 } else { max_length };

        debug_assert!(self.length == 1 || self.length == length);
        debug_assert!(mask.length == 1 || mask.length == length);
        debug_assert!(other.length == 1 || other.length == length);

        let mut if_true: Cow<ChunkedArray<StructType>> = Cow::Borrowed(self);
        let mut if_false: Cow<ChunkedArray<StructType>> = Cow::Borrowed(other);

        // Special case. In this case, we know what to do.
        // @TODO: Optimization. If all mask values are the same, select one of the two.
        if mask.length == 1 {
            // pl.when(None) <=> pl.when(False)
            let is_true = mask.get(0).unwrap_or(false);
            return Ok(if is_true && self.length == 1 {
                self.new_from_index(0, length)
            } else if is_true {
                self.clone()
            } else if other.length == 1 {
                let mut s = other.new_from_index(0, length);
                s.rename(self.name().clone());
                s
            } else {
                let mut s = other.clone();
                s.rename(self.name().clone());
                s
            });
        }

        // align_chunks_ternary can only align chunks if:
        // - Each chunkedarray only has 1 chunk
        // - Each chunkedarray has an equal length (i.e. is broadcasted)
        //
        // Therefore, we broadcast only those that are necessary to be broadcasted.
        let needs_broadcast =
            if_true.chunks().len() > 1 || if_false.chunks().len() > 1 || mask.chunks().len() > 1;
        if needs_broadcast && length > 1 {
            if self.length == 1 {
                let broadcasted = self.new_from_index(0, length);
                if_true = Cow::Owned(broadcasted);
            }
            if other.length == 1 {
                let broadcasted = other.new_from_index(0, length);
                if_false = Cow::Owned(broadcasted);
            }
        }

        let if_true = if_true.as_ref();
        let if_false = if_false.as_ref();

        let (if_true, if_false, mask) = align_chunks_ternary(if_true, if_false, mask);

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
        let fields = if_true
            .fields_as_series()
            .iter()
            .zip(if_false.fields_as_series())
            .map(|(lhs, rhs)| lhs.zip_with_same_type(&mask, &rhs))
            .collect::<PolarsResult<Vec<_>>>()?;

        let mut out = StructChunked::from_series(self.name().clone(), length, fields.iter())?;

        fn rechunk_bitmaps(
            total_length: usize,
            iter: impl Iterator<Item = (usize, Option<Bitmap>)>,
        ) -> Option<Bitmap> {
            let mut rechunked_length = 0;
            let mut rechunked_validity = None;
            for (chunk_length, validity) in iter {
                if let Some(validity) = validity {
                    if validity.unset_bits() > 0 {
                        rechunked_validity
                            .get_or_insert_with(|| {
                                let mut bm = BitmapBuilder::with_capacity(total_length);
                                bm.extend_constant(rechunked_length, true);
                                bm
                            })
                            .extend_from_bitmap(&validity);
                    }
                }

                rechunked_length += chunk_length;
            }

            if let Some(rechunked_validity) = rechunked_validity.as_mut() {
                rechunked_validity.extend_constant(total_length - rechunked_validity.len(), true);
            }

            rechunked_validity.map(BitmapBuilder::freeze)
        }

        // Zip the validities.
        //
        // We need to take two things into account:
        // 1. The chunk lengths of `out` might not necessarily match `l`, `r` and `mask`.
        // 2. `l` and `r` might still need to be broadcasted.
        if (if_true.null_count + if_false.null_count) > 0 {
            // Create one validity mask that spans the entirety of out.
            let rechunked_validity = match (if_true.len(), if_false.len()) {
                (1, 1) if length != 1 => {
                    match (if_true.null_count() == 0, if_false.null_count() == 0) {
                        (true, true) => None,
                        (false, true) => {
                            if mask.chunks().len() == 1 {
                                let m = mask.chunks()[0]
                                    .as_any()
                                    .downcast_ref::<BooleanArray>()
                                    .unwrap()
                                    .values();
                                Some(!m)
                            } else {
                                rechunk_bitmaps(
                                    length,
                                    mask.downcast_iter()
                                        .map(|m| (m.len(), Some(m.values().clone()))),
                                )
                            }
                        },
                        (true, false) => {
                            if mask.chunks().len() == 1 {
                                let m = mask.chunks()[0]
                                    .as_any()
                                    .downcast_ref::<BooleanArray>()
                                    .unwrap()
                                    .values();
                                Some(m.clone())
                            } else {
                                rechunk_bitmaps(
                                    length,
                                    mask.downcast_iter().map(|m| (m.len(), Some(!m.values()))),
                                )
                            }
                        },
                        (false, false) => Some(Bitmap::new_zeroed(length)),
                    }
                },
                (1, _) if length != 1 => {
                    debug_assert!(if_false
                        .chunk_lengths()
                        .zip(mask.chunk_lengths())
                        .all(|(r, m)| r == m));

                    let combine = if if_true.null_count() == 0 {
                        |if_false: Option<&Bitmap>, m: &Bitmap| {
                            if_false.map(|v| arrow::bitmap::or(v, m))
                        }
                    } else {
                        |if_false: Option<&Bitmap>, m: &Bitmap| {
                            Some(if_false.map_or_else(|| !m, |v| arrow::bitmap::and_not(v, m)))
                        }
                    };

                    if if_false.chunks().len() == 1 {
                        let if_false = if_false.chunks()[0].validity();
                        let m = mask.chunks()[0]
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .unwrap()
                            .values();

                        let validity = combine(if_false, m);
                        validity.filter(|v| v.unset_bits() > 0)
                    } else {
                        rechunk_bitmaps(
                            length,
                            if_false.chunks().iter().zip(mask.downcast_iter()).map(
                                |(chunk, mask)| {
                                    (mask.len(), combine(chunk.validity(), mask.values()))
                                },
                            ),
                        )
                    }
                },
                (_, 1) if length != 1 => {
                    debug_assert!(if_true
                        .chunk_lengths()
                        .zip(mask.chunk_lengths())
                        .all(|(l, m)| l == m));

                    let combine = if if_false.null_count() == 0 {
                        |if_true: Option<&Bitmap>, m: &Bitmap| {
                            if_true.map(|v| arrow::bitmap::or_not(v, m))
                        }
                    } else {
                        |if_true: Option<&Bitmap>, m: &Bitmap| {
                            Some(if_true.map_or_else(|| m.clone(), |v| arrow::bitmap::and(v, m)))
                        }
                    };

                    if if_true.chunks().len() == 1 {
                        let if_true = if_true.chunks()[0].validity();
                        let m = mask.chunks()[0]
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .unwrap()
                            .values();

                        let validity = combine(if_true, m);
                        validity.filter(|v| v.unset_bits() > 0)
                    } else {
                        rechunk_bitmaps(
                            length,
                            if_true.chunks().iter().zip(mask.downcast_iter()).map(
                                |(chunk, mask)| {
                                    (mask.len(), combine(chunk.validity(), mask.values()))
                                },
                            ),
                        )
                    }
                },
                (_, _) => {
                    debug_assert!(if_true
                        .chunk_lengths()
                        .zip(if_false.chunk_lengths())
                        .all(|(l, r)| l == r));
                    debug_assert!(if_true
                        .chunk_lengths()
                        .zip(mask.chunk_lengths())
                        .all(|(l, r)| l == r));

                    let validities = if_true
                        .chunks()
                        .iter()
                        .zip(if_false.chunks())
                        .map(|(l, r)| (l.validity(), r.validity()));

                    rechunk_bitmaps(
                        length,
                        validities
                            .zip(mask.downcast_iter())
                            .map(|((if_true, if_false), mask)| {
                                (
                                    mask.len(),
                                    if_then_else_validity(mask.values(), if_true, if_false),
                                )
                            }),
                    )
                },
            };

            // Apply the validity spreading over the chunks of out.
            if let Some(mut rechunked_validity) = rechunked_validity {
                assert_eq!(rechunked_validity.len(), out.len());

                let num_chunks = out.chunks().len();
                let null_count = rechunked_validity.unset_bits();

                // SAFETY: We do not change the lengths of the chunks and we update the null_count
                // afterwards.
                let chunks = unsafe { out.chunks_mut() };

                if num_chunks == 1 {
                    chunks[0] = chunks[0].with_validity(Some(rechunked_validity));
                } else {
                    for chunk in chunks {
                        let chunk_len = chunk.len();
                        let chunk_validity;

                        // SAFETY: We know that rechunked_validity.len() == out.len()
                        (chunk_validity, rechunked_validity) =
                            unsafe { rechunked_validity.split_at_unchecked(chunk_len) };
                        *chunk = chunk.with_validity(
                            (chunk_validity.unset_bits() > 0).then_some(chunk_validity),
                        );
                    }
                }

                out.null_count = null_count;
            } else {
                // SAFETY: We do not change the lengths of the chunks and we update the null_count
                // afterwards.
                let chunks = unsafe { out.chunks_mut() };

                for chunk in chunks {
                    *chunk = chunk.with_validity(None);
                }

                out.null_count = 0;
            }
        }

        if cfg!(debug_assertions) {
            let start_length = out.len();
            let start_null_count = out.null_count();

            out.compute_len();

            assert_eq!(start_length, out.len());
            assert_eq!(start_null_count, out.null_count());
        }
        Ok(out)
    }
}
