use std::borrow::Cow;

use arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_array::PlBitmap;
use polars_array::bitmap::{combine_validities_and, invert};
use polars_compute::if_then_else::{IfThenElseKernel, if_then_else_validity};
use polars_error::PolarsContext;
use polars_utils::broadcast::broadcast_len;

#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
use crate::prelude::*;
use crate::utils::{align_chunks_binary, align_chunks_ternary};

const SHAPE_MISMATCH_STR: &str =
    "shapes of `self`, `mask` and `other` are not suitable for `zip_with` operation";

/// The result of a mask that reads the same at every element: `mask_len` is the height that mask
/// covers, which is one for a column of a single element and the height of the column for one
/// whose only chunk repeats a single bit.
fn if_then_else_broadcast_mask<T: PolarsDataType>(
    mask: bool,
    mask_len: usize,
    if_true: &ChunkedArray<T>,
    if_false: &ChunkedArray<T>,
) -> PolarsResult<ChunkedArray<T>>
where
    ChunkedArray<T>: ChunkExpandAtIndex<T>,
{
    let src = if mask { if_true } else { if_false };
    let other = if mask { if_false } else { if_true };
    let len = broadcast_len([mask_len, src.len(), other.len()]).context(SHAPE_MISMATCH_STR)?;
    let ret = src.broadcast_to(len)?.into_owned();
    Ok(ret.with_name(if_true.name().clone()))
}

/// The bits of a mask chunk that [`bool_null_to_false`] left fully valid, written out one bit per
/// element for the kernels that read them that way.
fn mask_values(mask: &PlBooleanArray) -> Bitmap {
    bool_null_to_false(mask).into_bitmap()
}

/// The bits of `mask`, reading a null as unset — which is what a null means to `zip_with`.
fn bool_null_to_false(mask: &PlBooleanArray) -> PlBitmap {
    // An element the mask says nothing about is one it does not pick, which is what an unset bit
    // says in turn: the two fold together into the one mask the kernels read. A mask that repeats
    // a single bit stays a single bit.
    combine_validities_and(Some(mask.values()), mask.validity())
        .expect("the values of a mask are a mask of their own")
}

/// Combines the validity of `ca` with the bits of `mask`, which are inverted first if `not_mask`.
///
/// A null in the mask is read as unset, as it is throughout `zip_with`.
fn combine_validities_chunked<T: PolarsDataType>(
    ca: &ChunkedArray<T>,
    mask: &BooleanChunked,
    not_mask: bool,
) -> ChunkedArray<T> {
    let (ca_al, mask_al) = align_chunks_binary(ca, mask);
    let chunks = ca_al
        .downcast_iter()
        .zip(mask_al.downcast_iter())
        .map(|(a, m)| {
            let length = m.len();
            let mut bm = bool_null_to_false(m);
            if not_mask {
                // Inverting leaves the mask in the representation it is in: a single bit stays a
                // single bit, which keeps a fully null or fully valid result in `O(1)` memory.
                bm = PlBitmap::new_broadcast(invert(bm.as_ref()), length);
            }
            let validity = combine_validities_and(a.validity(), Some(bm.as_ref()));
            a.clone().with_validity_typed(validity)
        });
    ChunkedArray::from_chunk_iter_like(ca, chunks)
}

impl<T> ChunkZip<T> for ChunkedArray<T>
where
    T: PolarsDataType<IsStruct = FalseT>,
    T::Array: IfThenElseKernel,
    ChunkedArray<T>: ChunkExpandAtIndex<T>,
{
    fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &ChunkedArray<T>,
    ) -> PolarsResult<ChunkedArray<T>> {
        let if_true = self;
        let if_false = other;

        // Broadcast mask: a mask that reads the same at every element — a column of one element,
        // or one whose only chunk repeats a single bit — picks the same side throughout, so the
        // sides are neither zipped nor written out. A null reads as false, as it does below.
        if let Some(bit) = mask.scalar_value() {
            return if_then_else_broadcast_mask(
                bit.unwrap_or(false),
                mask.len(),
                if_true,
                if_false,
            );
        }

        // Broadcast both.
        let ret = if if_true.len() == 1 && if_false.len() == 1 {
            match (if_true.get(0), if_false.get(0)) {
                (None, None) => ChunkedArray::full_null_like(if_true, mask.len()),
                (None, Some(_)) => {
                    combine_validities_chunked(&if_false.new_from_index(0, mask.len()), mask, true)
                },
                (Some(_), None) => {
                    combine_validities_chunked(&if_true.new_from_index(0, mask.len()), mask, false)
                },
                (Some(t), Some(f)) => {
                    let chunks = mask.downcast_iter().map(|m| {
                        let bm = bool_null_to_false(m);
                        let t = t.clone();
                        let f = f.clone();
                        IfThenElseKernel::if_then_else_broadcast_both(bm.as_ref(), t, f)
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
                .map(|((m, t), f)| {
                    IfThenElseKernel::if_then_else(bool_null_to_false(m).as_ref(), t, f)
                });
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
                        IfThenElseKernel::if_then_else_broadcast_true(bm.as_ref(), t, f)
                    });
                ChunkedArray::from_chunk_iter_like(if_true, chunks)
            } else {
                combine_validities_chunked(if_false, mask, true)
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
                            IfThenElseKernel::if_then_else_broadcast_false(bm.as_ref(), t, f)
                        });
                ChunkedArray::from_chunk_iter_like(if_false, chunks)
            } else {
                combine_validities_chunked(if_true, mask, false)
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
    fn if_then_else_flat(mask: &Bitmap, if_true: &Flat<Self>, if_false: &Flat<Self>) -> Self {
        mask.iter()
            .zip(if_true.iter())
            .zip(if_false.iter())
            .map(|((m, t), f)| if m { t } else { f })
            .collect_arr()
    }

    fn if_then_else_flat_broadcast_true(
        mask: &Bitmap,
        if_true: Self::ValueT<'_>,
        if_false: &Flat<Self>,
    ) -> Self {
        mask.iter()
            .zip(if_false.iter())
            .map(|(m, f)| if m { Some(if_true) } else { f })
            .collect_arr()
    }

    fn if_then_else_flat_broadcast_false(
        mask: &Bitmap,
        if_true: &Flat<Self>,
        if_false: Self::ValueT<'_>,
    ) -> Self {
        mask.iter()
            .zip(if_true.iter())
            .map(|(m, t)| if m { t } else { Some(if_false) })
            .collect_arr()
    }

    fn if_then_else_flat_broadcast_both(
        mask: &Bitmap,
        if_true: Self::ValueT<'_>,
        if_false: Self::ValueT<'_>,
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
            return Ok(if is_true {
                self.broadcast_to(length)?.into_owned()
            } else {
                other
                    .broadcast_to(length)?
                    .into_owned()
                    .with_name(self.name().clone())
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
            if_true = self.broadcast_to(length)?;
            if_false = other.broadcast_to(length)?;
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
                let length = arr.len();
                // A mask that repeats a single bit says the same of every element and stays that
                // one bit; anything else holds one bit per element, as it did before.
                let bm = bool_null_to_false(arr);
                *arr = match bm.scalar_value() {
                    Some(bit) => PlBooleanArray::new_scalar(bit, length),
                    None => PlBooleanArray::new(bm.into_bitmap(), length, None),
                };
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
                        let v = rechunked_validity.get_or_insert_with(|| {
                            let mut bm = BitmapBuilder::with_capacity(total_length);
                            bm.extend_constant(rechunked_length, true);
                            bm
                        });
                        v.extend_constant(rechunked_length - v.len(), true);
                        v.extend_from_bitmap(&validity);
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
                                Some(!&mask_values(mask.downcast_get(0).unwrap()))
                            } else {
                                rechunk_bitmaps(
                                    length,
                                    mask.downcast_iter()
                                        .map(|m| (m.len(), Some(mask_values(m)))),
                                )
                            }
                        },
                        (true, false) => {
                            if mask.chunks().len() == 1 {
                                Some(mask_values(mask.downcast_get(0).unwrap()))
                            } else {
                                rechunk_bitmaps(
                                    length,
                                    mask.downcast_iter()
                                        .map(|m| (m.len(), Some(!&mask_values(m)))),
                                )
                            }
                        },
                        (false, false) => Some(Bitmap::new_zeroed(length)),
                    }
                },
                (1, _) if length != 1 => {
                    debug_assert!(
                        if_false
                            .chunk_lengths()
                            .zip(mask.chunk_lengths())
                            .all(|(r, m)| r == m)
                    );

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
                        let if_false = if_false.chunks()[0]
                            .validity()
                            .map(|v| v.to_flat().into_owned());
                        let m = mask_values(mask.downcast_get(0).unwrap());

                        let validity = combine(if_false.as_ref(), &m);
                        validity.filter(|v| v.unset_bits() > 0)
                    } else {
                        rechunk_bitmaps(
                            length,
                            if_false.chunks().iter().zip(mask.downcast_iter()).map(
                                |(chunk, mask)| {
                                    let validity =
                                        chunk.validity().map(|v| v.to_flat().into_owned());
                                    (mask.len(), combine(validity.as_ref(), &mask_values(mask)))
                                },
                            ),
                        )
                    }
                },
                (_, 1) if length != 1 => {
                    debug_assert!(
                        if_true
                            .chunk_lengths()
                            .zip(mask.chunk_lengths())
                            .all(|(l, m)| l == m)
                    );

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
                        let if_true = if_true.chunks()[0]
                            .validity()
                            .map(|v| v.to_flat().into_owned());
                        let m = mask_values(mask.downcast_get(0).unwrap());

                        let validity = combine(if_true.as_ref(), &m);
                        validity.filter(|v| v.unset_bits() > 0)
                    } else {
                        rechunk_bitmaps(
                            length,
                            if_true.chunks().iter().zip(mask.downcast_iter()).map(
                                |(chunk, mask)| {
                                    let validity =
                                        chunk.validity().map(|v| v.to_flat().into_owned());
                                    (mask.len(), combine(validity.as_ref(), &mask_values(mask)))
                                },
                            ),
                        )
                    }
                },
                (_, _) => {
                    debug_assert!(
                        if_true
                            .chunk_lengths()
                            .zip(if_false.chunk_lengths())
                            .all(|(l, r)| l == r)
                    );
                    debug_assert!(
                        if_true
                            .chunk_lengths()
                            .zip(mask.chunk_lengths())
                            .all(|(l, r)| l == r)
                    );

                    let validities =
                        if_true
                            .chunks()
                            .iter()
                            .zip(if_false.chunks())
                            .map(|(l, r)| {
                                (
                                    l.validity().map(|v| v.to_flat().into_owned()),
                                    r.validity().map(|v| v.to_flat().into_owned()),
                                )
                            });

                    rechunk_bitmaps(
                        length,
                        validities
                            .zip(mask.downcast_iter())
                            .map(|((if_true, if_false), mask)| {
                                (
                                    mask.len(),
                                    if_then_else_validity(
                                        &mask_values(mask),
                                        if_true.as_ref(),
                                        if_false.as_ref(),
                                    ),
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
                    chunks[0] =
                        chunks[0].with_validity(Some(PlBitmap::from_bitmap(rechunked_validity)));
                } else {
                    for chunk in chunks {
                        let chunk_len = chunk.len();
                        let chunk_validity;

                        // SAFETY: We know that rechunked_validity.len() == out.len()
                        (chunk_validity, rechunked_validity) =
                            unsafe { rechunked_validity.split_at_unchecked(chunk_len) };
                        *chunk = chunk.with_validity(
                            (chunk_validity.unset_bits() > 0)
                                .then_some(PlBitmap::from_bitmap(chunk_validity)),
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

#[cfg(test)]
mod tests {
    use polars_array::PlPrimitiveArray;

    use super::*;

    /// A mask of two chunks, the first of which repeats `bit` and the second of which is laid out
    /// one bit per element. Only a mask of more than one chunk reaches the per-chunk kernels with
    /// a chunk that repeats a single bit — a single one is answered before them.
    fn split_mask(bit: bool, rest: [bool; 3]) -> BooleanChunked {
        BooleanChunked::from_chunk_iter(
            "mask".into(),
            [
                PlBooleanArray::new_scalar(bit, 3),
                PlBooleanArray::from_vec(rest.to_vec()),
            ],
        )
    }

    /// Six elements over two chunks, to line up with [`split_mask`].
    fn split_values(values: [Option<i32>; 6]) -> Int32Chunked {
        Int32Chunked::from_chunk_iter(
            "values".into(),
            [
                PlPrimitiveArray::from_iter(values[..3].iter().copied()),
                PlPrimitiveArray::from_iter(values[3..].iter().copied()),
            ],
        )
    }

    /// A chunk of the mask that repeats one bit picks the same side for every element it covers.
    #[test]
    fn a_repeated_bit_in_one_chunk_picks_one_side() {
        let if_true = split_values([Some(1), Some(2), Some(3), Some(4), Some(5), Some(6)]);
        let if_false = split_values([Some(-1), None, Some(-3), Some(-4), Some(-5), Some(-6)]);

        let zipped = if_true
            .zip_with(&split_mask(true, [true, false, true]), &if_false)
            .unwrap();
        assert_eq!(
            Vec::from_iter(zipped.iter()),
            [Some(1), Some(2), Some(3), Some(4), Some(-5), Some(6)],
        );

        let zipped = if_true
            .zip_with(&split_mask(false, [true, false, true]), &if_false)
            .unwrap();
        assert_eq!(
            Vec::from_iter(zipped.iter()),
            [Some(-1), None, Some(-3), Some(4), Some(-5), Some(6)],
        );
    }

    /// The same holds where one side is a single element that stands for the whole column, whose
    /// nulls reach the result through the mask alone.
    #[test]
    fn a_repeated_bit_in_one_chunk_picks_a_broadcast_side() {
        let if_true = Int32Chunked::from_slice("t".into(), &[7]);
        let if_false = split_values([Some(-1), Some(-2), Some(-3), Some(-4), Some(-5), Some(-6)]);

        for bit in [false, true] {
            let mask = split_mask(bit, [true, false, true]);
            let zipped = if_true.zip_with(&mask, &if_false).unwrap();

            let expected = Vec::from_iter(
                mask.iter()
                    .zip(if_false.iter())
                    .map(|(m, f)| if m == Some(true) { Some(7) } else { f }),
            );
            assert_eq!(Vec::from_iter(zipped.iter()), expected);
        }

        // A null on the broadcast side leaves the mask alone to say which elements survive, which
        // is the path that combines the two validities rather than zipping any values.
        let null = Int32Chunked::full_null("t".into(), 1);
        for bit in [false, true] {
            let mask = split_mask(bit, [true, false, true]);

            let zipped = null.zip_with(&mask, &if_false).unwrap();
            let expected = Vec::from_iter(
                mask.iter()
                    .zip(if_false.iter())
                    .map(|(m, f)| if m == Some(true) { None } else { f }),
            );
            assert_eq!(Vec::from_iter(zipped.iter()), expected);

            // The same the other way around, which inverts the mask before combining it.
            let zipped = if_false.zip_with(&mask, &null).unwrap();
            let expected = Vec::from_iter(
                mask.iter()
                    .zip(if_false.iter())
                    .map(|(m, t)| if m == Some(true) { t } else { None }),
            );
            assert_eq!(Vec::from_iter(zipped.iter()), expected);
        }
    }
}
