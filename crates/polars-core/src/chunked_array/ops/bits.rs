use polars_array::PlBooleanArray;

use super::BooleanChunked;

fn first_true_idx_impl(ca: &BooleanChunked, invert: bool) -> Option<usize> {
    let null_count = ca.null_count();
    if null_count == ca.len() {
        return None;
    }

    if (ca.is_sorted_ascending_flag() && invert) || (ca.is_sorted_descending_flag() && !invert) {
        return ca.first_non_null();
    }

    let invert_mask = if invert { u64::MAX } else { 0 };
    let mut offset = 0;
    for arr in ca.downcast_iter() {
        // A chunk that says the same of every element answers for itself: either it holds the bit
        // being looked for under a non-null element, which is the first such element, or it holds
        // none and the search moves on to the next chunk.
        if let Some(value) = arr.scalar_values() {
            if value != invert {
                if let Some(i) = first_valid(arr) {
                    return Some(offset + i);
                }
            }
            offset += arr.len();
            continue;
        }

        // The bits are walked as one run, so this reads the values a chunk lays out per element.
        let values = arr.flat_values().unwrap();
        if let Some(validity) = arr.validity().and_then(|v| v.flat_bitmap()) {
            let mut x_it = values.fast_iter_u56();
            let mut v_it = validity.fast_iter_u56();
            for (x, v) in x_it.by_ref().zip(v_it.by_ref()) {
                let n = ((x ^ invert_mask) & v).trailing_zeros() as usize;
                if n < 56 {
                    return Some(offset + n);
                }
                offset += 56;
            }

            let (x, rest_len) = x_it.remainder();
            let (v, _rest_len) = v_it.remainder();
            let n = ((x ^ invert_mask) & v).trailing_zeros() as usize;
            if n < rest_len {
                return Some(offset + n);
            }
            offset += rest_len;
        } else if arr
            .validity()
            .is_none_or(|v| v.scalar_value() == Some(true))
        {
            // No mask, or one that marks every element valid: the run of values is the answer.
            let n = if invert {
                values.leading_ones()
            } else {
                values.leading_zeros()
            };
            if n < values.len() {
                return Some(offset + n);
            }
            offset += values.len();
        } else {
            // A mask that marks every element null leaves nothing for this chunk to answer with.
            offset += arr.len();
        }
    }

    None
}

/// The position of the first non-null element of `arr`.
fn first_valid(arr: &PlBooleanArray) -> Option<usize> {
    match arr.validity() {
        None => (!arr.is_empty()).then_some(0),
        Some(validity) => match validity.scalar_value() {
            Some(valid) => (valid && !arr.is_empty()).then_some(0),
            None => validity.flat_bitmap().unwrap().true_idx_iter().next(),
        },
    }
}

/// The number of elements of `arr` that are both valid and `true`. A chunk whose values and mask
/// are both scalar is one bit each, so this is `O(1)` for it.
pub(crate) fn true_count(arr: &PlBooleanArray) -> usize {
    let values = arr.values();
    match arr.validity() {
        None => values.set_bits(),
        Some(validity) => match (values.scalar_value(), validity.scalar_value()) {
            // A scalar side shares one bit with every element, which settles the `and` on its own
            // wherever that bit is unset. Only two flat masks are walked.
            (Some(false), _) | (_, Some(false)) => 0,
            (Some(true), Some(true)) => arr.len(),
            (Some(true), None) => validity.set_bits(),
            (None, Some(true)) => values.set_bits(),
            (None, None) => values
                .flat_bitmap()
                .unwrap()
                .num_intersections_with(validity.flat_bitmap().unwrap()),
        },
    }
}

/// The number of elements of `arr` that are valid and `false` — see [`true_count`].
pub(crate) fn false_count(arr: &PlBooleanArray) -> usize {
    let values = arr.values();
    match arr.validity() {
        None => values.unset_bits(),
        Some(validity) => match (values.scalar_value(), validity.scalar_value()) {
            (Some(true), _) | (_, Some(false)) => 0,
            (Some(false), Some(true)) => arr.len(),
            (Some(false), None) => validity.set_bits(),
            (None, Some(true)) => values.unset_bits(),
            (None, None) => (!values.flat_bitmap().unwrap())
                .num_intersections_with(validity.flat_bitmap().unwrap()),
        },
    }
}

impl BooleanChunked {
    pub fn num_trues(&self) -> usize {
        self.downcast_iter().map(true_count).sum()
    }

    pub fn num_falses(&self) -> usize {
        self.downcast_iter().map(false_count).sum()
    }

    pub fn first_true_idx(&self) -> Option<usize> {
        first_true_idx_impl(self, false)
    }

    pub fn first_false_idx(&self) -> Option<usize> {
        first_true_idx_impl(self, true)
    }
}

#[cfg(test)]
mod tests {
    use polars_array::PlBitmap;

    use super::*;
    use crate::prelude::*;

    const LENGTH: usize = 5;

    /// Every crossing of the two representations, as `(values, validity)` over `LENGTH` elements.
    fn chunks() -> Vec<PlBooleanArray> {
        let flat_values = || PlBitmap::from_bitmap([true, false, true, true, false].into());
        let flat_validity = || PlBitmap::from_bitmap([true, true, false, true, true].into());

        let mut out = Vec::new();
        for values in [
            PlBitmap::new_scalar(false, LENGTH),
            PlBitmap::new_scalar(true, LENGTH),
            flat_values(),
        ] {
            for validity in [
                None,
                Some(PlBitmap::new_scalar(false, LENGTH)),
                Some(PlBitmap::new_scalar(true, LENGTH)),
                Some(flat_validity()),
            ] {
                out.push(PlBooleanArray::from_pl_bitmap(values.clone()).with_validity(validity));
            }
        }
        out
    }

    /// The same chunk with every buffer laid out one slot per element, which is what the counts
    /// and the search used to be given.
    fn written_out(arr: &PlBooleanArray) -> PlBooleanArray {
        arr.to_flat().into_owned().into_array()
    }

    /// Counting and searching cannot depend on the representation: a chunk that repeats one bit
    /// has to answer exactly what the same chunk written out does.
    #[test]
    fn a_scalar_chunk_counts_as_its_written_out_form() {
        for arr in chunks() {
            let scalar = BooleanChunked::with_chunk(PlSmallStr::EMPTY, arr.clone());
            let flat = BooleanChunked::with_chunk(PlSmallStr::EMPTY, written_out(&arr));

            assert_eq!(scalar.num_trues(), flat.num_trues(), "{arr:?}");
            assert_eq!(scalar.num_falses(), flat.num_falses(), "{arr:?}");
            assert_eq!(scalar.first_true_idx(), flat.first_true_idx(), "{arr:?}");
            assert_eq!(scalar.first_false_idx(), flat.first_false_idx(), "{arr:?}");
        }
    }

    /// The search runs over the chunks in order, so a scalar chunk has to carry the offset of the
    /// elements behind it along with the ones it holds itself.
    #[test]
    fn the_search_crosses_chunks_of_either_representation() {
        for arr in chunks() {
            for lead in chunks() {
                let scalar =
                    BooleanChunked::from_chunk_iter(PlSmallStr::EMPTY, [lead.clone(), arr.clone()]);
                let flat = BooleanChunked::from_chunk_iter(
                    PlSmallStr::EMPTY,
                    [written_out(&lead), written_out(&arr)],
                );

                assert_eq!(
                    scalar.first_true_idx(),
                    flat.first_true_idx(),
                    "{lead:?} then {arr:?}",
                );
                assert_eq!(
                    scalar.first_false_idx(),
                    flat.first_false_idx(),
                    "{lead:?} then {arr:?}",
                );
                assert_eq!(scalar.num_trues(), flat.num_trues());
            }
        }
    }
}
