/// Forked from Arrow until their API stabilizes.
///
/// Note that the bound checks are optimized away.
///
use crate::bitmap::utils::{BitChunkIterExact, BitChunks};

pub fn find_first_true_false_null(
    mut bit_chunks: BitChunks<u64>,
    mut validity_chunks: BitChunks<u64>,
) -> (Option<usize>, Option<usize>, Option<usize>) {
    let (mut true_index, mut false_index, mut null_index) = (None, None, None);
    let (mut true_not_found_mask, mut false_not_found_mask, mut null_not_found_mask) =
        (!0u64, !0u64, !0u64); // All ones while not found.
    let mut offset: usize = 0;
    let mut all_found = false;
    for (truth_mask, null_mask) in (&mut bit_chunks).zip(&mut validity_chunks) {
        let mask = null_mask & truth_mask & true_not_found_mask;
        if mask > 0 {
            true_index = Some(offset + mask.trailing_zeros() as usize);
            true_not_found_mask = 0;
        }
        let mask = null_mask & !truth_mask & false_not_found_mask;
        if mask > 0 {
            false_index = Some(offset + mask.trailing_zeros() as usize);
            false_not_found_mask = 0;
        }
        if !null_mask & null_not_found_mask > 0 {
            null_index = Some(offset + null_mask.trailing_ones() as usize);
            null_not_found_mask = 0;
        }
        if null_not_found_mask | true_not_found_mask | false_not_found_mask == 0 {
            all_found = true;
            break;
        }
        offset += 64;
    }
    if !all_found {
        for (val, not_null) in bit_chunks
            .remainder_iter()
            .zip(validity_chunks.remainder_iter())
        {
            if true_index.is_none() && not_null && val {
                true_index = Some(offset);
            } else if false_index.is_none() && not_null && !val {
                false_index = Some(offset);
            } else if null_index.is_none() && !not_null {
                null_index = Some(offset);
            }
            offset += 1;
        }
    }
    (true_index, false_index, null_index)
}

pub fn find_first_true_false_no_null(
    mut bit_chunks: BitChunks<u64>,
) -> (Option<usize>, Option<usize>) {
    let (mut true_index, mut false_index) = (None, None);
    let (mut true_not_found_mask, mut false_not_found_mask) = (!0u64, !0u64); // All ones while not found.
    let mut offset: usize = 0;
    let mut all_found = false;
    for truth_mask in &mut bit_chunks {
        let mask = truth_mask & true_not_found_mask;
        if mask > 0 {
            true_index = Some(offset + mask.trailing_zeros() as usize);
            true_not_found_mask = 0;
        }
        let mask = !truth_mask & false_not_found_mask;
        if mask > 0 {
            false_index = Some(offset + mask.trailing_zeros() as usize);
            false_not_found_mask = 0;
        }
        if true_not_found_mask | false_not_found_mask == 0 {
            all_found = true;
            break;
        }
        offset += 64;
    }
    if !all_found {
        for val in bit_chunks.remainder_iter() {
            if true_index.is_none() && val {
                true_index = Some(offset);
            } else if false_index.is_none() && !val {
                false_index = Some(offset);
            }
            offset += 1;
        }
    }
    (true_index, false_index)
}
