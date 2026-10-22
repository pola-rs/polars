use polars_arrow::bitmap::Bitmap;
use polars_expr::hash_keys::HashKeys;
use polars_utils::IdxSize;
use polars_utils::hashing::HashPartitioner;

/// Row indices grouped by partition: one array, increasing within each partition (first/last
/// rely on it), and `num_partitions + 1` offsets into it.
pub struct SplitRows {
    idxs: Vec<IdxSize>,
    offsets: Vec<IdxSize>,
}

impl SplitRows {
    /// The rows of partition `p`, in increasing order.
    pub fn partition(&self, p: usize) -> &[IdxSize] {
        &self.idxs[self.offsets[p] as usize..self.offsets[p + 1] as usize]
    }
}

/// Groups the rows of `keys` (only those set in `mask`, if given) by partition: one pass for the
/// partitions and counts, a prefix sum, one pass placing the indices. Rows without a hash go to
/// partition 0. `None` with one partition and no mask: all rows, nothing computed.
pub fn split(
    keys: &HashKeys,
    mask: Option<&Bitmap>,
    partitioner: &HashPartitioner,
) -> Option<SplitRows> {
    let num_partitions = partitioner.num_partitions();
    if let Some(mask) = mask {
        assert_eq!(mask.len(), keys.len());
    }

    if num_partitions == 1 {
        let mask = mask?;
        let idxs: Vec<IdxSize> = mask.true_idx_iter().map(|i| i as IdxSize).collect();
        let offsets = vec![0, idxs.len() as IdxSize];
        return Some(SplitRows { idxs, offsets });
    }

    let mut partitions = Vec::new();
    keys.gen_partitions(partitioner, &mut partitions, true);
    // SAFETY: there is one partition per row, and the mask has one bit per row.
    let in_mask = |i: usize| mask.is_none_or(|m| unsafe { m.get_bit_unchecked(i) });

    let mut offsets: Vec<IdxSize> = vec![0; num_partitions + 1];
    for (i, p) in partitions.iter().enumerate() {
        if in_mask(i) {
            offsets[*p as usize + 1] += 1;
        }
    }
    let mut total = 0;
    for offset in &mut offsets {
        total += *offset;
        *offset = total;
    }

    let mut next = offsets[..num_partitions].to_vec();
    let mut idxs: Vec<IdxSize> = vec![0; total as usize];
    for (i, p) in partitions.iter().enumerate() {
        if in_mask(i) {
            let pos = &mut next[*p as usize];
            idxs[*pos as usize] = i as IdxSize;
            *pos += 1;
        }
    }

    Some(SplitRows { idxs, offsets })
}
