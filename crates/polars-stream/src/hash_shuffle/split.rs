use polars_arrow::bitmap::Bitmap;
use polars_expr::hash_keys::HashKeys;
use polars_utils::IdxSize;
use polars_utils::hashing::HashPartitioner;

/// Row indices grouped by partition: one array, increasing within each group (first/last rely on
/// it), and one more offset into it than there are groups.
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

/// Where `split` puts the rows without a hash: those with a null key, unless nulls are valid keys.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NullRows {
    /// They go to partition 0.
    FirstPartition,
    /// They are left out.
    Skip,
    /// They form one more group after the last partition, which is there even when empty.
    Separate,
}

/// The validity of `keys`, if some rows have no hash: those with a null key, unless nulls are
/// valid keys.
pub fn unhashed_validity(keys: &HashKeys) -> Option<&Bitmap> {
    if keys.null_is_valid() {
        return None;
    }
    keys.validity().filter(|v| v.unset_bits() > 0)
}

/// Groups the rows of `keys` (only those set in `mask`, if given) by partition: one pass for the
/// partitions and counts, a prefix sum, one pass placing the indices. Rows without a hash go
/// where `nulls` says. `None` with one partition, no mask and no rows left out or separated: all
/// rows, nothing computed.
pub fn split(
    keys: &HashKeys,
    mask: Option<&Bitmap>,
    partitioner: &HashPartitioner,
    nulls: NullRows,
) -> Option<SplitRows> {
    let num_partitions = partitioner.num_partitions();
    if let Some(mask) = mask {
        assert_eq!(mask.len(), keys.len());
    }

    let unhashed = unhashed_validity(keys);
    let hashed_mask;
    let mut mask = mask;
    if nulls == NullRows::Skip
        && let Some(validity) = unhashed
    {
        hashed_mask = mask.map_or_else(|| validity.clone(), |m| m & validity);
        mask = Some(&hashed_mask);
    }

    let separate = nulls == NullRows::Separate;
    if num_partitions == 1 && !(separate && unhashed.is_some()) {
        let mask = mask?;
        let idxs: Vec<IdxSize> = mask.true_idx_iter().map(|i| i as IdxSize).collect();
        let mut offsets = vec![0, idxs.len() as IdxSize];
        if separate {
            offsets.push(idxs.len() as IdxSize);
        }
        return Some(SplitRows { idxs, offsets });
    }

    let mut partitions = Vec::new();
    keys.gen_partitions(partitioner, &mut partitions, !separate);
    if separate && unhashed.is_some() {
        // `gen_partitions` gives the rows without a hash `IdxSize::MAX`.
        for p in &mut partitions {
            if *p == IdxSize::MAX {
                *p = num_partitions as IdxSize;
            }
        }
    }
    // SAFETY: there is one partition per row, and the mask has one bit per row.
    let in_mask = |i: usize| mask.is_none_or(|m| unsafe { m.get_bit_unchecked(i) });
    let num_groups = num_partitions + usize::from(separate);
    // SAFETY: the partitioner gives partitions below `num_partitions`, and the rows without a hash
    // are below `num_groups`.
    Some(unsafe { group_rows(&partitions, num_groups, in_mask) })
}

/// `rows`, or else the first `len` row indices, from `all_rows`.
pub fn rows_or_all<'a>(
    rows: Option<&'a [IdxSize]>,
    all_rows: &'a mut Vec<IdxSize>,
    len: usize,
) -> &'a [IdxSize] {
    match rows {
        Some(rows) => rows,
        None => {
            if all_rows.len() < len {
                all_rows.extend(all_rows.len() as IdxSize..len as IdxSize);
            }
            &all_rows[..len]
        },
    }
}

impl SplitRows {
    /// Groups rows `0..num_rows` by partition, where row `i` is in partition `partition_of(i)`
    /// and partition `p` has `counts[p]` rows.
    ///
    /// # Safety
    /// Every `partition_of(i)` is below `counts.len()` and `counts` holds the exact counts.
    pub unsafe fn from_counts(
        counts: &[IdxSize],
        num_rows: usize,
        partition_of: impl Fn(usize) -> IdxSize,
    ) -> Self {
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0);
        let mut total = 0;
        for count in counts {
            total += *count;
            offsets.push(total);
        }
        debug_assert_eq!(total as usize, num_rows);
        // SAFETY: guaranteed by the caller.
        unsafe { place_rows(offsets, (0..num_rows).map(|i| (i, partition_of(i)))) }
    }
}

/// Groups the rows by partition, where row `i` is in partition `partitions[i]`.
///
/// # Safety
/// Every partition is below `num_partitions`.
unsafe fn group_rows(
    partitions: &[IdxSize],
    num_partitions: usize,
    in_mask: impl Fn(usize) -> bool,
) -> SplitRows {
    let mut offsets: Vec<IdxSize> = vec![0; num_partitions + 1];
    for (i, p) in partitions.iter().enumerate() {
        if in_mask(i) {
            debug_assert!((*p as usize) < num_partitions);
            // SAFETY: guaranteed by the caller.
            unsafe { *offsets.get_unchecked_mut(*p as usize + 1) += 1 };
        }
    }
    let mut total = 0;
    for offset in &mut offsets {
        total += *offset;
        *offset = total;
    }
    let rows = partitions
        .iter()
        .enumerate()
        .filter(|(i, _)| in_mask(*i))
        .map(|(i, p)| (i, *p));
    // SAFETY: the partitions are in bounds, and the offsets count the rows in the mask.
    unsafe { place_rows(offsets, rows) }
}

/// Puts each row index at the next free place of its partition.
///
/// # Safety
/// The partitions are below `offsets.len() - 1`, and `offsets` holds the prefix sums of the
/// number of `rows` of each partition, starting at 0.
unsafe fn place_rows(
    offsets: Vec<IdxSize>,
    rows: impl Iterator<Item = (usize, IdxSize)>,
) -> SplitRows {
    let num_partitions = offsets.len() - 1;
    let mut next = offsets[..num_partitions].to_vec();
    let mut idxs: Vec<IdxSize> = vec![0; offsets[num_partitions] as usize];
    for (i, p) in rows {
        // SAFETY: guaranteed by the caller.
        unsafe {
            let pos = next.get_unchecked_mut(p as usize);
            *idxs.get_unchecked_mut(*pos as usize) = i as IdxSize;
            *pos += 1;
        }
    }
    SplitRows { idxs, offsets }
}
