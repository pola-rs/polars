use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_utils::itertools::Itertools;
use polars_utils::IdxSize;

/// Partitions this Vec into multiple Vecs.
///
/// # Safety
/// partitions_idxs[i] < partition_sizes.len() for all i.
/// idx_in_partition[i] < partition_sizes[partition_idxs[i]] for all i.
/// Each partition p has an associated set of idx_in_partition, this set
/// contains 0..partition_size[p] exactly once.
pub unsafe fn partition_vec<T>(
    v: Vec<T>,
    partition_sizes: &[IdxSize],
    partition_idxs: &[IdxSize],
    idx_in_partition: &[IdxSize],
) -> Vec<Vec<T>> {
    assert!(idx_in_partition.len() == v.len());
    assert!(partition_idxs.len() == v.len());
        
    let mut partitions = partition_sizes
        .iter()
        .map(|sz| Vec::<T>::with_capacity(*sz as usize))
        .collect_vec();

    unsafe {
        // Scatter into each partition.
        for (i, val) in v.into_iter().enumerate() {
            let p_idx = *partition_idxs.get_unchecked(i) as usize;
            let idx_in_p = *idx_in_partition.get_unchecked(i) as usize;
            debug_assert!(p_idx < partitions.len());
            let p = partitions.get_unchecked_mut(p_idx);
            debug_assert!(idx_in_p < p.capacity());
            p.as_mut_ptr().add(idx_in_p).write(val);
        }

        for (p, sz) in partitions.iter_mut().zip(partition_sizes) {
            p.set_len(*sz as usize);
        }
    }

    partitions
}

/// # Safety
/// Same as partition_vec.
pub unsafe fn partition_mask(
    m: &Bitmap,
    partition_sizes: &[IdxSize],
    partition_idxs: &[IdxSize],
    idx_in_partition: &[IdxSize],
) -> Vec<MutableBitmap> {
    assert!(idx_in_partition.len() == m.len());
    assert!(partition_idxs.len() == m.len());
        
    let mut partitions = partition_sizes
        .iter()
        .map(|sz| MutableBitmap::from_len_zeroed(*sz as usize))
        .collect_vec();

    unsafe {
        // Scatter into each partition.
        for i in 0..m.len() {
            let p_idx = *partition_idxs.get_unchecked(i) as usize;
            let idx_in_p = *idx_in_partition.get_unchecked(i) as usize;
            debug_assert!(p_idx < partitions.len());
            let p = partitions.get_unchecked_mut(p_idx);
            debug_assert!(idx_in_p < p.capacity());
            p.set_unchecked(idx_in_p, m.get_bit_unchecked(i));
        }
    }

    partitions
}

/// A fused loop of partition_vec and partition_mask.
/// # Safety
/// Same as partition_vec.
pub unsafe fn partition_vec_mask<T>(
    v: Vec<T>,
    m: &Bitmap,
    partition_sizes: &[IdxSize],
    partition_idxs: &[IdxSize],
    idx_in_partition: &[IdxSize],
) -> Vec<(Vec<T>, MutableBitmap)> {
    assert!(idx_in_partition.len() == v.len());
    assert!(partition_idxs.len() == v.len());
    assert!(m.len() == v.len());
        
    let mut partitions = partition_sizes
        .iter()
        .map(|sz| (Vec::<T>::with_capacity(*sz as usize), MutableBitmap::from_len_zeroed(*sz as usize)))
        .collect_vec();

    unsafe {
        // Scatter into each partition.
        for (i, val) in v.into_iter().enumerate() {
            let p_idx = *partition_idxs.get_unchecked(i) as usize;
            let idx_in_p = *idx_in_partition.get_unchecked(i) as usize;
            debug_assert!(p_idx < partitions.len());
            let (pv, pm) = partitions.get_unchecked_mut(p_idx);
            debug_assert!(idx_in_p < pv.capacity());
            debug_assert!(idx_in_p < pm.len());
            pv.as_mut_ptr().add(idx_in_p).write(val);
            pm.set_unchecked(idx_in_p, m.get_bit_unchecked(i));
        }

        for (p, sz) in partitions.iter_mut().zip(partition_sizes) {
            p.0.set_len(*sz as usize);
        }
    }

    partitions
}
