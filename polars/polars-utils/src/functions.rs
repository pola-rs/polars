use std::hash::{BuildHasher, Hash, Hasher};

// Faster than collecting from a flattened iterator.
pub fn flatten<T: Clone, R: AsRef<[T]>>(bufs: &[R], len: Option<usize>) -> Vec<T> {
    let len = len.unwrap_or_else(|| bufs.iter().map(|b| b.as_ref().len()).sum());

    let mut out = Vec::with_capacity(len);
    for b in bufs {
        out.extend_from_slice(b.as_ref());
    }
    out
}

#[inline]
pub fn hash_to_partition(h: u64, n_partitions: usize) -> usize {
    debug_assert!(n_partitions.is_power_of_two());
    // n % 2^i = n & (2^i - 1)
    h as usize & n_partitions.wrapping_sub(1)
}

#[inline]
pub fn get_hash<T: Hash, B: BuildHasher>(value: T, hb: &B) -> u64 {
    let mut hasher = hb.build_hasher();
    value.hash(&mut hasher);
    hasher.finish()
}
