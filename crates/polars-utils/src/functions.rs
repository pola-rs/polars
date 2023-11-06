use std::hash::{BuildHasher, Hash};
use std::ops::Range;


#[inline(always)]
pub fn extract_highest_bit(n: usize) -> usize {
    if n <= 1 { return n; }
    1 << (usize::BITS - 1 - n.leading_zeros())
}

// The ith portion of a range split in k (as equal as possible) parts.
#[inline(always)]
pub fn range_portion(i: usize, k: usize, r: Range<usize>) -> Range<usize> {
    // Each portion having size n / k leaves n % k elements unaccounted for.
    // Make the first n % k portions have 1 extra element.
    let n = r.len();
    let base_size = n / k;
    let num_one_larger = n % k;
    let num_before = base_size * i + i.min(num_one_larger);
    let our_size = base_size + (i < num_one_larger) as usize;
    r.start + num_before..r.start + num_before + our_size
}

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
    // Now, assuming h is a 64-bit random number, we note that
    // h / 2^64 is almost a uniform random number in [0, 1), and thus
    // floor(h * n_partitions / 2^64) is almost a uniform random integer in
    // [0, n_partitions). Despite being written with u128 multiplication this
    // compiles to a single mul / mulhi instruction on x86-x64/aarch64.
    ((h as u128 * n_partitions as u128) >> 64) as usize
}

#[inline]
pub fn get_hash<T: Hash, B: BuildHasher>(value: T, hb: &B) -> u64 {
    hb.hash_one(value)
}
