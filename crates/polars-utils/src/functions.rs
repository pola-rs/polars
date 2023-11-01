use std::hash::{BuildHasher, Hash};
use std::ops::Range;


#[inline(always)]
pub fn extract_highest_bit(n: usize) -> usize {
    if n <= 1 { return n; }
    1 << (usize::BITS - 1 - n.leading_zeros())
}

pub fn sqrt_approx(n: usize) -> usize {
    // Note that sqrt(n) = n^(1/2), and that 2^log2(n) = n. We combine these
    // two facts to approximate sqrt(n) as 2^(log2(n) / 2). Because our integer
    // log floors we want to add 0.5 to compensate for this on average, so our
    // initial approximation is 2^((1 + floor(log2(n))) / 2).
    //
    // We then apply an iteration of Newton's method to improve our
    // approximation, which for sqrt(n) is a1 = (a0 + n / a0) / 2.
    //
    // Finally we note that the exponentiation / division can be done directly
    // with shifts. We OR with 1 to avoid zero-checks in the integer log.
    let ilog = (n | 1).ilog2();
    let shift = (1 + ilog) / 2;
    ((1 << shift) + (n >> shift)) / 2
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
