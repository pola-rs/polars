use std::ops::Range;

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
