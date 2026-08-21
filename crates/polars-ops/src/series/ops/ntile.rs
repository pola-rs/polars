use std::num::NonZeroU32;

use polars_core::prelude::*;

/// Assign each of `r` ordered rows to one of `n` buckets of equal size, numbered `1..=n`.
pub fn ntile(r: IdxSize, n: NonZeroU32) -> IdxCa {
    let n = IdxSize::from(n.get());
    let mut out: Vec<IdxSize> = Vec::with_capacity(r as usize);
    if r <= n {
        // fast-path: every bucket holds at *most* one row
        out.extend(1..=r);
    } else {
        // `r > n`, so no bucket is empty and every size is at *least* one
        let (quotient, remainder) = (r / n, r % n);
        for bucket in 1..=n {
            let size = quotient + IdxSize::from(bucket <= remainder);
            out.extend(std::iter::repeat_n(bucket, size as usize));
        }
    }
    debug_assert_eq!(out.len(), r as usize);
    IdxCa::from_vec(PlSmallStr::EMPTY, out)
}
