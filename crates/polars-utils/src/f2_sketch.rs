const NUM_BUCKETS: usize = 64;

/// Estimates the second frequency moment F2 = sum_k f_k^2 of a stream of hashed
/// keys, where f_k is the number of times key k was inserted.
///
/// This is a single row of a Count-Sketch (the "fast AMS" sketch): every hash
/// adds +1 or -1 to one of 64 buckets, and the sum of squared buckets is an
/// unbiased estimate of F2 with a relative standard error of ~18%.
#[derive(Clone)]
pub struct F2Sketch {
    buckets: [i64; NUM_BUCKETS],
    num_inserts: u64,
}

impl Default for F2Sketch {
    fn default() -> Self {
        Self::new()
    }
}

impl F2Sketch {
    pub fn new() -> Self {
        Self {
            buckets: [0; NUM_BUCKETS],
            num_inserts: 0,
        }
    }

    /// Add a new hash to the sketch.
    #[inline]
    pub fn insert(&mut self, h: u64) {
        // An arbitrary odd number different from the one in the cardinality
        // sketch, so the two use independent bits.
        const ARBITRARY_ODD: u64 = 0xd6e8feb86659fd93;
        let x = h.wrapping_mul(ARBITRARY_ODD);
        let idx = (x >> 58) as usize;
        let negative = ((x >> 57) & 1) as i64;
        self.buckets[idx] += 1 - 2 * negative;
        self.num_inserts += 1;
    }

    pub fn num_inserts(&self) -> u64 {
        self.num_inserts
    }

    pub fn estimate(&self) -> f64 {
        self.buckets.iter().map(|c| (*c as f64) * (*c as f64)).sum()
    }
}
