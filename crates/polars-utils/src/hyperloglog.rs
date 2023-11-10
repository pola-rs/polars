// Computes 2^-n by directly subtracting from the IEEE754 double exponent.
fn inv_pow2(n: u8) -> f64 {
    let base = f64::to_bits(1.0);
    f64::from_bits(base - ((n as u64) << 52))
}

/// HyperLogLog in Practice: Algorithmic Engineering of
/// a State of The Art Cardinality Estimation Algorithm
/// Stefan Heule, Marc Nunkesser, Alexander Hall
///
/// We use m = 256 which gives a relative error of ~6.5% of the cardinality
/// estimate. We don't bother with stuffing the counts in 6 bits, byte access is
/// fast.
///
/// The bias correction described in the paper is not implemented, so this is
/// somewhere in between HyperLogLog and HyperLogLog++.
pub struct HyperLogLog {
    buckets: Box<[u8; 256]>,
    multiplier: u64,
}

impl HyperLogLog {
    pub fn with_seed(seed: u64) -> Self {
        Self {
            buckets: Box::new([0; 256]),
            multiplier: seed | 1,
        }
    }

    /// Add a new hash to the sketch.
    pub fn insert(&mut self, mut h: u64) {
        // We do a bit of mixing on the hash, in case it was already used to
        // partition the data which would lead to skewed estimates on the
        // partition. Different multipliers also de-bias different sketches from
        // each other.
        h ^= h >> 32;
        h = h.wrapping_mul(self.multiplier);

        // The top bits should be high quality now.
        let idx = (h >> 56) as usize;
        let p = 1 + (h << 8).leading_zeros() as u8;
        self.buckets[idx] = self.buckets[idx].max(p);
    }

    pub fn estimate(&self) -> usize {
        let m = 256.0;
        let alpha_m = 0.7123 / (1.0 + 1.079 / m);

        let mut sum = 0.0;
        let mut num_zero = 0;
        for x in self.buckets.iter() {
            sum += inv_pow2(*x);
            num_zero += (*x == 0) as usize;
        }

        let est = (alpha_m * m * m) / sum;
        let corr_est = if est <= 5.0 / 2.0 * m && num_zero != 0 {
            // Small cardinality estimate, full 64-bit logarithm is overkill.
            m * (m as f32 / num_zero as f32).ln() as f64
        } else {
            est
        };

        corr_est as usize
    }
}
