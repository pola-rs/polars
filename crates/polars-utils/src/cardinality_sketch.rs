use crate::algebraic_ops::alg_add_f64;

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
#[derive(Clone)]
pub struct CardinalitySketch {
    buckets: [u8; 256],
}

impl Default for CardinalitySketch {
    fn default() -> Self {
        Self::new()
    }
}

impl CardinalitySketch {
    pub fn new() -> Self {
        Self {
            buckets: [0u8; 256],
        }
    }

    /// Add a new hash to the sketch.
    pub fn insert(&mut self, mut h: u64) {
        const ARBITRARY_ODD: u64 = 0x902813a5785dc787;
        // We multiply by this arbitrarily chosen odd number and then take the
        // top bits to ensure the sketch is influenced by all bits of the hash.
        h = h.wrapping_mul(ARBITRARY_ODD);
        let idx = (h >> 56) as usize;
        let p = 1 + (h << 8).leading_zeros() as u8;
        self.buckets[idx] = self.buckets[idx].max(p);
    }

    pub fn combine(&mut self, other: &CardinalitySketch) {
        self.buckets = std::array::from_fn(|i| std::cmp::max(self.buckets[i], other.buckets[i]));
    }

    pub fn estimate(&self) -> usize {
        let m = 256.0;
        let alpha_m = 0.7123 / (1.0 + 1.079 / m);

        let mut sum = 0.0;
        let mut num_zero = 0;
        for x in self.buckets.iter() {
            sum = alg_add_f64(sum, inv_pow2(*x));
            num_zero += (*x == 0) as usize;
        }

        let est = (alpha_m * m * m) / sum;
        let corr_est = if est <= 5.0 / 2.0 * m && num_zero != 0 {
            // Small cardinality estimate, full 64-bit logarithm is overkill.
            m * (m as f32 / num_zero as f32).ln() as f64
        } else {
            est
        };

        if num_zero == self.buckets.len() {
            0
        } else {
            ((corr_est + 0.5) as usize).max(1)
        }
    }

    pub fn into_state(self) -> [u8; 256] {
        self.buckets
    }

    pub fn from_state(buckets: [u8; 256]) -> Self {
        Self { buckets }
    }
}
