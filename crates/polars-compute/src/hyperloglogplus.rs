//! # HyperLogLogPlus
//!
//! `hyperloglogplus` module contains implementation of HyperLogLogPlus
//! algorithm for cardinality estimation so that [`crate::series::approx_n_unique`] function can
//! be efficiently implemented.
//!
//! This module borrows code from [arrow-datafusion](https://github.com/apache/arrow-datafusion/blob/93771052c5ac31f2cf22b8c25bf938656afe1047/datafusion/physical-expr/src/aggregate/hyperloglog.rs).
//!
//! # Examples
//!
//! ```
//!     # use polars_ops::prelude::*;
//!     let mut hllp = HyperLogLog::new();
//!     hllp.add(&12345);
//!     hllp.add(&23456);
//!
//!     assert_eq!(hllp.count(), 2);
//! ```

use std::hash::Hash;
use std::marker::PhantomData;

use polars_utils::aliases::PlRandomStateQuality;

/// The greater is P, the smaller the error.
const HLL_P: usize = 14_usize;
/// The number of bits of the hash value used determining the number of leading zeros
const HLL_Q: usize = 64_usize - HLL_P;
const NUM_REGISTERS: usize = 1_usize << HLL_P;
/// Mask to obtain index into the registers
const HLL_P_MASK: u64 = (NUM_REGISTERS as u64) - 1;

#[derive(Clone, Debug)]
pub struct HyperLogLog<T>
where
    T: Hash + ?Sized,
{
    registers: [u8; NUM_REGISTERS],
    phantom: PhantomData<T>,
}

impl<T> Default for HyperLogLog<T>
where
    T: Hash + ?Sized,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed seed for the hashing so that values are consistent across runs
///
/// Note that when we later move on to have serialized HLL register binaries
/// shared across cluster, this SEED will have to be consistent across all
/// parties otherwise we might have corruption. So ideally for later this seed
/// shall be part of the serialized form (or stay unchanged across versions).
const SEED: PlRandomStateQuality = PlRandomStateQuality::with_seeds(
    0x885f6cab121d01a3_u64,
    0x71e4379f2976ad8f_u64,
    0xbf30173dd28a8816_u64,
    0x0eaea5d736d733a4_u64,
);

impl<T> HyperLogLog<T>
where
    T: Hash + ?Sized,
{
    /// Creates a new, empty HyperLogLog.
    pub fn new() -> Self {
        let registers = [0; NUM_REGISTERS];
        Self::new_with_registers(registers)
    }

    /// Creates a HyperLogLog from already populated registers
    /// note that this method should not be invoked in untrusted environment
    /// because the internal structure of registers are not examined.
    pub(crate) fn new_with_registers(registers: [u8; NUM_REGISTERS]) -> Self {
        Self {
            registers,
            phantom: PhantomData,
        }
    }

    #[inline]
    fn hash_value(&self, obj: &T) -> u64 {
        SEED.hash_one(obj)
    }

    /// Adds an element to the HyperLogLog.
    pub fn add(&mut self, obj: &T) {
        let hash = self.hash_value(obj);
        let index = (hash & HLL_P_MASK) as usize;
        let p = ((hash >> HLL_P) | (1_u64 << HLL_Q)).trailing_zeros() + 1;
        self.registers[index] = self.registers[index].max(p as u8);
    }

    /// Get the register histogram (each value in register index into
    /// the histogram; u32 is enough because we only have 2**14=16384 registers
    #[inline]
    fn get_histogram(&self) -> [u32; HLL_Q + 2] {
        let mut histogram = [0; HLL_Q + 2];
        // hopefully this can be unrolled
        for r in self.registers {
            histogram[r as usize] += 1;
        }
        histogram
    }

    /// Merge the other [`HyperLogLog`] into this one
    pub fn merge(&mut self, other: &HyperLogLog<T>) {
        assert!(
            self.registers.len() == other.registers.len(),
            "unexpected got unequal register size, expect {}, got {}",
            self.registers.len(),
            other.registers.len()
        );
        for i in 0..self.registers.len() {
            self.registers[i] = self.registers[i].max(other.registers[i]);
        }
    }

    /// Guess the number of unique elements seen by the HyperLogLog.
    pub fn count(&self) -> usize {
        let histogram = self.get_histogram();
        let m = NUM_REGISTERS as f64;
        let mut z = m * hll_tau((m - histogram[HLL_Q + 1] as f64) / m);
        for i in histogram[1..=HLL_Q].iter().rev() {
            z += *i as f64;
            z *= 0.5;
        }
        z += m * hll_sigma(histogram[0] as f64 / m);
        (0.5 / 2_f64.ln() * m * m / z).round() as usize
    }
}

/// Helper function sigma as defined in
/// "New cardinality estimation algorithms for HyperLogLog sketches"
/// Otmar Ertl, arXiv:1702.01284
#[inline]
fn hll_sigma(x: f64) -> f64 {
    if x == 1. {
        f64::INFINITY
    } else {
        let mut y = 1.0;
        let mut z = x;
        let mut x = x;
        loop {
            x *= x;
            let z_prime = z;
            z += x * y;
            y += y;
            if z_prime == z {
                break;
            }
        }
        z
    }
}

/// Helper function tau as defined in
/// "New cardinality estimation algorithms for HyperLogLog sketches"
/// Otmar Ertl, arXiv:1702.01284
#[inline]
fn hll_tau(x: f64) -> f64 {
    if x == 0.0 || x == 1.0 {
        0.0
    } else {
        let mut y = 1.0;
        let mut z = 1.0 - x;
        let mut x = x;
        loop {
            x = x.sqrt();
            let z_prime = z;
            y *= 0.5;
            z -= (1.0 - x).powi(2) * y;
            if z_prime == z {
                break;
            }
        }
        z / 3.0
    }
}

impl<T> AsRef<[u8]> for HyperLogLog<T>
where
    T: Hash + ?Sized,
{
    fn as_ref(&self) -> &[u8] {
        &self.registers
    }
}

impl<T> Extend<T> for HyperLogLog<T>
where
    T: Hash,
{
    fn extend<S: IntoIterator<Item = T>>(&mut self, iter: S) {
        for elem in iter {
            self.add(&elem);
        }
    }
}

impl<'a, T> Extend<&'a T> for HyperLogLog<T>
where
    T: 'a + Hash + ?Sized,
{
    fn extend<S: IntoIterator<Item = &'a T>>(&mut self, iter: S) {
        for elem in iter {
            self.add(elem);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HyperLogLog, NUM_REGISTERS};

    fn compare_with_delta(got: usize, expected: usize) {
        let expected = expected as f64;
        let diff = (got as f64) - expected;
        let diff = diff.abs() / expected;
        // times 6 because we want the tests to be stable
        // so we allow a rather large margin of error
        // this is adopted from redis's unit test version as well
        let margin = 1.04 / ((NUM_REGISTERS as f64).sqrt()) * 6.0;
        assert!(
            diff <= margin,
            "{} is not near {} percent of {} which is ({}, {})",
            got,
            margin,
            expected,
            expected * (1.0 - margin),
            expected * (1.0 + margin)
        );
    }

    macro_rules! sized_number_test {
        ($SIZE: expr, $T: tt) => {{
            let mut hll = HyperLogLog::<$T>::new();
            for i in 0..$SIZE {
                hll.add(&i);
            }
            compare_with_delta(hll.count(), $SIZE);
        }};
    }

    macro_rules! typed_large_number_test {
        ($SIZE: expr) => {{
            sized_number_test!($SIZE, u64);
            sized_number_test!($SIZE, u128);
            sized_number_test!($SIZE, i64);
            sized_number_test!($SIZE, i128);
        }};
    }

    macro_rules! typed_number_test {
        ($SIZE: expr) => {{
            sized_number_test!($SIZE, u16);
            sized_number_test!($SIZE, u32);
            sized_number_test!($SIZE, i16);
            sized_number_test!($SIZE, i32);
            typed_large_number_test!($SIZE);
        }};
    }

    #[test]
    fn test_empty() {
        let hll = HyperLogLog::<u64>::new();
        assert_eq!(hll.count(), 0);
    }

    #[test]
    fn test_one() {
        let mut hll = HyperLogLog::<u64>::new();
        hll.add(&1);
        assert_eq!(hll.count(), 1);
    }

    #[test]
    fn test_number_100() {
        typed_number_test!(100);
    }

    #[test]
    fn test_number_1k() {
        typed_number_test!(1_000);
    }

    #[test]
    fn test_number_10k() {
        typed_number_test!(10_000);
    }

    #[test]
    fn test_number_100k() {
        typed_large_number_test!(100_000);
    }

    #[test]
    fn test_number_1m() {
        typed_large_number_test!(1_000_000);
    }

    #[test]
    fn test_u8() {
        let mut hll = HyperLogLog::<[u8]>::new();
        for i in 0..1000 {
            let s = i.to_string();
            let b = s.as_bytes();
            hll.add(b);
        }
        compare_with_delta(hll.count(), 1000);
    }

    #[test]
    fn test_string() {
        let mut hll = HyperLogLog::<String>::new();
        hll.extend((0..1000).map(|i| i.to_string()));
        compare_with_delta(hll.count(), 1000);
    }

    #[test]
    fn test_empty_merge() {
        let mut hll = HyperLogLog::<u64>::new();
        hll.merge(&HyperLogLog::<u64>::new());
        assert_eq!(hll.count(), 0);
    }

    #[test]
    fn test_merge_overlapped() {
        let mut hll = HyperLogLog::<String>::new();
        hll.extend((0..1000).map(|i| i.to_string()));

        let mut other = HyperLogLog::<String>::new();
        other.extend((0..1000).map(|i| i.to_string()));

        hll.merge(&other);
        compare_with_delta(hll.count(), 1000);
    }

    #[test]
    fn test_repetition() {
        let mut hll = HyperLogLog::<u32>::new();
        for i in 0..1_000_000 {
            hll.add(&(i % 1000));
        }
        compare_with_delta(hll.count(), 1000);
    }
}
