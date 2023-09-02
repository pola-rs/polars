use std::ops::{Add, IndexMut};
#[cfg(feature = "simd")]
use std::simd::{Mask, Simd, SimdElement, ToBitMask};

use arrow::bitmap::Bitmap;
#[cfg(feature = "simd")]
use num_traits::AsPrimitive;

const STRIPE: usize = 16;
const PAIRWISE_RECURSION_LIMIT: usize = 128;

// Load 8 bytes as little-endian into a u64, padding with zeros if it's too short.
#[cfg(feature = "simd")]
pub fn load_padded_le_u64(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    if len >= 8 {
        return u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    }

    if len >= 4 {
        let lo = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let hi = u32::from_le_bytes(bytes[len - 4..len].try_into().unwrap());
        return (lo as u64) | ((hi as u64) << (8 * (len - 4)));
    }

    if len == 0 {
        return 0;
    }

    let lo = bytes[0] as u64;
    let mid = (bytes[len / 2] as u64) << (8 * (len / 2));
    let hi = (bytes[len - 1] as u64) << (8 * (len - 1));
    lo | mid | hi
}

struct BitMask<'a> {
    bytes: &'a [u8],
    offset: usize,
    len: usize,
}

impl<'a> BitMask<'a> {
    pub fn new(bitmap: &'a Bitmap) -> Self {
        let (bytes, offset, len) = bitmap.as_slice();
        // Check length so we can use unsafe access in our get.
        assert!(bytes.len() * 8 >= len + offset);
        Self { bytes, offset, len }
    }

    fn split_at(&self, idx: usize) -> (Self, Self) {
        assert!(idx <= self.len);
        unsafe { self.split_at_unchecked(idx) }
    }

    unsafe fn split_at_unchecked(&self, idx: usize) -> (Self, Self) {
        debug_assert!(idx <= self.len);
        let left = Self { len: idx, ..*self };
        let right = Self {
            len: self.len - idx,
            offset: self.offset + idx,
            ..*self
        };
        (left, right)
    }

    #[cfg(feature = "simd")]
    pub fn get_simd<T>(&self, idx: usize) -> T
    where
        T: ToBitMask,
        <T as ToBitMask>::BitMask: Copy + 'static,
        u64: AsPrimitive<<T as ToBitMask>::BitMask>,
    {
        // We don't support 64-lane masks because then we couldn't load our
        // bitwise mask as a u64 and then do the byteshift on it.

        let lanes = std::mem::size_of::<T::BitMask>() * 8;
        assert!(lanes < 64);

        let start_byte_idx = (self.offset + idx) / 8;
        let byte_shift = (self.offset + idx) % 8;
        if idx + lanes <= self.len {
            // SAFETY: fast path, we know this is completely in-bounds.
            let mask = load_padded_le_u64(unsafe { self.bytes.get_unchecked(start_byte_idx..) });
            T::from_bitmask((mask >> byte_shift).as_())
        } else if idx < self.len {
            // SAFETY: we know that at least the first byte is in-bounds.
            // This is partially out of bounds, we have to do extra masking.
            let mask = load_padded_le_u64(unsafe { self.bytes.get_unchecked(start_byte_idx..) });
            let num_out_of_bounds = idx + lanes - self.len;
            let shifted = (mask << num_out_of_bounds) >> (num_out_of_bounds + byte_shift);
            T::from_bitmask(shifted.as_())
        } else {
            T::from_bitmask((0u64).as_())
        }
    }

    pub fn get(&self, idx: usize) -> bool {
        let byte_idx = (self.offset + idx) / 8;
        let byte_shift = (self.offset + idx) % 8;

        if idx < self.len {
            // SAFETY: we know this is in-bounds.
            let byte = unsafe { *self.bytes.get_unchecked(byte_idx) };
            (byte >> byte_shift) & 1 == 1
        } else {
            false
        }
    }
}

fn vector_horizontal_sum<V, T>(mut v: V) -> T
where
    V: IndexMut<usize, Output = T>,
    T: Add<T, Output = T> + Sized + Copy,
{
    // We have to be careful about this reduction, floating
    // point math is NOT associative so we have to write this
    // in a form that maps to good shuffle instructions.
    // We fold the vector onto itself, halved, until we are down to
    // four elements which we add in a shuffle-friendly way.
    let mut width = STRIPE;
    while width > 4 {
        for j in 0..width / 2 {
            v[j] = v[j] + v[width / 2 + j];
        }
        width /= 2;
    }

    (v[0] + v[2]) + (v[1] + v[3])
}

#[cfg(feature = "simd")]
fn sum_block_vectorized<T>(f: &[T; PAIRWISE_RECURSION_LIMIT]) -> T
where
    T: SimdElement + Add<Output = T> + Default,
    Simd<T, STRIPE>: std::iter::Sum,
{
    let vsum = f
        .chunks_exact(STRIPE)
        .map(|a| Simd::from_slice(a))
        .sum::<Simd<T, STRIPE>>();
    vector_horizontal_sum(vsum)
}

#[cfg(not(feature = "simd"))]
fn sum_block_vectorized<T>(f: &[T; PAIRWISE_RECURSION_LIMIT]) -> T
where
    T: Default + Add<Output = T> + Copy,
{
    let mut vsum = [T::default(); STRIPE];
    for chunk in f.chunks_exact(STRIPE) {
        for j in 0..STRIPE {
            vsum[j] = vsum[j] + chunk[j];
        }
    }
    vector_horizontal_sum(vsum)
}

#[cfg(feature = "simd")]
fn sum_block_vectorized_with_mask<T>(f: &[T; PAIRWISE_RECURSION_LIMIT], mask: BitMask<'_>) -> T
where
    T: SimdElement + Add<Output = T> + Default,
    Simd<T, STRIPE>: std::iter::Sum,
{
    let zero = Simd::default();
    let vsum = f
        .chunks_exact(STRIPE)
        .enumerate()
        .map(|(i, a)| {
            let m: Mask<_, STRIPE> = mask.get_simd(i * STRIPE);
            m.select(Simd::from_slice(a), zero)
        })
        .sum::<Simd<T, STRIPE>>();
    vector_horizontal_sum(vsum)
}

#[cfg(not(feature = "simd"))]
fn sum_block_vectorized_with_mask<T>(f: &[T; PAIRWISE_RECURSION_LIMIT], mask: BitMask<'_>) -> T
where
    T: Default + Add<Output = T> + Copy,
{
    let mut vsum = [T::default(); STRIPE];
    for (i, chunk) in f.chunks_exact(STRIPE).enumerate() {
        for j in 0..STRIPE {
            // Unconditional add with select for better branch-free opts.
            let addend = if mask.get(i * STRIPE + j) {
                chunk[j]
            } else {
                T::default()
            };
            vsum[j] = vsum[j] + addend;
        }
    }
    vector_horizontal_sum(vsum)
}

macro_rules! def_sum {
    ($T:ty, $mod:ident) => {
        pub mod $mod {
            use super::*;
            /// Invariant: f.len() % PAIRWISE_RECURSION_LIMIT == 0 and f.len() > 0.
            unsafe fn pairwise_sum(f: &[$T]) -> f64 {
                debug_assert!(f.len() > 0 && f.len() % PAIRWISE_RECURSION_LIMIT == 0);

                if let Ok(block) = f.try_into() {
                    return sum_block_vectorized(block) as f64;
                }

                // SAFETY: we maintain the invariant. `try_into` array of len PAIRWISE_RECURSION_LIMIT
                // failed so we know f.len() >= 2*PAIRWISE_RECURSION_LIMIT, and thus blocks >= 2.
                // This means 0 < left_len < f.len() and left_len is divisible by PAIRWISE_RECURSION_LIMIT,
                // maintaining the invariant for both recursive calls.
                unsafe {
                    let blocks = f.len() / PAIRWISE_RECURSION_LIMIT;
                    let left_len = (blocks / 2) * PAIRWISE_RECURSION_LIMIT;
                    let (left, right) = (f.get_unchecked(..left_len), f.get_unchecked(left_len..));
                    pairwise_sum(left) + pairwise_sum(right)
                }
            }

            /// Invariant: f.len() % PAIRWISE_RECURSION_LIMIT == 0 and f.len() > 0.
            /// Also, f.len() == mask.len().
            unsafe fn pairwise_sum_with_mask(f: &[$T], mask: BitMask<'_>) -> f64 {
                debug_assert!(f.len() > 0 && f.len() % PAIRWISE_RECURSION_LIMIT == 0);
                debug_assert!(f.len() == mask.len);

                if let Ok(block) = f.try_into() {
                    return sum_block_vectorized_with_mask(block, mask) as f64;
                }

                // SAFETY: see pairwise_sum.
                unsafe {
                    let blocks = f.len() / PAIRWISE_RECURSION_LIMIT;
                    let left_len = (blocks / 2) * PAIRWISE_RECURSION_LIMIT;
                    let (left, right) = (f.get_unchecked(..left_len), f.get_unchecked(left_len..));
                    let (left_mask, right_mask) = mask.split_at_unchecked(left_len);
                    pairwise_sum_with_mask(left, left_mask)
                        + pairwise_sum_with_mask(right, right_mask)
                }
            }

            pub fn sum(f: &[$T]) -> f64 {
                let remainder = f.len() % PAIRWISE_RECURSION_LIMIT;
                let (rest, main) = f.split_at(remainder);
                let mainsum = if f.len() > remainder {
                    unsafe { pairwise_sum(main) }
                } else {
                    0.0
                };
                // TODO: faster remainder.
                let restsum: f64 = rest.iter().map(|x| *x as f64).sum();
                mainsum + restsum
            }

            pub fn sum_with_validity(f: &[$T], validity: &Bitmap) -> f64 {
                let mask = BitMask::new(validity);
                assert!(f.len() == mask.len);

                let remainder = f.len() % PAIRWISE_RECURSION_LIMIT;
                let (rest, main) = f.split_at(remainder);
                let (rest_mask, main_mask) = mask.split_at(remainder);
                let mainsum = if f.len() > remainder {
                    unsafe { pairwise_sum_with_mask(main, main_mask) }
                } else {
                    0.0
                };
                // TODO: faster remainder.
                let restsum: f64 = rest
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        // No filter but rather select of 0.0 for cmov opt.
                        if rest_mask.get(i) {
                            *x as f64
                        } else {
                            0.0
                        }
                    })
                    .sum();
                mainsum + restsum
            }
        }
    };
}

def_sum!(f32, f32);
def_sum!(f64, f64);
