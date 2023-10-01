use std::ops::{Add, IndexMut};
#[cfg(feature = "simd")]
use std::simd::{Mask, Simd, SimdElement};

use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::Bitmap;

const STRIPE: usize = 16;
const PAIRWISE_RECURSION_LIMIT: usize = 128;

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
                debug_assert!(f.len() == mask.len());

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
                let mask = BitMask::from_bitmap(validity);
                assert!(f.len() == mask.len());

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
