use std::ops::{Add, IndexMut};
#[cfg(feature = "simd")]
use std::simd::{prelude::*, *};

use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use num_traits::{AsPrimitive, Float};

const STRIPE: usize = 16;
const PAIRWISE_RECURSION_LIMIT: usize = 128;

// We want to be generic over both integers and floats, requiring this helper trait.
#[cfg(feature = "simd")]
pub trait SimdCastGeneric<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn cast_generic<U: SimdCast>(self) -> Simd<U, N>;
}

macro_rules! impl_cast_custom {
    ($_type:ty) => {
        #[cfg(feature = "simd")]
        impl<const N: usize> SimdCastGeneric<N> for Simd<$_type, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            fn cast_generic<U: SimdCast>(self) -> Simd<U, N> {
                self.cast::<U>()
            }
        }
    };
}

impl_cast_custom!(u8);
impl_cast_custom!(u16);
impl_cast_custom!(u32);
impl_cast_custom!(u64);
impl_cast_custom!(i8);
impl_cast_custom!(i16);
impl_cast_custom!(i32);
impl_cast_custom!(i64);
impl_cast_custom!(f32);
impl_cast_custom!(f64);

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

// As a trait to not proliferate SIMD bounds.
pub trait SumBlock<F> {
    fn sum_block_vectorized(&self) -> F;
    fn sum_block_vectorized_with_mask(&self, mask: BitMask<'_>) -> F;
}

#[cfg(feature = "simd")]
impl<T, F> SumBlock<F> for [T; PAIRWISE_RECURSION_LIMIT]
where
    T: SimdElement,
    F: SimdElement + SimdCast + Add<Output = F> + Default,
    Simd<T, STRIPE>: SimdCastGeneric<STRIPE>,
    Simd<F, STRIPE>: std::iter::Sum,
{
    fn sum_block_vectorized(&self) -> F {
        let vsum = self
            .chunks_exact(STRIPE)
            .map(|a| Simd::<T, STRIPE>::from_slice(a).cast_generic::<F>())
            .sum::<Simd<F, STRIPE>>();
        vector_horizontal_sum(vsum)
    }

    fn sum_block_vectorized_with_mask(&self, mask: BitMask<'_>) -> F {
        let zero = Simd::default();
        let vsum = self
            .chunks_exact(STRIPE)
            .enumerate()
            .map(|(i, a)| {
                let m: Mask<_, STRIPE> = mask.get_simd(i * STRIPE);
                m.select(Simd::from_slice(a).cast_generic::<F>(), zero)
            })
            .sum::<Simd<F, STRIPE>>();
        vector_horizontal_sum(vsum)
    }
}

#[cfg(feature = "simd")]
impl<F> SumBlock<F> for [i128; PAIRWISE_RECURSION_LIMIT]
where
    i128: AsPrimitive<F>,
    F: Float + std::iter::Sum + 'static,
{
    fn sum_block_vectorized(&self) -> F {
        self.iter().map(|x| x.as_()).sum()
    }

    fn sum_block_vectorized_with_mask(&self, mask: BitMask<'_>) -> F {
        self.iter()
            .enumerate()
            .map(|(idx, x)| if mask.get(idx) { x.as_() } else { F::zero() })
            .sum()
    }
}

#[cfg(not(feature = "simd"))]
impl<T, F> SumBlock<F> for [T; PAIRWISE_RECURSION_LIMIT]
where
    T: AsPrimitive<F> + 'static,
    F: Default + Add<Output = F> + Copy + 'static,
{
    fn sum_block_vectorized(&self) -> F {
        let mut vsum = [F::default(); STRIPE];
        for chunk in self.chunks_exact(STRIPE) {
            for j in 0..STRIPE {
                vsum[j] = vsum[j] + chunk[j].as_();
            }
        }
        vector_horizontal_sum(vsum)
    }

    fn sum_block_vectorized_with_mask(&self, mask: BitMask<'_>) -> F {
        let mut vsum = [F::default(); STRIPE];
        for (i, chunk) in self.chunks_exact(STRIPE).enumerate() {
            for j in 0..STRIPE {
                // Unconditional add with select for better branch-free opts.
                let addend = if mask.get(i * STRIPE + j) {
                    chunk[j].as_()
                } else {
                    F::default()
                };
                vsum[j] = vsum[j] + addend;
            }
        }
        vector_horizontal_sum(vsum)
    }
}

/// Invariant: f.len() % PAIRWISE_RECURSION_LIMIT == 0 and f.len() > 0.
unsafe fn pairwise_sum<F, T>(f: &[T]) -> F
where
    [T; PAIRWISE_RECURSION_LIMIT]: SumBlock<F>,
    F: Add<Output = F>,
{
    debug_assert!(!f.is_empty() && f.len() % PAIRWISE_RECURSION_LIMIT == 0);

    let block: Option<&[T; PAIRWISE_RECURSION_LIMIT]> = f.try_into().ok();
    if let Some(block) = block {
        return block.sum_block_vectorized();
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
unsafe fn pairwise_sum_with_mask<F, T>(f: &[T], mask: BitMask<'_>) -> F
where
    [T; PAIRWISE_RECURSION_LIMIT]: SumBlock<F>,
    F: Add<Output = F>,
{
    debug_assert!(!f.is_empty() && f.len() % PAIRWISE_RECURSION_LIMIT == 0);
    debug_assert!(f.len() == mask.len());

    let block: Option<&[T; PAIRWISE_RECURSION_LIMIT]> = f.try_into().ok();
    if let Some(block) = block {
        return block.sum_block_vectorized_with_mask(mask);
    }

    // SAFETY: see pairwise_sum.
    unsafe {
        let blocks = f.len() / PAIRWISE_RECURSION_LIMIT;
        let left_len = (blocks / 2) * PAIRWISE_RECURSION_LIMIT;
        let (left, right) = (f.get_unchecked(..left_len), f.get_unchecked(left_len..));
        let (left_mask, right_mask) = mask.split_at_unchecked(left_len);
        pairwise_sum_with_mask(left, left_mask) + pairwise_sum_with_mask(right, right_mask)
    }
}

pub trait FloatSum<F>: Sized {
    fn sum(f: &[Self]) -> F;
    fn sum_with_validity(f: &[Self], validity: &Bitmap) -> F;
}

impl<T, F> FloatSum<F> for T
where
    F: Float + std::iter::Sum + 'static,
    T: AsPrimitive<F>,
    [T; PAIRWISE_RECURSION_LIMIT]: SumBlock<F>,
{
    fn sum(f: &[Self]) -> F {
        let remainder = f.len() % PAIRWISE_RECURSION_LIMIT;
        let (rest, main) = f.split_at(remainder);
        let mainsum = if f.len() > remainder {
            unsafe { pairwise_sum(main) }
        } else {
            F::zero()
        };
        // TODO: faster remainder.
        let restsum: F = rest.iter().map(|x| x.as_()).sum();
        mainsum + restsum
    }

    fn sum_with_validity(f: &[Self], validity: &Bitmap) -> F {
        let mask = BitMask::from_bitmap(validity);
        assert!(f.len() == mask.len());

        let remainder = f.len() % PAIRWISE_RECURSION_LIMIT;
        let (rest, main) = f.split_at(remainder);
        let (rest_mask, main_mask) = mask.split_at(remainder);
        let mainsum = if f.len() > remainder {
            unsafe { pairwise_sum_with_mask(main, main_mask) }
        } else {
            F::zero()
        };
        // TODO: faster remainder.
        let restsum: F = rest
            .iter()
            .enumerate()
            .map(|(i, x)| {
                // No filter but rather select of 0.0 for cmov opt.
                if rest_mask.get(i) {
                    x.as_()
                } else {
                    F::zero()
                }
            })
            .sum();
        mainsum + restsum
    }
}

pub fn sum_arr_as_f32<T>(arr: &PrimitiveArray<T>) -> f32
where
    T: NativeType + FloatSum<f32>,
{
    let validity = arr.validity().filter(|_| arr.null_count() > 0);
    if let Some(mask) = validity {
        FloatSum::sum_with_validity(arr.values(), mask)
    } else {
        FloatSum::sum(arr.values())
    }
}

pub fn sum_arr_as_f64<T>(arr: &PrimitiveArray<T>) -> f64
where
    T: NativeType + FloatSum<f64>,
{
    let validity = arr.validity().filter(|_| arr.null_count() > 0);
    if let Some(mask) = validity {
        FloatSum::sum_with_validity(arr.values(), mask)
    } else {
        FloatSum::sum(arr.values())
    }
}
