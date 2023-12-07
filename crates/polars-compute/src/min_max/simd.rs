use std::simd::*;

use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::bitmask::BitMask;
use arrow::types::NativeType;
use num_traits::AsPrimitive;
use polars_utils::min_max::MinMax;

use super::MinMaxKernel;

fn fold_agg_kernel<const N: usize, T, F>(
    arr: &PrimitiveArray<T>,
    identity: Simd<T, N>,
    mut simd_f: F,
) -> Option<Simd<T, N>>
where
    T: SimdElement + NativeType,
    F: FnMut(Simd<T, N>, Simd<T, N>) -> Simd<T, N>,
    LaneCount<N>: SupportedLaneCount,
    Mask<<T as SimdElement>::Mask, N>: ToBitMask,
    <Mask<<T as SimdElement>::Mask, N> as ToBitMask>::BitMask: Copy + 'static,
    u64: AsPrimitive<<Mask<<T as SimdElement>::Mask, N> as ToBitMask>::BitMask>,
{
    if arr.null_count() == arr.len() {
        return None;
    }

    let buf = arr.values().as_slice();
    let mut buf_chunks = buf.chunks_exact(N);

    let mut state = identity;
    if arr.null_count() == 0 {
        for c in buf_chunks.by_ref() {
            state = simd_f(state, Simd::from_slice(c));
        }
        if arr.len() % N > 0 {
            let mut rest: [T; N] = identity.to_array();
            rest.copy_from_slice(buf_chunks.remainder());
            state = simd_f(state, Simd::from_array(rest));
        }
    } else {
        let mask = BitMask::from_bitmap(arr.validity().unwrap());
        let mut offset = 0;
        for c in buf_chunks.by_ref() {
            let m: Mask<_, N> = mask.get_simd(offset);
            state = simd_f(state, m.select(Simd::from_slice(c), identity));
            offset += N;
        }
        if arr.len() % N > 0 {
            let mut rest: [T; N] = identity.to_array();
            rest.copy_from_slice(buf_chunks.remainder());
            let m: Mask<_, N> = mask.get_simd(offset);
            state = simd_f(state, m.select(Simd::from_array(rest), identity));
        }
    }

    Some(state)
}

macro_rules! impl_min_max_kernel_int {
    ($T:ty, $N:literal) => {
        impl MinMaxKernel for PrimitiveArray<$T> {
            type Scalar = $T;

            fn min_ignore_nan(&self) -> Option<Self::Scalar> {
                fold_agg_kernel::<$N, $T, _>(self, Simd::splat(<$T>::MAX), |a, b| a.simd_min(b))
                    .map(|s| s.reduce_min())
            }

            fn max_ignore_nan(&self) -> Option<Self::Scalar> {
                fold_agg_kernel::<$N, $T, _>(self, Simd::splat(<$T>::MIN), |a, b| a.simd_max(b))
                    .map(|s| s.reduce_max())
            }

            fn min_propagate_nan(&self) -> Option<Self::Scalar> {
                self.min_ignore_nan()
            }

            fn max_propagate_nan(&self) -> Option<Self::Scalar> {
                self.max_ignore_nan()
            }
        }
    };
}

impl_min_max_kernel_int!(u8, 32);
impl_min_max_kernel_int!(u16, 16);
impl_min_max_kernel_int!(u32, 16);
impl_min_max_kernel_int!(u64, 8);
impl_min_max_kernel_int!(i8, 32);
impl_min_max_kernel_int!(i16, 16);
impl_min_max_kernel_int!(i32, 16);
impl_min_max_kernel_int!(i64, 8);

macro_rules! impl_min_max_kernel_float {
    ($T:ty, $N:literal) => {
        impl MinMaxKernel for PrimitiveArray<$T> {
            type Scalar = $T;

            fn min_ignore_nan(&self) -> Option<Self::Scalar> {
                fold_agg_kernel::<$N, $T, _>(self, Simd::splat(<$T>::INFINITY), |a, b| {
                    a.simd_min(b)
                })
                .map(|s| s.reduce_min())
            }

            fn max_ignore_nan(&self) -> Option<Self::Scalar> {
                fold_agg_kernel::<$N, $T, _>(self, Simd::splat(<$T>::NEG_INFINITY), |a, b| {
                    a.simd_max(b)
                })
                .map(|s| s.reduce_max())
            }

            fn min_propagate_nan(&self) -> Option<Self::Scalar> {
                fold_agg_kernel::<$N, $T, _>(self, Simd::splat(<$T>::INFINITY), |a, b| {
                    let a_is_nan = a.simd_ne(a);
                    (a.simd_lt(b) | a_is_nan).select(a, b)
                })
                .map(|s| {
                    s.to_array()
                        .into_iter()
                        .reduce(<$T>::min_propagate_nan)
                        .unwrap()
                })
            }

            fn max_propagate_nan(&self) -> Option<Self::Scalar> {
                fold_agg_kernel::<$N, $T, _>(self, Simd::splat(<$T>::NEG_INFINITY), |a, b| {
                    let a_is_nan = a.simd_ne(a);
                    (a.simd_gt(b) | a_is_nan).select(a, b)
                })
                .map(|s| {
                    s.to_array()
                        .into_iter()
                        .reduce(<$T>::max_propagate_nan)
                        .unwrap()
                })
            }
        }
    };
}

impl_min_max_kernel_float!(f32, 16);
impl_min_max_kernel_float!(f64, 8);
