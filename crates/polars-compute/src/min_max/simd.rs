use std::simd::prelude::*;
use std::simd::{LaneCount, SimdElement, SupportedLaneCount};

use arrow::array::PrimitiveArray;
use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_utils::min_max::MinMax;

use super::MinMaxKernel;

fn scalar_reduce_min_propagate_nan<T: MinMax + Copy, const N: usize>(arr: &[T; N]) -> T {
    let it = arr.iter().copied();
    it.reduce(MinMax::min_propagate_nan).unwrap()
}

fn scalar_reduce_max_propagate_nan<T: MinMax + Copy, const N: usize>(arr: &[T; N]) -> T {
    let it = arr.iter().copied();
    it.reduce(MinMax::max_propagate_nan).unwrap()
}

fn fold_agg_kernel<const N: usize, T, F>(
    arr: &[T],
    validity: Option<&Bitmap>,
    scalar_identity: T,
    mut simd_f: F,
) -> Option<Simd<T, N>>
where
    T: SimdElement + NativeType,
    F: FnMut(Simd<T, N>, Simd<T, N>) -> Simd<T, N>,
    LaneCount<N>: SupportedLaneCount,
{
    let mut arr_chunks = arr.chunks_exact(N);

    let identity = Simd::splat(scalar_identity);
    let mut state = identity;
    if let Some(valid) = validity {
        if valid.unset_bits() == arr.len() {
            return None;
        }

        let mask = BitMask::from_bitmap(valid);
        let mut offset = 0;
        for c in arr_chunks.by_ref() {
            let m: Mask<_, N> = mask.get_simd(offset);
            state = simd_f(state, m.select(Simd::from_slice(c), identity));
            offset += N;
        }
        if arr.len() % N > 0 {
            let mut rest: [T; N] = identity.to_array();
            let arr_rest = arr_chunks.remainder();
            rest[..arr_rest.len()].copy_from_slice(arr_rest);
            let m: Mask<_, N> = mask.get_simd(offset);
            state = simd_f(state, m.select(Simd::from_array(rest), identity));
        }
    } else {
        for c in arr_chunks.by_ref() {
            state = simd_f(state, Simd::from_slice(c));
        }
        if arr.len() % N > 0 {
            let mut rest: [T; N] = identity.to_array();
            let arr_rest = arr_chunks.remainder();
            rest[..arr_rest.len()].copy_from_slice(arr_rest);
            state = simd_f(state, Simd::from_array(rest));
        }
    }

    Some(state)
}

macro_rules! impl_min_max_kernel_int {
    ($T:ty, $N:literal) => {
        impl MinMaxKernel for PrimitiveArray<$T> {
            type Scalar<'a> = $T;

            fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                fold_agg_kernel::<$N, $T, _>(self.values(), self.validity(), <$T>::MAX, |a, b| {
                    a.simd_min(b)
                })
                .map(|s| s.reduce_min())
            }

            fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                fold_agg_kernel::<$N, $T, _>(self.values(), self.validity(), <$T>::MIN, |a, b| {
                    a.simd_max(b)
                })
                .map(|s| s.reduce_max())
            }

            fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                self.min_ignore_nan_kernel()
            }

            fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                self.max_ignore_nan_kernel()
            }
        }

        impl MinMaxKernel for [$T] {
            type Scalar<'a> = $T;

            fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                fold_agg_kernel::<$N, $T, _>(self, None, <$T>::MAX, |a, b| a.simd_min(b))
                    .map(|s| s.reduce_min())
            }

            fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                fold_agg_kernel::<$N, $T, _>(self, None, <$T>::MIN, |a, b| a.simd_max(b))
                    .map(|s| s.reduce_max())
            }

            fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                self.min_ignore_nan_kernel()
            }

            fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                self.max_ignore_nan_kernel()
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
            type Scalar<'a> = $T;

            fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                fold_agg_kernel::<$N, $T, _>(
                    self.values(),
                    self.validity(),
                    <$T>::INFINITY,
                    |a, b| a.simd_min(b),
                )
                .map(|s| s.reduce_min())
            }

            fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                fold_agg_kernel::<$N, $T, _>(
                    self.values(),
                    self.validity(),
                    <$T>::NEG_INFINITY,
                    |a, b| a.simd_max(b),
                )
                .map(|s| s.reduce_max())
            }

            fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                fold_agg_kernel::<$N, $T, _>(
                    self.values(),
                    self.validity(),
                    <$T>::INFINITY,
                    |a, b| (a.simd_lt(b) | a.simd_ne(a)).select(a, b),
                )
                .map(|s| scalar_reduce_min_propagate_nan(s.as_array()))
            }

            fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                fold_agg_kernel::<$N, $T, _>(
                    self.values(),
                    self.validity(),
                    <$T>::NEG_INFINITY,
                    |a, b| (a.simd_gt(b) | a.simd_ne(a)).select(a, b),
                )
                .map(|s| scalar_reduce_max_propagate_nan(s.as_array()))
            }
        }

        impl MinMaxKernel for [$T] {
            type Scalar<'a> = $T;

            fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                fold_agg_kernel::<$N, $T, _>(self, None, <$T>::INFINITY, |a, b| a.simd_min(b))
                    .map(|s| s.reduce_min())
            }

            fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                fold_agg_kernel::<$N, $T, _>(self, None, <$T>::NEG_INFINITY, |a, b| a.simd_max(b))
                    .map(|s| s.reduce_max())
            }

            fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                fold_agg_kernel::<$N, $T, _>(self, None, <$T>::INFINITY, |a, b| {
                    (a.simd_lt(b) | a.simd_ne(a)).select(a, b)
                })
                .map(|s| scalar_reduce_min_propagate_nan(s.as_array()))
            }

            fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                fold_agg_kernel::<$N, $T, _>(self, None, <$T>::NEG_INFINITY, |a, b| {
                    (a.simd_gt(b) | a.simd_ne(a)).select(a, b)
                })
                .map(|s| scalar_reduce_max_propagate_nan(s.as_array()))
            }
        }
    };
}

impl_min_max_kernel_float!(f32, 16);
impl_min_max_kernel_float!(f64, 8);
