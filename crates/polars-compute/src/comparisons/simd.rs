use std::simd::prelude::{Simd, SimdPartialEq, SimdPartialOrd};

use arrow::array::PrimitiveArray;
use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use bytemuck::Pod;

use super::TotalOrdKernel;

fn apply_binary_kernel<const N: usize, M: Pod, T, F>(
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    mut f: F,
) -> Bitmap
where
    T: NativeType,
    F: FnMut(&[T; N], &[T; N]) -> M,
{
    assert!(std::mem::size_of::<M>() == N);
    assert!(lhs.len() == rhs.len());
    let n = lhs.len();

    let lhs_buf = lhs.values().as_slice();
    let rhs_buf = rhs.values().as_slice();
    let mut lhs_chunks = lhs_buf.chunks_exact(N);
    let mut rhs_chunks = rhs_buf.chunks_exact(N);
    let mut v = Vec::with_capacity(n.div_ceil(N));
    v.extend(
        lhs_chunks
            .by_ref()
            .zip(rhs_chunks.by_ref())
            .map(|(l, r)| unsafe {
                f(
                    l.try_into().unwrap_unchecked(),
                    r.try_into().unwrap_unchecked(),
                )
            }),
    );

    if n % N > 0 {
        let mut l: [T; N] = [T::zeroed(); N];
        let mut r: [T; N] = [T::zeroed(); N];
        l.copy_from_slice(lhs_chunks.remainder());
        r.copy_from_slice(rhs_chunks.remainder());
        v.push(f(&l, &r));
    }

    Bitmap::from_u8_vec(bytemuck::cast_vec(v), n)
}

fn apply_unary_kernel<const N: usize, M: Pod, T, F>(arg: &PrimitiveArray<T>, mut f: F) -> Bitmap
where
    T: NativeType,
    F: FnMut(&[T; N]) -> M,
{
    assert!(std::mem::size_of::<M>() == N);
    let n = arg.len();

    let arg_buf = arg.values().as_slice();
    let mut arg_chunks = arg_buf.chunks_exact(N);
    let mut v = Vec::with_capacity(n.div_ceil(N));
    v.extend(
        arg_chunks
            .by_ref()
            .map(|l| unsafe { f(l.try_into().unwrap_unchecked()) }),
    );

    if n % N > 0 {
        let mut l: [T; N] = [T::zeroed(); N];
        l.copy_from_slice(arg_chunks.remainder());
        v.push(f(&l));
    }

    Bitmap::from_u8_vec(bytemuck::cast_vec(v), n)
}

macro_rules! impl_int_total_ord_kernel {
    ($T: ty, $width: literal, $mask: ty) => {
        impl TotalOrdKernel for PrimitiveArray<$T> {
            type Scalar = $T;

            fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
                apply_binary_kernel::<$width, $mask, _, _>(self, other, |l, r| {
                    Simd::from(*l).simd_eq(Simd::from(*r)).to_bitmask() as $mask
                })
            }

            fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
                apply_binary_kernel::<$width, $mask, _, _>(self, other, |l, r| {
                    Simd::from(*l).simd_ne(Simd::from(*r)).to_bitmask() as $mask
                })
            }

            fn tot_lt_kernel(&self, other: &Self) -> Bitmap {
                apply_binary_kernel::<$width, $mask, _, _>(self, other, |l, r| {
                    Simd::from(*l).simd_lt(Simd::from(*r)).to_bitmask() as $mask
                })
            }

            fn tot_le_kernel(&self, other: &Self) -> Bitmap {
                apply_binary_kernel::<$width, $mask, _, _>(self, other, |l, r| {
                    Simd::from(*l).simd_le(Simd::from(*r)).to_bitmask() as $mask
                })
            }

            fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                let r = Simd::splat(*other);
                apply_unary_kernel::<$width, $mask, _, _>(self, |l| {
                    Simd::from(*l).simd_eq(r).to_bitmask() as $mask
                })
            }

            fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                let r = Simd::splat(*other);
                apply_unary_kernel::<$width, $mask, _, _>(self, |l| {
                    Simd::from(*l).simd_ne(r).to_bitmask() as $mask
                })
            }

            fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                let r = Simd::splat(*other);
                apply_unary_kernel::<$width, $mask, _, _>(self, |l| {
                    Simd::from(*l).simd_lt(r).to_bitmask() as $mask
                })
            }

            fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                let r = Simd::splat(*other);
                apply_unary_kernel::<$width, $mask, _, _>(self, |l| {
                    Simd::from(*l).simd_le(r).to_bitmask() as $mask
                })
            }

            fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                let r = Simd::splat(*other);
                apply_unary_kernel::<$width, $mask, _, _>(self, |l| {
                    Simd::from(*l).simd_gt(r).to_bitmask() as $mask
                })
            }

            fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                let r = Simd::splat(*other);
                apply_unary_kernel::<$width, $mask, _, _>(self, |l| {
                    Simd::from(*l).simd_ge(r).to_bitmask() as $mask
                })
            }
        }
    };
}

macro_rules! impl_float_total_ord_kernel {
    ($T: ty, $width: literal, $mask: ty) => {
        impl TotalOrdKernel for PrimitiveArray<$T> {
            type Scalar = $T;

            fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
                apply_binary_kernel::<$width, $mask, _, _>(self, other, |l, r| {
                    let ls = Simd::from(*l);
                    let rs = Simd::from(*r);
                    let lhs_is_nan = ls.simd_ne(ls);
                    let rhs_is_nan = rs.simd_ne(rs);
                    ((lhs_is_nan & rhs_is_nan) | ls.simd_eq(rs)).to_bitmask() as $mask
                })
            }

            fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
                apply_binary_kernel::<$width, $mask, _, _>(self, other, |l, r| {
                    let ls = Simd::from(*l);
                    let rs = Simd::from(*r);
                    let lhs_is_nan = ls.simd_ne(ls);
                    let rhs_is_nan = rs.simd_ne(rs);
                    (!((lhs_is_nan & rhs_is_nan) | ls.simd_eq(rs))).to_bitmask() as $mask
                })
            }

            fn tot_lt_kernel(&self, other: &Self) -> Bitmap {
                apply_binary_kernel::<$width, $mask, _, _>(self, other, |l, r| {
                    let ls = Simd::from(*l);
                    let rs = Simd::from(*r);
                    let lhs_is_nan = ls.simd_ne(ls);
                    (!(lhs_is_nan | ls.simd_ge(rs))).to_bitmask() as $mask
                })
            }

            fn tot_le_kernel(&self, other: &Self) -> Bitmap {
                apply_binary_kernel::<$width, $mask, _, _>(self, other, |l, r| {
                    let ls = Simd::from(*l);
                    let rs = Simd::from(*r);
                    let rhs_is_nan = rs.simd_ne(rs);
                    (rhs_is_nan | ls.simd_le(rs)).to_bitmask() as $mask
                })
            }

            fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                let rs = Simd::splat(*other);
                apply_unary_kernel::<$width, $mask, _, _>(self, |l| {
                    let ls = Simd::from(*l);
                    let lhs_is_nan = ls.simd_ne(ls);
                    let rhs_is_nan = rs.simd_ne(rs);
                    ((lhs_is_nan & rhs_is_nan) | ls.simd_eq(rs)).to_bitmask() as $mask
                })
            }

            fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                let rs = Simd::splat(*other);
                apply_unary_kernel::<$width, $mask, _, _>(self, |l| {
                    let ls = Simd::from(*l);
                    let lhs_is_nan = ls.simd_ne(ls);
                    let rhs_is_nan = rs.simd_ne(rs);
                    (!((lhs_is_nan & rhs_is_nan) | ls.simd_eq(rs))).to_bitmask() as $mask
                })
            }

            fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                let rs = Simd::splat(*other);
                apply_unary_kernel::<$width, $mask, _, _>(self, |l| {
                    let ls = Simd::from(*l);
                    let lhs_is_nan = ls.simd_ne(ls);
                    (!(lhs_is_nan | ls.simd_ge(rs))).to_bitmask() as $mask
                })
            }

            fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                let rs = Simd::splat(*other);
                apply_unary_kernel::<$width, $mask, _, _>(self, |l| {
                    let ls = Simd::from(*l);
                    let rhs_is_nan = rs.simd_ne(rs);
                    (rhs_is_nan | ls.simd_le(rs)).to_bitmask() as $mask
                })
            }

            fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                let rs = Simd::splat(*other);
                apply_unary_kernel::<$width, $mask, _, _>(self, |l| {
                    let ls = Simd::from(*l);
                    let rhs_is_nan = rs.simd_ne(rs);
                    (!(rhs_is_nan | rs.simd_ge(ls))).to_bitmask() as $mask
                })
            }

            fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
                let rs = Simd::splat(*other);
                apply_unary_kernel::<$width, $mask, _, _>(self, |l| {
                    let ls = Simd::from(*l);
                    let lhs_is_nan = ls.simd_ne(ls);
                    (lhs_is_nan | rs.simd_le(ls)).to_bitmask() as $mask
                })
            }
        }
    };
}

impl_int_total_ord_kernel!(u8, 32, u32);
impl_int_total_ord_kernel!(u16, 16, u16);
impl_int_total_ord_kernel!(u32, 8, u8);
impl_int_total_ord_kernel!(u64, 8, u8);
impl_int_total_ord_kernel!(i8, 32, u32);
impl_int_total_ord_kernel!(i16, 16, u16);
impl_int_total_ord_kernel!(i32, 8, u8);
impl_int_total_ord_kernel!(i64, 8, u8);
impl_float_total_ord_kernel!(f32, 8, u8);
impl_float_total_ord_kernel!(f64, 8, u8);
