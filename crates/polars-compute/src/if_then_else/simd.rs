#[cfg(target_arch = "x86_64")]
use std::mem::MaybeUninit;
#[cfg(target_arch = "x86_64")]
use std::simd::{Mask, Simd, SimdElement};

use arrow::array::PrimitiveArray;
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;

use super::{
    if_then_else_loop, if_then_else_loop_broadcast_both, if_then_else_loop_broadcast_false,
    if_then_else_validity, scalar, IfThenElseKernel,
};

#[cfg(target_arch = "x86_64")]
fn select_simd_64<T: Copy + SimdElement>(
    mask: u64,
    if_true: Simd<T, 64>,
    if_false: Simd<T, 64>,
    out: &mut [MaybeUninit<T>; 64],
) {
    let mv = Mask::<<T as SimdElement>::Mask, 64>::from_bitmask(mask);
    let ret = mv.select(if_true, if_false);
    unsafe {
        let src = ret.as_array().as_ptr() as *const MaybeUninit<T>;
        core::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), 64);
    }
}

#[cfg(target_arch = "x86_64")]
fn if_then_else_simd_64<T: Copy + SimdElement>(
    mask: u64,
    if_true: &[T; 64],
    if_false: &[T; 64],
    out: &mut [MaybeUninit<T>; 64],
) {
    select_simd_64(
        mask,
        Simd::from_slice(if_true),
        Simd::from_slice(if_false),
        out,
    )
}

#[cfg(target_arch = "x86_64")]
fn if_then_else_broadcast_false_simd_64<T: Copy + SimdElement>(
    mask: u64,
    if_true: &[T; 64],
    if_false: T,
    out: &mut [MaybeUninit<T>; 64],
) {
    select_simd_64(mask, Simd::from_slice(if_true), Simd::splat(if_false), out)
}

#[cfg(target_arch = "x86_64")]
fn if_then_else_broadcast_both_simd_64<T: Copy + SimdElement>(
    mask: u64,
    if_true: T,
    if_false: T,
    out: &mut [MaybeUninit<T>; 64],
) {
    select_simd_64(mask, Simd::splat(if_true), Simd::splat(if_false), out)
}

macro_rules! impl_if_then_else {
    ($T: ty) => {
        impl IfThenElseKernel for PrimitiveArray<$T> {
            type Scalar<'a> = $T;

            fn if_then_else(mask: &Bitmap, if_true: &Self, if_false: &Self) -> Self {
                let values = if_then_else_loop(
                    mask,
                    if_true.values(),
                    if_false.values(),
                    scalar::if_then_else_scalar_rest,
                    // Auto-generated SIMD was slower on ARM.
                    #[cfg(target_arch = "x86_64")]
                    if_then_else_simd_64,
                    #[cfg(not(target_arch = "x86_64"))]
                    scalar::if_then_else_scalar_64,
                );
                let validity = if_then_else_validity(mask, if_true.validity(), if_false.validity());
                PrimitiveArray::from_vec(values).with_validity(validity)
            }

            fn if_then_else_broadcast_true(
                mask: &Bitmap,
                if_true: Self::Scalar<'_>,
                if_false: &Self,
            ) -> Self {
                let values = if_then_else_loop_broadcast_false(
                    true,
                    mask,
                    if_false.values(),
                    if_true,
                    // Auto-generated SIMD was slower on ARM.
                    #[cfg(target_arch = "x86_64")]
                    if_then_else_broadcast_false_simd_64,
                    #[cfg(not(target_arch = "x86_64"))]
                    scalar::if_then_else_broadcast_false_scalar_64,
                );
                let validity = if_then_else_validity(mask, None, if_false.validity());
                PrimitiveArray::from_vec(values).with_validity(validity)
            }

            fn if_then_else_broadcast_false(
                mask: &Bitmap,
                if_true: &Self,
                if_false: Self::Scalar<'_>,
            ) -> Self {
                let values = if_then_else_loop_broadcast_false(
                    false,
                    mask,
                    if_true.values(),
                    if_false,
                    // Auto-generated SIMD was slower on ARM.
                    #[cfg(target_arch = "x86_64")]
                    if_then_else_broadcast_false_simd_64,
                    #[cfg(not(target_arch = "x86_64"))]
                    scalar::if_then_else_broadcast_false_scalar_64,
                );
                let validity = if_then_else_validity(mask, if_true.validity(), None);
                PrimitiveArray::from_vec(values).with_validity(validity)
            }

            fn if_then_else_broadcast_both(
                _dtype: ArrowDataType,
                mask: &Bitmap,
                if_true: Self::Scalar<'_>,
                if_false: Self::Scalar<'_>,
            ) -> Self {
                let values = if_then_else_loop_broadcast_both(
                    mask,
                    if_true,
                    if_false,
                    // Auto-generated SIMD was slower on ARM.
                    #[cfg(target_arch = "x86_64")]
                    if_then_else_broadcast_both_simd_64,
                    #[cfg(not(target_arch = "x86_64"))]
                    scalar::if_then_else_broadcast_both_scalar_64,
                );
                PrimitiveArray::from_vec(values)
            }
        }
    };
}

impl_if_then_else!(i8);
impl_if_then_else!(i16);
impl_if_then_else!(i32);
impl_if_then_else!(i64);
impl_if_then_else!(u8);
impl_if_then_else!(u16);
impl_if_then_else!(u32);
impl_if_then_else!(u64);
impl_if_then_else!(f32);
impl_if_then_else!(f64);
