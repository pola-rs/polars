use std::any::TypeId;

use arrow::bitmap::BitmapBuilder;
use arrow::types::NativeType;
use polars_array::{Flat, PlBitmap, PlPrimitiveArray};

use self::pl_array::{binary, unary};

/// The array the inner kernels read: flat, so its every buffer holds one slot per element.
///
/// [`ArithmeticKernel`] itself takes a chunk in whatever representation it is in and hands the
/// inner kernel only what it has to read; see [`pl_array`].
pub(crate) type PArr<T> = Flat<PlPrimitiveArray<T>>;
/// The array a kernel writes, which is flat unless it is a single value repeated.
pub(crate) type POut<T> = PlPrimitiveArray<T>;

// Low-level comparison kernel.
pub trait ArithmeticKernel: Sized {
    /// The element type of the array this operates on, which a broadcasting kernel takes one of.
    type Scalar: NativeType;
    type TrueDivT: NativeType;

    fn wrapping_abs(self) -> POut<Self::Scalar>;
    fn wrapping_neg(self) -> POut<Self::Scalar>;
    fn wrapping_add(self, rhs: Self) -> POut<Self::Scalar>;
    fn wrapping_sub(self, rhs: Self) -> POut<Self::Scalar>;
    fn wrapping_mul(self, rhs: Self) -> POut<Self::Scalar>;
    fn wrapping_floor_div(self, rhs: Self) -> POut<Self::Scalar>;
    fn wrapping_trunc_div(self, rhs: Self) -> POut<Self::Scalar>;
    fn wrapping_mod(self, rhs: Self) -> POut<Self::Scalar>;

    fn wrapping_add_scalar(self, rhs: Self::Scalar) -> POut<Self::Scalar>;
    fn wrapping_sub_scalar(self, rhs: Self::Scalar) -> POut<Self::Scalar>;
    fn wrapping_sub_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> POut<Self::Scalar>;
    fn wrapping_mul_scalar(self, rhs: Self::Scalar) -> POut<Self::Scalar>;
    fn wrapping_floor_div_scalar(self, rhs: Self::Scalar) -> POut<Self::Scalar>;
    fn wrapping_floor_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> POut<Self::Scalar>;
    fn wrapping_trunc_div_scalar(self, rhs: Self::Scalar) -> POut<Self::Scalar>;
    fn wrapping_trunc_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> POut<Self::Scalar>;
    fn wrapping_mod_scalar(self, rhs: Self::Scalar) -> POut<Self::Scalar>;
    fn wrapping_mod_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> POut<Self::Scalar>;

    fn checked_mul_scalar(self, rhs: Self::Scalar) -> POut<Self::Scalar>;

    fn true_div(self, rhs: Self) -> POut<Self::TrueDivT>;
    fn true_div_scalar(self, rhs: Self::Scalar) -> POut<Self::TrueDivT>;
    fn true_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> POut<Self::TrueDivT>;

    // TODO: remove these.
    // These are flooring division for integer types, true division for floating point types.
    fn legacy_div(self, rhs: Self) -> POut<Self::Scalar> {
        if TypeId::of::<Self::Scalar>() == TypeId::of::<Self::TrueDivT>() {
            let ret = self.true_div(rhs);
            unsafe {
                let cast_ret = std::mem::transmute_copy(&ret);
                std::mem::forget(ret);
                cast_ret
            }
        } else {
            self.wrapping_floor_div(rhs)
        }
    }
    fn legacy_div_scalar(self, rhs: Self::Scalar) -> POut<Self::Scalar> {
        if TypeId::of::<Self::Scalar>() == TypeId::of::<Self::TrueDivT>() {
            let ret = self.true_div_scalar(rhs);
            unsafe {
                let cast_ret = std::mem::transmute_copy(&ret);
                std::mem::forget(ret);
                cast_ret
            }
        } else {
            self.wrapping_floor_div_scalar(rhs)
        }
    }

    fn legacy_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> POut<Self::Scalar> {
        if TypeId::of::<Self::Scalar>() == TypeId::of::<Self::TrueDivT>() {
            let ret = ArithmeticKernel::true_div_scalar_lhs(lhs, rhs);
            unsafe {
                let cast_ret = std::mem::transmute_copy(&ret);
                std::mem::forget(ret);
                cast_ret
            }
        } else {
            ArithmeticKernel::wrapping_floor_div_scalar_lhs(lhs, rhs)
        }
    }
}

// Proxy trait so one can bound T: HasPrimitiveArithmeticKernel. Sadly Rust
// doesn't support adding supertraits for other types.
#[allow(private_bounds)]
pub trait HasPrimitiveArithmeticKernel: NativeType + PrimitiveArithmeticKernelImpl {}
impl<T: NativeType + PrimitiveArithmeticKernelImpl> HasPrimitiveArithmeticKernel for T {}

use num_traits::{CheckedMul, WrappingMul};
use polars_utils::vec::PushUnchecked;

#[doc(hidden)]
pub trait PrimitiveArithmeticKernelImpl: NativeType {
    type TrueDivT: NativeType;

    fn prim_wrapping_abs(lhs: PArr<Self>) -> POut<Self>;
    fn prim_wrapping_neg(lhs: PArr<Self>) -> POut<Self>;
    fn prim_wrapping_add(lhs: PArr<Self>, rhs: PArr<Self>) -> POut<Self>;
    fn prim_wrapping_sub(lhs: PArr<Self>, rhs: PArr<Self>) -> POut<Self>;
    fn prim_wrapping_mul(lhs: PArr<Self>, rhs: PArr<Self>) -> POut<Self>;
    fn prim_wrapping_floor_div(lhs: PArr<Self>, rhs: PArr<Self>) -> POut<Self>;
    fn prim_wrapping_trunc_div(lhs: PArr<Self>, rhs: PArr<Self>) -> POut<Self>;
    fn prim_wrapping_mod(lhs: PArr<Self>, rhs: PArr<Self>) -> POut<Self>;

    fn prim_wrapping_add_scalar(lhs: PArr<Self>, rhs: Self) -> POut<Self>;
    fn prim_wrapping_sub_scalar(lhs: PArr<Self>, rhs: Self) -> POut<Self>;
    fn prim_wrapping_sub_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> POut<Self>;
    fn prim_wrapping_mul_scalar(lhs: PArr<Self>, rhs: Self) -> POut<Self>;
    fn prim_wrapping_floor_div_scalar(lhs: PArr<Self>, rhs: Self) -> POut<Self>;
    fn prim_wrapping_floor_div_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> POut<Self>;
    fn prim_wrapping_trunc_div_scalar(lhs: PArr<Self>, rhs: Self) -> POut<Self>;
    fn prim_wrapping_trunc_div_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> POut<Self>;
    fn prim_wrapping_mod_scalar(lhs: PArr<Self>, rhs: Self) -> POut<Self>;
    fn prim_wrapping_mod_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> POut<Self>;

    fn prim_checked_mul_scalar(lhs: PArr<Self>, rhs: Self) -> POut<Self>;

    fn prim_true_div(lhs: PArr<Self>, rhs: PArr<Self>) -> POut<Self::TrueDivT>;
    fn prim_true_div_scalar(lhs: PArr<Self>, rhs: Self) -> POut<Self::TrueDivT>;
    fn prim_true_div_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> POut<Self::TrueDivT>;
}

/// The kernels of [`PrimitiveArithmeticKernelImpl`], each behind the dispatch that hands it only
/// the part of a chunk it has to read. A chunk that repeats a single value is never written out
/// one slot per element to get to one; see [`pl_array`].
#[rustfmt::skip]
impl<T: HasPrimitiveArithmeticKernel> ArithmeticKernel for PlPrimitiveArray<T> {
    type Scalar = T;
    type TrueDivT = T::TrueDivT;

    fn wrapping_abs(self) -> POut<T> { unary(self, T::prim_wrapping_abs) }
    fn wrapping_neg(self) -> POut<T> { unary(self, T::prim_wrapping_neg) }

    // Addition and multiplication are the two that commute, so a repeated left operand reaches
    // the same kernel a repeated right one does, with the sides swapped.
    fn wrapping_add(self, rhs: Self) -> POut<T> { binary(self, rhs, T::prim_wrapping_add, |l, r| T::prim_wrapping_add_scalar(r, l), T::prim_wrapping_add_scalar) }
    fn wrapping_sub(self, rhs: Self) -> POut<T> { binary(self, rhs, T::prim_wrapping_sub, T::prim_wrapping_sub_scalar_lhs, T::prim_wrapping_sub_scalar) }
    fn wrapping_mul(self, rhs: Self) -> POut<T> { binary(self, rhs, T::prim_wrapping_mul, |l, r| T::prim_wrapping_mul_scalar(r, l), T::prim_wrapping_mul_scalar) }
    fn wrapping_floor_div(self, rhs: Self) -> POut<T> { binary(self, rhs, T::prim_wrapping_floor_div, T::prim_wrapping_floor_div_scalar_lhs, T::prim_wrapping_floor_div_scalar) }
    fn wrapping_trunc_div(self, rhs: Self) -> POut<T> { binary(self, rhs, T::prim_wrapping_trunc_div, T::prim_wrapping_trunc_div_scalar_lhs, T::prim_wrapping_trunc_div_scalar) }
    fn wrapping_mod(self, rhs: Self) -> POut<T> { binary(self, rhs, T::prim_wrapping_mod, T::prim_wrapping_mod_scalar_lhs, T::prim_wrapping_mod_scalar) }

    fn wrapping_add_scalar(self, rhs: Self::Scalar) -> POut<T> { unary(self, |lhs| T::prim_wrapping_add_scalar(lhs, rhs)) }
    fn wrapping_sub_scalar(self, rhs: Self::Scalar) -> POut<T> { unary(self, |lhs| T::prim_wrapping_sub_scalar(lhs, rhs)) }
    fn wrapping_sub_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> POut<T> { unary(rhs, |rhs| T::prim_wrapping_sub_scalar_lhs(lhs, rhs)) }
    fn wrapping_mul_scalar(self, rhs: Self::Scalar) -> POut<T> { unary(self, |lhs| T::prim_wrapping_mul_scalar(lhs, rhs)) }
    fn wrapping_floor_div_scalar(self, rhs: Self::Scalar) -> POut<T> { unary(self, |lhs| T::prim_wrapping_floor_div_scalar(lhs, rhs)) }
    fn wrapping_floor_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> POut<T> { unary(rhs, |rhs| T::prim_wrapping_floor_div_scalar_lhs(lhs, rhs)) }
    fn wrapping_trunc_div_scalar(self, rhs: Self::Scalar) -> POut<T> { unary(self, |lhs| T::prim_wrapping_trunc_div_scalar(lhs, rhs)) }
    fn wrapping_trunc_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> POut<T> { unary(rhs, |rhs| T::prim_wrapping_trunc_div_scalar_lhs(lhs, rhs)) }
    fn wrapping_mod_scalar(self, rhs: Self::Scalar) -> POut<T> { unary(self, |lhs| T::prim_wrapping_mod_scalar(lhs, rhs)) }
    fn wrapping_mod_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> POut<T> { unary(rhs, |rhs| T::prim_wrapping_mod_scalar_lhs(lhs, rhs)) }

    fn checked_mul_scalar(self, rhs: Self::Scalar) -> POut<T> { unary(self, |lhs| T::prim_checked_mul_scalar(lhs, rhs)) }

    fn true_div(self, rhs: Self) -> POut<Self::TrueDivT> { binary(self, rhs, T::prim_true_div, T::prim_true_div_scalar_lhs, T::prim_true_div_scalar) }
    fn true_div_scalar(self, rhs: Self::Scalar) -> POut<Self::TrueDivT> { unary(self, |lhs| T::prim_true_div_scalar(lhs, rhs)) }
    fn true_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> POut<Self::TrueDivT> { unary(rhs, |rhs| T::prim_true_div_scalar_lhs(lhs, rhs)) }
}

mod float;
mod pl_array;
pub mod pl_num;
mod signed;
mod unsigned;

fn prim_checked_mul_scalar<I: NativeType + CheckedMul + WrappingMul>(
    array: &PArr<I>,
    factor: I,
) -> POut<I> {
    let values = array.values();
    let mut out = Vec::with_capacity(array.len());
    let mut i = 0;

    while i < array.len() && values[i].checked_mul(&factor).is_some() {
        // SAFETY: We allocated enough before.
        unsafe { out.push_unchecked(values[i].wrapping_mul(&factor)) };
        i += 1;
    }

    if out.len() == array.len() {
        return POut::<I>::new(
            out.into(),
            array.len(),
            array.validity().cloned().map(PlBitmap::from_bitmap),
        );
    }

    let mut validity = BitmapBuilder::with_capacity(array.len());
    validity.extend_constant(out.len(), true);

    for &value in &values[out.len()..] {
        // SAFETY: We allocated enough before.
        unsafe {
            out.push_unchecked(value.wrapping_mul(&factor));
            validity.push_unchecked(value.checked_mul(&factor).is_some());
        }
    }

    debug_assert_eq!(out.len(), array.len());
    debug_assert_eq!(validity.len(), array.len());

    let validity = validity.freeze();
    let validity = match array.validity() {
        None => validity,
        Some(arr_validity) => arrow::bitmap::and(&validity, arr_validity),
    };

    POut::<I>::new(
        out.into(),
        array.len(),
        Some(PlBitmap::from_bitmap(validity)),
    )
}
