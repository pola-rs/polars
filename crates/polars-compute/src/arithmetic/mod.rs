use std::any::TypeId;

use arrow::array::{Array, PrimitiveArray};
use arrow::types::NativeType;

// Low-level comparison kernel.
pub trait ArithmeticKernel: Sized + Array {
    type Scalar;
    type TrueDivT: NativeType;

    fn wrapping_abs(self) -> Self;
    fn wrapping_neg(self) -> Self;
    fn wrapping_add(self, rhs: Self) -> Self;
    fn wrapping_sub(self, rhs: Self) -> Self;
    fn wrapping_mul(self, rhs: Self) -> Self;
    fn wrapping_floor_div(self, rhs: Self) -> Self;
    fn wrapping_trunc_div(self, rhs: Self) -> Self;
    fn wrapping_mod(self, rhs: Self) -> Self;

    fn wrapping_add_scalar(self, rhs: Self::Scalar) -> Self;
    fn wrapping_sub_scalar(self, rhs: Self::Scalar) -> Self;
    fn wrapping_sub_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self;
    fn wrapping_mul_scalar(self, rhs: Self::Scalar) -> Self;
    fn wrapping_floor_div_scalar(self, rhs: Self::Scalar) -> Self;
    fn wrapping_floor_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self;
    fn wrapping_trunc_div_scalar(self, rhs: Self::Scalar) -> Self;
    fn wrapping_trunc_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self;
    fn wrapping_mod_scalar(self, rhs: Self::Scalar) -> Self;
    fn wrapping_mod_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self;

    fn true_div(self, rhs: Self) -> PrimitiveArray<Self::TrueDivT>;
    fn true_div_scalar(self, rhs: Self::Scalar) -> PrimitiveArray<Self::TrueDivT>;
    fn true_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> PrimitiveArray<Self::TrueDivT>;

    // TODO: remove these.
    // These are flooring division for integer types, true division for floating point types.
    fn legacy_div(self, rhs: Self) -> Self {
        if TypeId::of::<Self>() == TypeId::of::<PrimitiveArray<Self::TrueDivT>>() {
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
    fn legacy_div_scalar(self, rhs: Self::Scalar) -> Self {
        if TypeId::of::<Self>() == TypeId::of::<PrimitiveArray<Self::TrueDivT>>() {
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

    fn legacy_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self {
        if TypeId::of::<Self>() == TypeId::of::<PrimitiveArray<Self::TrueDivT>>() {
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

use PrimitiveArray as PArr;

#[doc(hidden)]
pub trait PrimitiveArithmeticKernelImpl: NativeType {
    type TrueDivT: NativeType;

    fn prim_wrapping_abs(lhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_neg(lhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_add(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_sub(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_mul(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_floor_div(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_trunc_div(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_mod(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self>;

    fn prim_wrapping_add_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self>;
    fn prim_wrapping_sub_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self>;
    fn prim_wrapping_sub_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_mul_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self>;
    fn prim_wrapping_floor_div_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self>;
    fn prim_wrapping_floor_div_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_trunc_div_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self>;
    fn prim_wrapping_trunc_div_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_mod_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self>;
    fn prim_wrapping_mod_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> PArr<Self>;

    fn prim_true_div(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self::TrueDivT>;
    fn prim_true_div_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self::TrueDivT>;
    fn prim_true_div_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> PArr<Self::TrueDivT>;
}

#[rustfmt::skip]
impl<T: HasPrimitiveArithmeticKernel> ArithmeticKernel for PrimitiveArray<T> {
    type Scalar = T;
    type TrueDivT = T::TrueDivT;

    fn wrapping_abs(self) -> Self { T::prim_wrapping_abs(self) }
    fn wrapping_neg(self) -> Self { T::prim_wrapping_neg(self) }
    fn wrapping_add(self, rhs: Self) -> Self { T::prim_wrapping_add(self, rhs) }
    fn wrapping_sub(self, rhs: Self) -> Self { T::prim_wrapping_sub(self, rhs) }
    fn wrapping_mul(self, rhs: Self) -> Self { T::prim_wrapping_mul(self, rhs) }
    fn wrapping_floor_div(self, rhs: Self) -> Self { T::prim_wrapping_floor_div(self, rhs) }
    fn wrapping_trunc_div(self, rhs: Self) -> Self { T::prim_wrapping_trunc_div(self, rhs) }
    fn wrapping_mod(self, rhs: Self) -> Self { T::prim_wrapping_mod(self, rhs) }

    fn wrapping_add_scalar(self, rhs: Self::Scalar) -> Self { T::prim_wrapping_add_scalar(self, rhs) }
    fn wrapping_sub_scalar(self, rhs: Self::Scalar) -> Self { T::prim_wrapping_sub_scalar(self, rhs) }
    fn wrapping_sub_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self { T::prim_wrapping_sub_scalar_lhs(lhs, rhs) }
    fn wrapping_mul_scalar(self, rhs: Self::Scalar) -> Self { T::prim_wrapping_mul_scalar(self, rhs) }
    fn wrapping_floor_div_scalar(self, rhs: Self::Scalar) -> Self { T::prim_wrapping_floor_div_scalar(self, rhs) }
    fn wrapping_floor_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self { T::prim_wrapping_floor_div_scalar_lhs(lhs, rhs) }
    fn wrapping_trunc_div_scalar(self, rhs: Self::Scalar) -> Self { T::prim_wrapping_trunc_div_scalar(self, rhs) }
    fn wrapping_trunc_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self { T::prim_wrapping_trunc_div_scalar_lhs(lhs, rhs) }
    fn wrapping_mod_scalar(self, rhs: Self::Scalar) -> Self { T::prim_wrapping_mod_scalar(self, rhs) }
    fn wrapping_mod_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self { T::prim_wrapping_mod_scalar_lhs(lhs, rhs) }

    fn true_div(self, rhs: Self) -> PrimitiveArray<Self::TrueDivT> { T::prim_true_div(self, rhs) }
    fn true_div_scalar(self, rhs: Self::Scalar) -> PrimitiveArray<Self::TrueDivT> { T::prim_true_div_scalar(self, rhs) }
    fn true_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> PrimitiveArray<Self::TrueDivT> { T::prim_true_div_scalar_lhs(lhs, rhs) }
}

mod float;
mod signed;
mod unsigned;
