use arrow::array::{Array, PrimitiveArray};
use arrow::types::NativeType;

// Low-level comparison kernel.
pub trait ArithmeticKernel: Sized + Array {
    type Scalar: ?Sized;
    type TrueDivT: NativeType;

    fn wrapping_neg(self) -> Self;
    fn wrapping_add(self, rhs: Self) -> Self;
    fn wrapping_sub(self, rhs: Self) -> Self;
    fn wrapping_mul(self, rhs: Self) -> Self;
    fn wrapping_floor_div(self, rhs: Self) -> Self;
    fn wrapping_mod(self, rhs: Self) -> Self;

    fn wrapping_add_scalar(self, rhs: Self::Scalar) -> Self;
    fn wrapping_sub_scalar(self, rhs: Self::Scalar) -> Self;
    fn wrapping_sub_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self;
    fn wrapping_mul_scalar(self, rhs: Self::Scalar) -> Self;
    fn wrapping_floor_div_scalar(self, rhs: Self::Scalar) -> Self;
    fn wrapping_floor_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self;
    fn wrapping_mod_scalar(self, rhs: Self::Scalar) -> Self;
    fn wrapping_mod_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self;

    fn true_div(self, rhs: Self) -> PrimitiveArray<Self::TrueDivT>;
    fn true_div_scalar(self, rhs: Self::Scalar) -> PrimitiveArray<Self::TrueDivT>;
    fn true_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> PrimitiveArray<Self::TrueDivT>;
}

// Proxy trait so one can bound T: HasPrimitiveArithmeticKernel. Sadly Rust
// doesn't support adding supertraits for other types.
#[allow(private_bounds)]
pub trait HasPrimitiveArithmeticKernel: NativeType + PrimitiveArithmeticKernelImpl {}
impl<T: NativeType + PrimitiveArithmeticKernelImpl> HasPrimitiveArithmeticKernel for T { }

use PrimitiveArray as PArr;

#[doc(hidden)]
pub trait PrimitiveArithmeticKernelImpl: NativeType {
    type TrueDivT: NativeType;

    fn prim_wrapping_neg(lhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_add(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_sub(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_mul(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_floor_div(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_mod(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self>;

    fn prim_wrapping_add_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self>;
    fn prim_wrapping_sub_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self>;
    fn prim_wrapping_sub_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_mul_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self>;
    fn prim_wrapping_floor_div_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self>;
    fn prim_wrapping_floor_div_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> PArr<Self>;
    fn prim_wrapping_mod_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self>;
    fn prim_wrapping_mod_scalar_lhs(lhs: Self, lhs: PArr<Self>) -> PArr<Self>;

    fn prim_true_div(lhs: PArr<Self>, rhs: PArr<Self>) -> PArr<Self::TrueDivT>;
    fn prim_true_div_scalar(lhs: PArr<Self>, rhs: Self) -> PArr<Self::TrueDivT>;
    fn prim_true_div_scalar_lhs(lhs: Self, rhs: PArr<Self>) -> PArr<Self::TrueDivT>;
}

impl<T: HasPrimitiveArithmeticKernel> ArithmeticKernel for PrimitiveArray<T> {
    type Scalar = T;
    type TrueDivT = T::TrueDivT;

    fn wrapping_neg(self) -> Self { T::prim_wrapping_neg(self) }
    fn wrapping_add(self, rhs: Self) -> Self { T::prim_wrapping_add(self, rhs) }
    fn wrapping_sub(self, rhs: Self) -> Self { T::prim_wrapping_sub(self, rhs) }
    fn wrapping_mul(self, rhs: Self) -> Self { T::prim_wrapping_mul(self, rhs) }
    fn wrapping_floor_div(self, rhs: Self) -> Self { T::prim_wrapping_floor_div(self, rhs) }
    fn wrapping_mod(self, rhs: Self) -> Self { T::prim_wrapping_mod(self, rhs) }

    fn wrapping_add_scalar(self, rhs: Self::Scalar) -> Self { T::prim_wrapping_add_scalar(self, rhs) }
    fn wrapping_sub_scalar(self, rhs: Self::Scalar) -> Self { T::prim_wrapping_sub_scalar(self, rhs) }
    fn wrapping_sub_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self { T::prim_wrapping_sub_scalar_lhs(lhs, rhs) }
    fn wrapping_mul_scalar(self, rhs: Self::Scalar) -> Self { T::prim_wrapping_mul_scalar(self, rhs) }
    fn wrapping_floor_div_scalar(self, rhs: Self::Scalar) -> Self { T::prim_wrapping_floor_div_scalar(self, rhs) }
    fn wrapping_floor_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self { T::prim_wrapping_floor_div_scalar_lhs(lhs, rhs) }
    fn wrapping_mod_scalar(self, rhs: Self::Scalar) -> Self { T::prim_wrapping_mod_scalar(self, rhs) }
    fn wrapping_mod_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self { T::prim_wrapping_mod_scalar_lhs(lhs, rhs) }

    fn true_div(self, rhs: Self) -> PrimitiveArray<Self::TrueDivT> { T::prim_true_div(self, rhs) }
    fn true_div_scalar(self, rhs: Self::Scalar) -> PrimitiveArray<Self::TrueDivT> { T::prim_true_div_scalar(self, rhs) }
    fn true_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> PrimitiveArray<Self::TrueDivT> { T::prim_true_div_scalar_lhs(lhs, rhs) }
}

mod float;
mod signed;
mod unsigned;
