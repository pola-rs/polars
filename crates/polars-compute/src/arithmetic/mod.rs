use arrow::array::{Array, PrimitiveArray};
use arrow::types::NativeType;

// Low-level comparison kernel.
pub trait ArithmeticKernel: Sized + Array {
    type Scalar: ?Sized;
    type TrueDivT: NativeType;

    fn wrapping_neg(self) -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
    fn wrapping_floor_div(self, other: Self) -> Self;
    fn wrapping_mod(self, other: Self) -> Self;

    fn wrapping_add_scalar(self, scalar: Self::Scalar) -> Self;
    fn wrapping_sub_scalar(self, scalar: Self::Scalar) -> Self;
    fn wrapping_sub_scalar_lhs(self, scalar: Self::Scalar) -> Self;
    fn wrapping_mul_scalar(self, scalar: Self::Scalar) -> Self;
    fn wrapping_floor_div_scalar(self, scalar: Self::Scalar) -> Self;
    fn wrapping_floor_div_scalar_lhs(self, scalar: Self::Scalar) -> Self;
    fn wrapping_mod_scalar(self, scalar: Self::Scalar) -> Self;
    fn wrapping_mod_scalar_lhs(self, scalar: Self::Scalar) -> Self;

    fn true_div(self, other: Self) -> PrimitiveArray<Self::TrueDivT>;
    fn true_div_scalar(self, scalar: Self::Scalar) -> PrimitiveArray<Self::TrueDivT>;
    fn true_div_scalar_lhs(self, scalar: Self::Scalar) -> PrimitiveArray<Self::TrueDivT>;
}

mod float;
mod signed;
mod unsigned;
