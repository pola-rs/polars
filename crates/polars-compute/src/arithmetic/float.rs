use arrow::array::PrimitiveArray;

use super::ArithmeticKernel;
use crate::arity::{prim_binary_values, prim_unary_values};

macro_rules! impl_float_arith_kernel {
    ($T:ty) => {
        impl ArithmeticKernel for PrimitiveArray<$T> {
            type Scalar = $T;
            type TrueDivT = $T;

            fn wrapping_neg(self) -> Self {
                prim_unary_values(self, |x| -x)
            }

            fn wrapping_add(self, other: Self) -> Self {
                prim_binary_values(self, other, |l, r| l + r)
            }

            fn wrapping_sub(self, other: Self) -> Self {
                prim_binary_values(self, other, |l, r| l - r)
            }

            fn wrapping_mul(self, other: Self) -> Self {
                prim_binary_values(self, other, |l, r| l * r)
            }

            fn wrapping_floor_div(self, other: Self) -> Self {
                prim_binary_values(self, other, |l, r| (l / r).floor())
            }

            fn wrapping_mod(self, other: Self) -> Self {
                prim_binary_values(self, other, |l, r| l - r * (l / r).floor())
            }

            fn wrapping_add_scalar(self, scalar: Self::Scalar) -> Self {
                if scalar == 0.0 {
                    return self;
                }
                prim_unary_values(self, |x| x + scalar)
            }

            fn wrapping_sub_scalar(self, scalar: Self::Scalar) -> Self {
                if scalar == 0.0 {
                    return self;
                }
                self.wrapping_add_scalar(-scalar)
            }

            fn wrapping_sub_scalar_lhs(self, scalar: Self::Scalar) -> Self {
                if scalar == 0.0 {
                    self.wrapping_neg()
                } else {
                    prim_unary_values(self, |x| scalar - x)
                }
            }

            fn wrapping_mul_scalar(self, scalar: Self::Scalar) -> Self {
                // No optimization for multiplication by zero, would invalidate NaNs/infinities.
                if scalar == 1.0 {
                    self
                } else if scalar == -1.0 {
                    self.wrapping_neg()
                } else {
                    prim_unary_values(self, |x| x * scalar)
                }
            }

            fn wrapping_floor_div_scalar(self, scalar: Self::Scalar) -> Self {
                let inv = 1.0 / scalar;
                prim_unary_values(self, |x| (x * inv).floor())
            }

            fn wrapping_floor_div_scalar_lhs(self, scalar: Self::Scalar) -> Self {
                prim_unary_values(self, |x| (scalar / x).floor())
            }

            fn wrapping_mod_scalar(self, scalar: Self::Scalar) -> Self {
                let inv = 1.0 / scalar;
                prim_unary_values(self, |x| x - scalar * (x * inv).floor())
            }

            fn wrapping_mod_scalar_lhs(self, scalar: Self::Scalar) -> Self {
                prim_unary_values(self, |x| scalar - x * (scalar / x).floor())
            }

            fn true_div(self, other: Self) -> PrimitiveArray<Self::TrueDivT> {
                prim_binary_values(self, other, |l, r| l / r)
            }

            fn true_div_scalar(self, scalar: Self::Scalar) -> PrimitiveArray<Self::TrueDivT> {
                self.wrapping_mul_scalar(1.0 / scalar)
            }

            fn true_div_scalar_lhs(self, scalar: Self::Scalar) -> PrimitiveArray<Self::TrueDivT> {
                prim_unary_values(self, |x| scalar / x)
            }
        }
    };
}

impl_float_arith_kernel!(f32);
impl_float_arith_kernel!(f64);
