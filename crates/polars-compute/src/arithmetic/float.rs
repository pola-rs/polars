use arrow::array::PrimitiveArray as PArr;

use super::PrimitiveArithmeticKernelImpl;
use crate::arity::{prim_binary_values, prim_unary_values};

macro_rules! impl_float_arith_kernel {
    ($T:ty) => {
        impl PrimitiveArithmeticKernelImpl for $T {
            type TrueDivT = $T;

            fn prim_wrapping_abs(lhs: PArr<$T>) -> PArr<$T> {
                prim_unary_values(lhs, |x| x.abs())
            }

            fn prim_wrapping_neg(lhs: PArr<$T>) -> PArr<$T> {
                prim_unary_values(lhs, |x| -x)
            }

            fn prim_wrapping_add(lhs: PArr<$T>, rhs: PArr<$T>) -> PArr<$T> {
                prim_binary_values(lhs, rhs, |l, r| l + r)
            }

            fn prim_wrapping_sub(lhs: PArr<$T>, rhs: PArr<$T>) -> PArr<$T> {
                prim_binary_values(lhs, rhs, |l, r| l - r)
            }

            fn prim_wrapping_mul(lhs: PArr<$T>, rhs: PArr<$T>) -> PArr<$T> {
                prim_binary_values(lhs, rhs, |l, r| l * r)
            }

            fn prim_wrapping_floor_div(lhs: PArr<$T>, rhs: PArr<$T>) -> PArr<$T> {
                prim_binary_values(lhs, rhs, |l, r| (l / r).floor())
            }

            fn prim_wrapping_trunc_div(lhs: PArr<$T>, rhs: PArr<$T>) -> PArr<$T> {
                prim_binary_values(lhs, rhs, |l, r| (l / r).trunc())
            }

            fn prim_wrapping_mod(lhs: PArr<$T>, rhs: PArr<$T>) -> PArr<$T> {
                prim_binary_values(lhs, rhs, |l, r| l - r * (l / r).floor())
            }

            fn prim_wrapping_add_scalar(lhs: PArr<$T>, rhs: $T) -> PArr<$T> {
                if rhs == 0.0 {
                    return lhs;
                }
                prim_unary_values(lhs, |x| x + rhs)
            }

            fn prim_wrapping_sub_scalar(lhs: PArr<$T>, rhs: $T) -> PArr<$T> {
                if rhs == 0.0 {
                    return lhs;
                }
                Self::prim_wrapping_add_scalar(lhs, -rhs)
            }

            fn prim_wrapping_sub_scalar_lhs(lhs: $T, rhs: PArr<$T>) -> PArr<$T> {
                if lhs == 0.0 {
                    Self::prim_wrapping_neg(rhs)
                } else {
                    prim_unary_values(rhs, |x| lhs - x)
                }
            }

            fn prim_wrapping_mul_scalar(lhs: PArr<$T>, rhs: $T) -> PArr<$T> {
                // No optimization for multiplication by zero, would invalidate NaNs/infinities.
                if rhs == 1.0 {
                    lhs
                } else if rhs == -1.0 {
                    Self::prim_wrapping_neg(lhs)
                } else {
                    prim_unary_values(lhs, |x| x * rhs)
                }
            }

            fn prim_wrapping_floor_div_scalar(lhs: PArr<$T>, rhs: $T) -> PArr<$T> {
                let inv = 1.0 / rhs;
                prim_unary_values(lhs, |x| (x * inv).floor())
            }

            fn prim_wrapping_floor_div_scalar_lhs(lhs: $T, rhs: PArr<$T>) -> PArr<$T> {
                prim_unary_values(rhs, |x| (lhs / x).floor())
            }

            fn prim_wrapping_trunc_div_scalar(lhs: PArr<$T>, rhs: $T) -> PArr<$T> {
                let inv = 1.0 / rhs;
                prim_unary_values(lhs, |x| (x * inv).trunc())
            }

            fn prim_wrapping_trunc_div_scalar_lhs(lhs: $T, rhs: PArr<$T>) -> PArr<$T> {
                prim_unary_values(rhs, |x| (lhs / x).trunc())
            }

            fn prim_wrapping_mod_scalar(lhs: PArr<$T>, rhs: $T) -> PArr<$T> {
                let inv = 1.0 / rhs;
                prim_unary_values(lhs, |x| x - rhs * (x * inv).floor())
            }

            fn prim_wrapping_mod_scalar_lhs(lhs: $T, rhs: PArr<$T>) -> PArr<$T> {
                prim_unary_values(rhs, |x| lhs - x * (lhs / x).floor())
            }

            fn prim_true_div(lhs: PArr<$T>, rhs: PArr<$T>) -> PArr<Self::TrueDivT> {
                prim_binary_values(lhs, rhs, |l, r| l / r)
            }

            fn prim_true_div_scalar(lhs: PArr<$T>, rhs: $T) -> PArr<Self::TrueDivT> {
                Self::prim_wrapping_mul_scalar(lhs, 1.0 / rhs)
            }

            fn prim_true_div_scalar_lhs(lhs: $T, rhs: PArr<$T>) -> PArr<Self::TrueDivT> {
                prim_unary_values(rhs, |x| lhs / x)
            }
        }
    };
}

impl_float_arith_kernel!(f32);
impl_float_arith_kernel!(f64);
