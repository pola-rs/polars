use arrow::array::PrimitiveArray as PArr;

use super::PrimitiveArithmeticKernelImpl;
use crate::arity::{prim_binary_values, prim_unary_values, StoreIntrinsic};

macro_rules! impl_float_arith_kernel {
    ($T:ty) => {
        impl PrimitiveArithmeticKernelImpl for $T {
            type TrueDivT = $T;

            fn prim_wrapping_abs<S: StoreIntrinsic>(lhs: PArr<$T>, s: S) -> PArr<$T> {
                prim_unary_values(lhs, |x| x.abs(), s)
            }

            fn prim_wrapping_neg<S: StoreIntrinsic>(lhs: PArr<$T>, s: S) -> PArr<$T> {
                prim_unary_values(lhs, |x| -x, s)
            }

            fn prim_wrapping_add<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_binary_values(lhs, rhs, |l, r| l + r, s)
            }

            fn prim_wrapping_sub<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_binary_values(lhs, rhs, |l, r| l - r, s)
            }

            fn prim_wrapping_mul<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_binary_values(lhs, rhs, |l, r| l * r, s)
            }

            fn prim_wrapping_floor_div<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_binary_values(lhs, rhs, |l, r| (l / r).floor(), s)
            }

            fn prim_wrapping_trunc_div<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_binary_values(lhs, rhs, |l, r| (l / r).trunc(), s)
            }

            fn prim_wrapping_mod<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_binary_values(lhs, rhs, |l, r| l - r * (l / r).floor(), s)
            }

            fn prim_wrapping_add_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<$T> {
                if rhs == 0.0 {
                    return lhs;
                }
                prim_unary_values(lhs, |x| x + rhs, s)
            }

            fn prim_wrapping_sub_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<$T> {
                if rhs == 0.0 {
                    return lhs;
                }
                Self::prim_wrapping_add_scalar(lhs, -rhs, s)
            }

            fn prim_wrapping_sub_scalar_lhs<S: StoreIntrinsic>(
                lhs: $T,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                if lhs == 0.0 {
                    Self::prim_wrapping_neg(rhs, s)
                } else {
                    prim_unary_values(rhs, |x| lhs - x, s)
                }
            }

            fn prim_wrapping_mul_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<$T> {
                // No optimization for multiplication by zero, would invalidate NaNs/infinities.
                if rhs == 1.0 {
                    lhs
                } else if rhs == -1.0 {
                    Self::prim_wrapping_neg(lhs, s)
                } else {
                    prim_unary_values(lhs, |x| x * rhs, s)
                }
            }

            fn prim_wrapping_floor_div_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<$T> {
                let inv = 1.0 / rhs;
                prim_unary_values(lhs, |x| (x * inv).floor(), s)
            }

            fn prim_wrapping_floor_div_scalar_lhs<S: StoreIntrinsic>(
                lhs: $T,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_unary_values(rhs, |x| (lhs / x).floor(), s)
            }

            fn prim_wrapping_trunc_div_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<$T> {
                let inv = 1.0 / rhs;
                prim_unary_values(lhs, |x| (x * inv).trunc(), s)
            }

            fn prim_wrapping_trunc_div_scalar_lhs<S: StoreIntrinsic>(
                lhs: $T,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_unary_values(rhs, |x| (lhs / x).trunc(), s)
            }

            fn prim_wrapping_mod_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<$T> {
                let inv = 1.0 / rhs;
                prim_unary_values(lhs, |x| x - rhs * (x * inv).floor(), s)
            }

            fn prim_wrapping_mod_scalar_lhs<S: StoreIntrinsic>(
                lhs: $T,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_unary_values(rhs, |x| lhs - x * (lhs / x).floor(), s)
            }

            fn prim_true_div<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<Self::TrueDivT> {
                prim_binary_values(lhs, rhs, |l, r| l / r, s)
            }

            fn prim_true_div_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<Self::TrueDivT> {
                Self::prim_wrapping_mul_scalar(lhs, 1.0 / rhs, s)
            }

            fn prim_true_div_scalar_lhs<S: StoreIntrinsic>(
                lhs: $T,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<Self::TrueDivT> {
                prim_unary_values(rhs, |x| lhs / x, s)
            }
        }
    };
}

impl_float_arith_kernel!(f32);
impl_float_arith_kernel!(f64);
