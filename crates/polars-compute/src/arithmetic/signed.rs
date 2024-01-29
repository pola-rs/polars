use arrow::array::{PrimitiveArray, StaticArray};
use arrow::compute::utils::{combine_validities_and, combine_validities_and3};
use polars_utils::signed_divmod::SignedDivMod;
use strength_reduce::*;

use super::ArithmeticKernel;
use crate::arity::{prim_binary_values, prim_unary_values};
use crate::comparisons::TotalOrdKernel;

macro_rules! impl_signed_arith_kernel {
    ($T:ty, $U:ty, $StrRed:ty) => {
        impl ArithmeticKernel for PrimitiveArray<$T> {
            type Scalar = $T;
            type TrueDivT = f64;

            fn wrapping_neg(self) -> Self {
                // Wrapping signed and unsigned addition/subtraction are the same.
                self.transmute::<$U>().wrapping_neg().transmute::<$T>()
            }

            fn wrapping_add(self, other: Self) -> Self {
                // Wrapping signed and unsigned addition/subtraction are the same.
                let lhs = self.transmute::<$U>();
                let rhs = other.transmute::<$U>();
                lhs.wrapping_add(rhs).transmute::<$T>()
            }

            fn wrapping_sub(self, other: Self) -> Self {
                // Wrapping signed and unsigned addition/subtraction are the same.
                let lhs = self.transmute::<$U>();
                let rhs = other.transmute::<$U>();
                lhs.wrapping_sub(rhs).transmute::<$T>()
            }

            fn wrapping_mul(self, other: Self) -> Self {
                prim_binary_values(self, other, |a, b| a.wrapping_mul(b))
            }

            fn wrapping_floor_div(mut self, mut other: Self) -> Self {
                let mask = other.tot_ne_kernel_broadcast(&0);
                let valid = combine_validities_and3(
                    self.take_validity().as_ref(),  // Take validity so we don't
                    other.take_validity().as_ref(), // compute combination twice.
                    Some(&mask),
                );
                let ret = prim_binary_values(self, other, |lhs, rhs| lhs.wrapping_div_mod(rhs).0);
                ret.with_validity(valid)
            }

            fn wrapping_mod(mut self, mut other: Self) -> Self {
                let mask = other.tot_ne_kernel_broadcast(&0);
                let valid = combine_validities_and3(
                    self.take_validity().as_ref(),  // Take validity so we don't
                    other.take_validity().as_ref(), // compute combination twice.
                    Some(&mask),
                );
                let ret = prim_binary_values(self, other, |lhs, rhs| lhs.wrapping_div_mod(rhs).1);
                ret.with_validity(valid)
            }

            fn wrapping_add_scalar(self, scalar: Self::Scalar) -> Self {
                // Wrapping signed and unsigned addition/subtraction are the same.
                let lhs = self.transmute::<$U>();
                let rhs = scalar as $U;
                lhs.wrapping_add_scalar(rhs).transmute::<$T>()
            }

            fn wrapping_sub_scalar(self, scalar: Self::Scalar) -> Self {
                // Wrapping signed and unsigned addition/subtraction are the same.
                let lhs = self.transmute::<$U>();
                let rhs = scalar as $U;
                lhs.wrapping_sub_scalar(rhs).transmute::<$T>()
            }

            fn wrapping_sub_scalar_lhs(self, scalar: Self::Scalar) -> Self {
                // Wrapping signed and unsigned addition/subtraction are the same.
                let rhs = self.transmute::<$U>();
                let lhs = scalar as $U;
                rhs.wrapping_sub_scalar_lhs(lhs).transmute::<$T>()
            }

            fn wrapping_mul_scalar(self, scalar: Self::Scalar) -> Self {
                let scalar_u = scalar.unsigned_abs();
                if scalar == 0 {
                    self.fill_with(0)
                } else if scalar == 1 {
                    self
                } else if scalar_u & (scalar_u - 1) == 0 {
                    // Power of two.
                    let shift = scalar_u.trailing_zeros();
                    if scalar > 0 {
                        prim_unary_values(self, |x| x << shift)
                    } else {
                        prim_unary_values(self, |x| (x << shift).wrapping_neg())
                    }
                } else {
                    prim_unary_values(self, |x| x.wrapping_mul(scalar))
                }
            }

            fn wrapping_floor_div_scalar(self, scalar: Self::Scalar) -> Self {
                if scalar == 0 {
                    Self::full_null(self.len(), self.data_type().clone())
                } else if scalar == -1 {
                    self.wrapping_neg()
                } else if scalar == 1 {
                    self
                } else {
                    let red = <$StrRed>::new(scalar.unsigned_abs());
                    prim_unary_values(self, |x| {
                        let (quot, rem) = <$StrRed>::div_rem(x.unsigned_abs(), red);
                        if (x < 0) != (scalar < 0) {
                            // Different signs: result should be negative.
                            // Since we handled scalar.abs() <= 1, quot fits.
                            let mut ret = -(quot as $T);
                            if rem != 0 {
                                // Division had remainder, subtract 1 to floor to
                                // negative infinity, as we truncated to zero.
                                ret -= 1;
                            }
                            ret
                        } else {
                            quot as $T
                        }
                    })
                }
            }

            fn wrapping_floor_div_scalar_lhs(self, scalar: Self::Scalar) -> Self {
                if scalar == 0 {
                    return self.fill_with(0);
                }

                let mask = self.tot_ne_kernel_broadcast(&0);
                let valid = combine_validities_and(self.validity(), Some(&mask));
                let ret = prim_unary_values(self, |x| scalar.wrapping_div_mod(x).0);
                ret.with_validity(valid)
            }

            fn wrapping_mod_scalar(self, scalar: Self::Scalar) -> Self {
                if scalar == 0 {
                    Self::full_null(self.len(), self.data_type().clone())
                } else if scalar == -1 || scalar == 1 {
                    self.fill_with(0)
                } else {
                    let scalar_u = scalar.unsigned_abs();
                    let red = <$StrRed>::new(scalar_u);
                    prim_unary_values(self, |x| {
                        // Remainder fits in signed type after reduction.
                        // Largest possible modulo -I::MIN, with
                        // -I::MIN-1 == I::MAX as largest remainder.
                        let mut rem_u = x.unsigned_abs() % red;

                        // Mixed signs: swap direction of remainder.
                        if rem_u != 0 && (scalar < 0) != (x < 0) {
                            rem_u = scalar_u - rem_u;
                        }

                        // Remainder should have sign of RHS.
                        if scalar < 0 {
                            -(rem_u as $T)
                        } else {
                            rem_u as $T
                        }
                    })
                }
            }

            fn wrapping_mod_scalar_lhs(self, scalar: Self::Scalar) -> Self {
                if scalar == 0 {
                    return self.fill_with(0);
                }

                let mask = self.tot_ne_kernel_broadcast(&0);
                let valid = combine_validities_and(self.validity(), Some(&mask));
                let ret = prim_unary_values(self, |x| scalar.wrapping_div_mod(x).1);
                ret.with_validity(valid)
            }

            fn true_div(self, other: Self) -> PrimitiveArray<Self::TrueDivT> {
                prim_binary_values(self, other, |a, b| a as f64 / b as f64)
            }

            fn true_div_scalar(self, scalar: Self::Scalar) -> PrimitiveArray<Self::TrueDivT> {
                let inv = 1.0 / scalar as f64;
                prim_unary_values(self, |x| x as f64 * inv)
            }

            fn true_div_scalar_lhs(self, scalar: Self::Scalar) -> PrimitiveArray<Self::TrueDivT> {
                prim_unary_values(self, |x| scalar as f64 / x as f64)
            }
        }
    };
}

impl_signed_arith_kernel!(i8, u8, StrengthReducedU8);
impl_signed_arith_kernel!(i16, u16, StrengthReducedU16);
impl_signed_arith_kernel!(i32, u32, StrengthReducedU32);
impl_signed_arith_kernel!(i64, u64, StrengthReducedU64);
