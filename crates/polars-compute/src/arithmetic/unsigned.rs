use arrow::array::{PrimitiveArray, StaticArray};
use arrow::compute::utils::{combine_validities_and, combine_validities_and3};
use strength_reduce::*;

use super::ArithmeticKernel;
use crate::arity::{prim_binary_values, prim_unary_values};
use crate::comparisons::TotalOrdKernel;

macro_rules! impl_unsigned_arith_kernel {
    ($T:ty, $StrRed:ty) => {
        impl ArithmeticKernel for PrimitiveArray<$T> {
            type Scalar = $T;
            type TrueDivT = f64;

            fn wrapping_neg(self) -> Self {
                prim_unary_values(self, |x| x.wrapping_neg())
            }

            fn wrapping_add(self, other: Self) -> Self {
                prim_binary_values(self, other, |a, b| a.wrapping_add(b))
            }

            fn wrapping_sub(self, other: Self) -> Self {
                prim_binary_values(self, other, |a, b| a.wrapping_sub(b))
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
                let ret = prim_binary_values(self, other, |a, b| if b != 0 { a / b } else { 0 });
                ret.with_validity(valid)
            }

            fn wrapping_mod(mut self, mut other: Self) -> Self {
                let mask = other.tot_ne_kernel_broadcast(&0);
                let valid = combine_validities_and3(
                    self.take_validity().as_ref(),  // Take validity so we don't
                    other.take_validity().as_ref(), // compute combination twice.
                    Some(&mask),
                );
                let ret = prim_binary_values(self, other, |a, b| if b != 0 { a % b } else { 0 });
                ret.with_validity(valid)
            }

            fn wrapping_add_scalar(self, scalar: Self::Scalar) -> Self {
                prim_unary_values(self, |x| x.wrapping_add(scalar))
            }

            fn wrapping_sub_scalar(self, scalar: Self::Scalar) -> Self {
                self.wrapping_add_scalar(scalar.wrapping_neg())
            }

            fn wrapping_sub_scalar_lhs(self, scalar: Self::Scalar) -> Self {
                prim_unary_values(self, |x| scalar.wrapping_sub(x))
            }

            fn wrapping_mul_scalar(self, scalar: Self::Scalar) -> Self {
                if scalar == 0 {
                    self.fill_with(0)
                } else if scalar == 1 {
                    self
                } else {
                    prim_unary_values(self, |x| x.wrapping_mul(scalar))
                }
            }

            fn wrapping_floor_div_scalar(self, scalar: Self::Scalar) -> Self {
                if scalar == 0 {
                    Self::full_null(self.len(), self.data_type().clone())
                } else if scalar == 1 {
                    self
                } else {
                    let red = <$StrRed>::new(scalar);
                    prim_unary_values(self, |x| x / red)
                }
            }

            fn wrapping_floor_div_scalar_lhs(self, scalar: Self::Scalar) -> Self {
                if scalar == 0 {
                    return self.fill_with(0);
                }

                let mask = self.tot_ne_kernel_broadcast(&0);
                let valid = combine_validities_and(self.validity(), Some(&mask));
                let ret = prim_unary_values(self, |x| if x != 0 { scalar / x } else { 0 });
                ret.with_validity(valid)
            }

            fn wrapping_mod_scalar(self, scalar: Self::Scalar) -> Self {
                if scalar == 0 {
                    Self::full_null(self.len(), self.data_type().clone())
                } else if scalar == 1 {
                    self.fill_with(0)
                } else {
                    let red = <$StrRed>::new(scalar);
                    prim_unary_values(self, |x| x % red)
                }
            }

            fn wrapping_mod_scalar_lhs(self, scalar: Self::Scalar) -> Self {
                if scalar == 0 {
                    return self.fill_with(0);
                }

                let mask = self.tot_ne_kernel_broadcast(&0);
                let valid = combine_validities_and(self.validity(), Some(&mask));
                let ret = prim_unary_values(self, |x| if x != 0 { scalar % x } else { 0 });
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

impl_unsigned_arith_kernel!(u8, StrengthReducedU8);
impl_unsigned_arith_kernel!(u16, StrengthReducedU16);
impl_unsigned_arith_kernel!(u32, StrengthReducedU32);
impl_unsigned_arith_kernel!(u64, StrengthReducedU64);
