use core::any::TypeId;

use arrow::types::NativeType;
use polars_utils::floor_divmod::FloorDivMod;

/// Implements basic arithmetic between scalars with the same behavior as `ArithmeticKernel`.
///
/// Note, however, that the user is responsible for setting the validity of
/// results for e.g. div/mod operations with 0 in the denominator.
///
/// This is intended as a low-level utility for custom arithmetic loops
/// (e.g. in list arithmetic). In most cases prefer using `ArithmeticKernel` or
/// `ArithmeticChunked` instead.
pub trait PlNumArithmetic: Sized + Copy + 'static {
    type TrueDivT: NativeType;

    fn wrapping_abs(self) -> Self;
    fn wrapping_neg(self) -> Self;
    fn wrapping_add(self, rhs: Self) -> Self;
    fn wrapping_sub(self, rhs: Self) -> Self;
    fn wrapping_mul(self, rhs: Self) -> Self;
    fn wrapping_floor_div(self, rhs: Self) -> Self;
    fn wrapping_trunc_div(self, rhs: Self) -> Self;
    fn wrapping_mod(self, rhs: Self) -> Self;

    fn true_div(self, rhs: Self) -> Self::TrueDivT;

    #[inline(always)]
    fn legacy_div(self, rhs: Self) -> Self {
        if TypeId::of::<Self>() == TypeId::of::<Self::TrueDivT>() {
            let ret = self.true_div(rhs);
            unsafe { core::mem::transmute_copy(&ret) }
        } else {
            self.wrapping_floor_div(rhs)
        }
    }
}

macro_rules! impl_signed_pl_num_arith {
    ($T:ty) => {
        impl PlNumArithmetic for $T {
            type TrueDivT = f64;

            #[inline(always)]
            fn wrapping_abs(self) -> Self {
                self.wrapping_abs()
            }

            #[inline(always)]
            fn wrapping_neg(self) -> Self {
                self.wrapping_neg()
            }

            #[inline(always)]
            fn wrapping_add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn wrapping_sub(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn wrapping_mul(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            fn wrapping_floor_div(self, rhs: Self) -> Self {
                self.wrapping_floor_div_mod(rhs).0
            }

            #[inline(always)]
            fn wrapping_trunc_div(self, rhs: Self) -> Self {
                if rhs != 0 { self.wrapping_div(rhs) } else { 0 }
            }

            #[inline(always)]
            fn wrapping_mod(self, rhs: Self) -> Self {
                self.wrapping_floor_div_mod(rhs).1
            }

            #[inline(always)]
            fn true_div(self, rhs: Self) -> Self::TrueDivT {
                self as f64 / rhs as f64
            }
        }
    };
}

impl_signed_pl_num_arith!(i8);
impl_signed_pl_num_arith!(i16);
impl_signed_pl_num_arith!(i32);
impl_signed_pl_num_arith!(i64);
impl_signed_pl_num_arith!(i128);

macro_rules! impl_unsigned_pl_num_arith {
    ($T:ty) => {
        impl PlNumArithmetic for $T {
            type TrueDivT = f64;

            #[inline(always)]
            fn wrapping_abs(self) -> Self {
                self
            }

            #[inline(always)]
            fn wrapping_neg(self) -> Self {
                self.wrapping_neg()
            }

            #[inline(always)]
            fn wrapping_add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn wrapping_sub(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn wrapping_mul(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            fn wrapping_floor_div(self, rhs: Self) -> Self {
                if rhs != 0 { self / rhs } else { 0 }
            }

            #[inline(always)]
            fn wrapping_trunc_div(self, rhs: Self) -> Self {
                self.wrapping_floor_div(rhs)
            }

            #[inline(always)]
            fn wrapping_mod(self, rhs: Self) -> Self {
                if rhs != 0 { self % rhs } else { 0 }
            }

            #[inline(always)]
            fn true_div(self, rhs: Self) -> Self::TrueDivT {
                self as f64 / rhs as f64
            }
        }
    };
}

impl_unsigned_pl_num_arith!(u8);
impl_unsigned_pl_num_arith!(u16);
impl_unsigned_pl_num_arith!(u32);
impl_unsigned_pl_num_arith!(u64);
impl_unsigned_pl_num_arith!(u128);

macro_rules! impl_float_pl_num_arith {
    ($T:ty) => {
        impl PlNumArithmetic for $T {
            type TrueDivT = $T;

            #[inline(always)]
            fn wrapping_abs(self) -> Self {
                self.abs()
            }

            #[inline(always)]
            fn wrapping_neg(self) -> Self {
                -self
            }

            #[inline(always)]
            fn wrapping_add(self, rhs: Self) -> Self {
                self + rhs
            }

            #[inline(always)]
            fn wrapping_sub(self, rhs: Self) -> Self {
                self - rhs
            }

            #[inline(always)]
            fn wrapping_mul(self, rhs: Self) -> Self {
                self * rhs
            }

            #[inline(always)]
            fn wrapping_floor_div(self, rhs: Self) -> Self {
                let l = self;
                let r = rhs;
                (l / r).floor()
            }

            #[inline(always)]
            fn wrapping_trunc_div(self, rhs: Self) -> Self {
                let l = self;
                let r = rhs;
                (l / r).trunc()
            }

            #[inline(always)]
            fn wrapping_mod(self, rhs: Self) -> Self {
                let l = self;
                let r = rhs;
                l - r * (l / r).floor()
            }

            #[inline(always)]
            fn true_div(self, rhs: Self) -> Self::TrueDivT {
                self / rhs
            }
        }
    };
}

impl_float_pl_num_arith!(f32);
impl_float_pl_num_arith!(f64);
