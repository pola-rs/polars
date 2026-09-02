use super::*;
use crate::chunked_array::arithmetic::kernel::PlArithmeticKernel;
use crate::chunked_array::arity::{
    apply_binary_kernel_broadcast, apply_binary_kernel_broadcast_owned, unary_kernel_flat,
    unary_kernel_owned_flat,
};

macro_rules! impl_op_overload {
    ($op: ident, $trait_method: ident, $ca_method: ident, $ca_method_scalar: ident) => {
        impl<T: PolarsNumericType> $op for ChunkedArray<T> {
            type Output = ChunkedArray<T>;

            fn $trait_method(self, rhs: Self) -> Self::Output {
                ArithmeticChunked::$ca_method(self, rhs)
            }
        }

        impl<T: PolarsNumericType> $op for &ChunkedArray<T> {
            type Output = ChunkedArray<T>;

            fn $trait_method(self, rhs: Self) -> Self::Output {
                ArithmeticChunked::$ca_method(self, rhs)
            }
        }

        // TODO: make this more strict instead of casting.
        impl<T: PolarsNumericType, N: Num + ToPrimitive> $op<N> for ChunkedArray<T> {
            type Output = ChunkedArray<T>;

            fn $trait_method(self, rhs: N) -> Self::Output {
                let rhs: T::Native = NumCast::from(rhs).unwrap();
                ArithmeticChunked::$ca_method_scalar(self, rhs)
            }
        }

        impl<T: PolarsNumericType, N: Num + ToPrimitive> $op<N> for &ChunkedArray<T> {
            type Output = ChunkedArray<T>;

            fn $trait_method(self, rhs: N) -> Self::Output {
                let rhs: T::Native = NumCast::from(rhs).unwrap();
                ArithmeticChunked::$ca_method_scalar(self, rhs)
            }
        }
    };
}

impl_op_overload!(Add, add, wrapping_add, wrapping_add_scalar);
impl_op_overload!(Sub, sub, wrapping_sub, wrapping_sub_scalar);
impl_op_overload!(Mul, mul, wrapping_mul, wrapping_mul_scalar);
impl_op_overload!(Div, div, legacy_div, legacy_div_scalar); // TODO: replace this with true division.
impl_op_overload!(Rem, rem, wrapping_mod, wrapping_mod_scalar);

pub trait ArithmeticChunked {
    type Scalar;
    type Out;
    type TrueDivOut;

    fn wrapping_abs(self) -> Self::Out;
    fn wrapping_neg(self) -> Self::Out;
    fn wrapping_add(self, rhs: Self) -> Self::Out;
    fn wrapping_sub(self, rhs: Self) -> Self::Out;
    fn wrapping_mul(self, rhs: Self) -> Self::Out;
    fn wrapping_floor_div(self, rhs: Self) -> Self::Out;
    fn wrapping_trunc_div(self, rhs: Self) -> Self::Out;
    fn wrapping_mod(self, rhs: Self) -> Self::Out;

    fn wrapping_add_scalar(self, rhs: Self::Scalar) -> Self::Out;
    fn wrapping_sub_scalar(self, rhs: Self::Scalar) -> Self::Out;
    fn wrapping_sub_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out;
    fn wrapping_mul_scalar(self, rhs: Self::Scalar) -> Self::Out;
    fn wrapping_floor_div_scalar(self, rhs: Self::Scalar) -> Self::Out;
    fn wrapping_floor_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out;
    fn wrapping_trunc_div_scalar(self, rhs: Self::Scalar) -> Self::Out;
    fn wrapping_trunc_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out;
    fn wrapping_mod_scalar(self, rhs: Self::Scalar) -> Self::Out;
    fn wrapping_mod_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out;

    fn checked_mul_scalar(self, rhs: Self::Scalar) -> Self::Out;

    fn true_div(self, rhs: Self) -> Self::TrueDivOut;
    fn true_div_scalar(self, rhs: Self::Scalar) -> Self::TrueDivOut;
    fn true_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::TrueDivOut;

    // TODO: remove these.
    // These are flooring division for integer types, true division for floating point types.
    fn legacy_div(self, rhs: Self) -> Self::Out;
    fn legacy_div_scalar(self, rhs: Self::Scalar) -> Self::Out;
    fn legacy_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out;
}

impl<T: PolarsNumericType> ArithmeticChunked for ChunkedArray<T> {
    type Scalar = T::Native;
    type Out = ChunkedArray<T>;
    type TrueDivOut = ChunkedArray<<T::Native as NumericNative>::TrueDivPolarsType>;

    fn wrapping_abs(self) -> Self::Out {
        unary_kernel_owned_flat(self, PlArithmeticKernel::wrapping_abs)
    }

    fn wrapping_neg(self) -> Self::Out {
        unary_kernel_owned_flat(self, PlArithmeticKernel::wrapping_neg)
    }

    fn wrapping_add(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            PlArithmeticKernel::wrapping_add,
            |l, r| PlArithmeticKernel::wrapping_add_scalar(r, l),
            PlArithmeticKernel::wrapping_add_scalar,
        )
    }

    fn wrapping_sub(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            PlArithmeticKernel::wrapping_sub,
            PlArithmeticKernel::wrapping_sub_scalar_lhs,
            PlArithmeticKernel::wrapping_sub_scalar,
        )
    }

    fn wrapping_mul(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            PlArithmeticKernel::wrapping_mul,
            |l, r| PlArithmeticKernel::wrapping_mul_scalar(r, l),
            PlArithmeticKernel::wrapping_mul_scalar,
        )
    }

    fn wrapping_floor_div(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            PlArithmeticKernel::wrapping_floor_div,
            PlArithmeticKernel::wrapping_floor_div_scalar_lhs,
            PlArithmeticKernel::wrapping_floor_div_scalar,
        )
    }

    fn wrapping_trunc_div(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            PlArithmeticKernel::wrapping_trunc_div,
            PlArithmeticKernel::wrapping_trunc_div_scalar_lhs,
            PlArithmeticKernel::wrapping_trunc_div_scalar,
        )
    }

    fn wrapping_mod(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            PlArithmeticKernel::wrapping_mod,
            PlArithmeticKernel::wrapping_mod_scalar_lhs,
            PlArithmeticKernel::wrapping_mod_scalar,
        )
    }

    fn wrapping_add_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_owned_flat(self, |a| PlArithmeticKernel::wrapping_add_scalar(a, rhs))
    }

    fn wrapping_sub_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_owned_flat(self, |a| PlArithmeticKernel::wrapping_sub_scalar(a, rhs))
    }

    fn wrapping_sub_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out {
        unary_kernel_owned_flat(rhs, |a| PlArithmeticKernel::wrapping_sub_scalar_lhs(lhs, a))
    }

    fn wrapping_mul_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_owned_flat(self, |a| PlArithmeticKernel::wrapping_mul_scalar(a, rhs))
    }

    fn wrapping_floor_div_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_owned_flat(self, |a| {
            PlArithmeticKernel::wrapping_floor_div_scalar(a, rhs)
        })
    }

    fn wrapping_floor_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out {
        unary_kernel_owned_flat(rhs, |a| {
            PlArithmeticKernel::wrapping_floor_div_scalar_lhs(lhs, a)
        })
    }

    fn wrapping_trunc_div_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_owned_flat(self, |a| {
            PlArithmeticKernel::wrapping_trunc_div_scalar(a, rhs)
        })
    }

    fn wrapping_trunc_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out {
        unary_kernel_owned_flat(rhs, |a| {
            PlArithmeticKernel::wrapping_trunc_div_scalar_lhs(lhs, a)
        })
    }

    fn wrapping_mod_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_owned_flat(self, |a| PlArithmeticKernel::wrapping_mod_scalar(a, rhs))
    }

    fn wrapping_mod_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out {
        unary_kernel_owned_flat(rhs, |a| PlArithmeticKernel::wrapping_mod_scalar_lhs(lhs, a))
    }

    fn checked_mul_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_owned_flat(self, |a| PlArithmeticKernel::checked_mul_scalar(a, rhs))
    }

    fn true_div(self, rhs: Self) -> Self::TrueDivOut {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            PlArithmeticKernel::true_div,
            PlArithmeticKernel::true_div_scalar_lhs,
            PlArithmeticKernel::true_div_scalar,
        )
    }

    fn true_div_scalar(self, rhs: Self::Scalar) -> Self::TrueDivOut {
        unary_kernel_owned_flat(self, |a| PlArithmeticKernel::true_div_scalar(a, rhs))
    }

    fn true_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::TrueDivOut {
        unary_kernel_owned_flat(rhs, |a| PlArithmeticKernel::true_div_scalar_lhs(lhs, a))
    }

    fn legacy_div(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            PlArithmeticKernel::legacy_div,
            PlArithmeticKernel::legacy_div_scalar_lhs,
            PlArithmeticKernel::legacy_div_scalar,
        )
    }

    fn legacy_div_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_owned_flat(self, |a| PlArithmeticKernel::legacy_div_scalar(a, rhs))
    }

    fn legacy_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out {
        unary_kernel_owned_flat(rhs, |a| PlArithmeticKernel::legacy_div_scalar_lhs(lhs, a))
    }
}

impl<T: PolarsNumericType> ArithmeticChunked for &ChunkedArray<T> {
    type Scalar = T::Native;
    type Out = ChunkedArray<T>;
    type TrueDivOut = ChunkedArray<<T::Native as NumericNative>::TrueDivPolarsType>;

    fn wrapping_abs(self) -> Self::Out {
        unary_kernel_flat(self, |a| PlArithmeticKernel::wrapping_abs(a.clone()))
    }

    fn wrapping_neg(self) -> Self::Out {
        unary_kernel_flat(self, |a| PlArithmeticKernel::wrapping_neg(a.clone()))
    }

    fn wrapping_add(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| PlArithmeticKernel::wrapping_add(l.clone(), r.clone()),
            |l, r| PlArithmeticKernel::wrapping_add_scalar(r.clone(), l),
            |l, r| PlArithmeticKernel::wrapping_add_scalar(l.clone(), r),
        )
    }

    fn wrapping_sub(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| PlArithmeticKernel::wrapping_sub(l.clone(), r.clone()),
            |l, r| PlArithmeticKernel::wrapping_sub_scalar_lhs(l, r.clone()),
            |l, r| PlArithmeticKernel::wrapping_sub_scalar(l.clone(), r),
        )
    }

    fn wrapping_mul(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| PlArithmeticKernel::wrapping_mul(l.clone(), r.clone()),
            |l, r| PlArithmeticKernel::wrapping_mul_scalar(r.clone(), l),
            |l, r| PlArithmeticKernel::wrapping_mul_scalar(l.clone(), r),
        )
    }

    fn wrapping_floor_div(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| PlArithmeticKernel::wrapping_floor_div(l.clone(), r.clone()),
            |l, r| PlArithmeticKernel::wrapping_floor_div_scalar_lhs(l, r.clone()),
            |l, r| PlArithmeticKernel::wrapping_floor_div_scalar(l.clone(), r),
        )
    }

    fn wrapping_trunc_div(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| PlArithmeticKernel::wrapping_trunc_div(l.clone(), r.clone()),
            |l, r| PlArithmeticKernel::wrapping_trunc_div_scalar_lhs(l, r.clone()),
            |l, r| PlArithmeticKernel::wrapping_trunc_div_scalar(l.clone(), r),
        )
    }

    fn wrapping_mod(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| PlArithmeticKernel::wrapping_mod(l.clone(), r.clone()),
            |l, r| PlArithmeticKernel::wrapping_mod_scalar_lhs(l, r.clone()),
            |l, r| PlArithmeticKernel::wrapping_mod_scalar(l.clone(), r),
        )
    }

    fn wrapping_add_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_flat(self, |a| {
            PlArithmeticKernel::wrapping_add_scalar(a.clone(), rhs)
        })
    }

    fn wrapping_sub_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_flat(self, |a| {
            PlArithmeticKernel::wrapping_sub_scalar(a.clone(), rhs)
        })
    }

    fn wrapping_sub_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out {
        unary_kernel_flat(rhs, |a| {
            PlArithmeticKernel::wrapping_sub_scalar_lhs(lhs, a.clone())
        })
    }

    fn wrapping_mul_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_flat(self, |a| {
            PlArithmeticKernel::wrapping_mul_scalar(a.clone(), rhs)
        })
    }

    fn wrapping_floor_div_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_flat(self, |a| {
            PlArithmeticKernel::wrapping_floor_div_scalar(a.clone(), rhs)
        })
    }

    fn wrapping_floor_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out {
        unary_kernel_flat(rhs, |a| {
            PlArithmeticKernel::wrapping_floor_div_scalar_lhs(lhs, a.clone())
        })
    }

    fn wrapping_trunc_div_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_flat(self, |a| {
            PlArithmeticKernel::wrapping_trunc_div_scalar(a.clone(), rhs)
        })
    }

    fn wrapping_trunc_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out {
        unary_kernel_flat(rhs, |a| {
            PlArithmeticKernel::wrapping_trunc_div_scalar_lhs(lhs, a.clone())
        })
    }

    fn wrapping_mod_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_flat(self, |a| {
            PlArithmeticKernel::wrapping_mod_scalar(a.clone(), rhs)
        })
    }

    fn wrapping_mod_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out {
        unary_kernel_flat(rhs, |a| {
            PlArithmeticKernel::wrapping_mod_scalar_lhs(lhs, a.clone())
        })
    }

    fn checked_mul_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_flat(self, |a| {
            PlArithmeticKernel::checked_mul_scalar(a.clone(), rhs)
        })
    }

    fn true_div(self, rhs: Self) -> Self::TrueDivOut {
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| PlArithmeticKernel::true_div(l.clone(), r.clone()),
            |l, r| PlArithmeticKernel::true_div_scalar_lhs(l, r.clone()),
            |l, r| PlArithmeticKernel::true_div_scalar(l.clone(), r),
        )
    }

    fn true_div_scalar(self, rhs: Self::Scalar) -> Self::TrueDivOut {
        unary_kernel_flat(self, |a| {
            PlArithmeticKernel::true_div_scalar(a.clone(), rhs)
        })
    }

    fn true_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::TrueDivOut {
        unary_kernel_flat(rhs, |a| {
            PlArithmeticKernel::true_div_scalar_lhs(lhs, a.clone())
        })
    }

    fn legacy_div(self, rhs: Self) -> Self::Out {
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| PlArithmeticKernel::legacy_div(l.clone(), r.clone()),
            |l, r| PlArithmeticKernel::legacy_div_scalar_lhs(l, r.clone()),
            |l, r| PlArithmeticKernel::legacy_div_scalar(l.clone(), r),
        )
    }

    fn legacy_div_scalar(self, rhs: Self::Scalar) -> Self::Out {
        unary_kernel_flat(self, |a| {
            PlArithmeticKernel::legacy_div_scalar(a.clone(), rhs)
        })
    }

    fn legacy_div_scalar_lhs(lhs: Self::Scalar, rhs: Self) -> Self::Out {
        unary_kernel_flat(rhs, |a| {
            PlArithmeticKernel::legacy_div_scalar_lhs(lhs, a.clone())
        })
    }
}
