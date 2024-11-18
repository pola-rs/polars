/// Functionality shared between list and array arithmetic implementations.
use arrow::array::{Array, PrimitiveArray};
use arrow::compute::utils::combine_validities_and;
use num_traits::Zero;
use polars_compute::arithmetic::ArithmeticKernel;
use polars_compute::comparisons::TotalEqKernel;
use polars_error::PolarsResult;
use polars_utils::float::IsFloat;

use super::*;
use crate::series::ChunkedArray;
use crate::utils::try_get_supertype;

#[derive(Debug, Clone)]
pub(super) enum NumericOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    FloorDiv,
}

impl NumericOp {
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Rem => "rem",
            Self::FloorDiv => "floor_div",
        }
    }

    pub(super) fn try_get_leaf_supertype(
        &self,
        prim_dtype_lhs: &DataType,
        prim_dtype_rhs: &DataType,
    ) -> PolarsResult<DataType> {
        let dtype = try_get_supertype(prim_dtype_lhs, prim_dtype_rhs)?;

        Ok(if matches!(self, Self::Div) {
            if dtype.is_float() {
                dtype
            } else {
                DataType::Float64
            }
        } else {
            dtype
        })
    }

    /// For operations that perform divisions on integers, sets the validity to NULL on rows where
    /// the denominator is 0.
    pub(super) fn prepare_numeric_op_side_validities<T: PolarsNumericType>(
        &self,
        lhs: &mut PrimitiveArray<T::Native>,
        rhs: &mut PrimitiveArray<T::Native>,
        swapped: bool,
    ) where
        PrimitiveArray<T::Native>: polars_compute::comparisons::TotalEqKernel<Scalar = T::Native>,
        T::Native: Zero + IsFloat,
    {
        if !T::Native::is_float() {
            match self {
                Self::Div | Self::Rem | Self::FloorDiv => {
                    let target = if swapped { lhs } else { rhs };
                    let ne_0 = target.tot_ne_kernel_broadcast(&T::Native::zero());
                    let validity = combine_validities_and(target.validity(), Some(&ne_0));
                    target.set_validity(validity);
                },
                _ => {},
            }
        }
    }

    /// # Panics
    /// Panics if:
    /// * lhs.len() != rhs.len()
    /// * dtype is not numeric.
    pub(super) fn apply_series(&self, lhs: &Series, rhs: &Series) -> Box<dyn Array> {
        assert_eq!(lhs.len(), rhs.len());
        debug_assert_eq!(lhs.dtype(), rhs.dtype());

        let lhs = lhs.rechunk();
        let rhs = rhs.rechunk();

        with_match_physical_numeric_polars_type!(lhs.dtype(), |$T| {
            let lhs: &ChunkedArray<$T> = lhs.as_ref().as_ref().as_ref();
            let rhs: &ChunkedArray<$T> = rhs.as_ref().as_ref().as_ref();

            let lhs = lhs.downcast_get(0).unwrap();
            let rhs = rhs.downcast_get(0).unwrap();

            Box::new(self.apply_arithmetic_kernel::<$T>(lhs.clone(), rhs.clone()))
        })
    }

    fn apply_arithmetic_kernel<T: PolarsNumericType>(
        &self,
        lhs: PrimitiveArray<T::Native>,
        rhs: PrimitiveArray<T::Native>,
    ) -> PrimitiveArray<T::Native> {
        match self {
            Self::Add => ArithmeticKernel::wrapping_add(lhs, rhs),
            Self::Sub => ArithmeticKernel::wrapping_sub(lhs, rhs),
            Self::Mul => ArithmeticKernel::wrapping_mul(lhs, rhs),
            Self::Div => ArithmeticKernel::legacy_div(lhs, rhs),
            Self::Rem => ArithmeticKernel::wrapping_mod(lhs, rhs),
            Self::FloorDiv => ArithmeticKernel::wrapping_floor_div(lhs, rhs),
        }
    }

    /// For list<->primitive where the primitive is broadcasted, we can dispatch to
    /// `ArithmeticKernel`, which can have optimized codepaths for when one side is
    /// a scalar.
    pub(super) fn apply_array_to_scalar<T: PolarsNumericType>(
        &self,
        arr_lhs: PrimitiveArray<T::Native>,
        r: T::Native,
        swapped: bool,
    ) -> PrimitiveArray<T::Native> {
        match self {
            Self::Add => ArithmeticKernel::wrapping_add_scalar(arr_lhs, r),
            Self::Sub => {
                if swapped {
                    ArithmeticKernel::wrapping_sub_scalar_lhs(r, arr_lhs)
                } else {
                    ArithmeticKernel::wrapping_sub_scalar(arr_lhs, r)
                }
            },
            Self::Mul => ArithmeticKernel::wrapping_mul_scalar(arr_lhs, r),
            Self::Div => {
                if swapped {
                    ArithmeticKernel::legacy_div_scalar_lhs(r, arr_lhs)
                } else {
                    ArithmeticKernel::legacy_div_scalar(arr_lhs, r)
                }
            },
            Self::Rem => {
                if swapped {
                    ArithmeticKernel::wrapping_mod_scalar_lhs(r, arr_lhs)
                } else {
                    ArithmeticKernel::wrapping_mod_scalar(arr_lhs, r)
                }
            },
            Self::FloorDiv => {
                if swapped {
                    ArithmeticKernel::wrapping_floor_div_scalar_lhs(r, arr_lhs)
                } else {
                    ArithmeticKernel::wrapping_floor_div_scalar(arr_lhs, r)
                }
            },
        }
    }
}

macro_rules! with_match_pl_num_arith {
    ($op:expr, $swapped:expr, | $_:tt $OP:tt | $($body:tt)* ) => ({
        macro_rules! __with_func__ {( $_ $OP:tt ) => ( $($body)* )}

        match $op {
            NumericOp::Add => __with_func__! { (PlNumArithmetic::wrapping_add) },
            NumericOp::Sub => {
                if $swapped {
                    __with_func__! { (|b, a| PlNumArithmetic::wrapping_sub(a, b)) }
                } else {
                    __with_func__! { (PlNumArithmetic::wrapping_sub) }
                }
            },
            NumericOp::Mul => __with_func__! { (PlNumArithmetic::wrapping_mul) },
            NumericOp::Div => {
                if $swapped {
                    __with_func__! { (|b, a| PlNumArithmetic::legacy_div(a, b)) }
                } else {
                    __with_func__! { (PlNumArithmetic::legacy_div) }
                }
            },
            NumericOp::Rem => {
                if $swapped {
                    __with_func__! { (|b, a| PlNumArithmetic::wrapping_mod(a, b)) }
                } else {
                    __with_func__! { (PlNumArithmetic::wrapping_mod) }
                }
            },
            NumericOp::FloorDiv => {
                if $swapped {
                    __with_func__! { (|b, a| PlNumArithmetic::wrapping_floor_div(a, b)) }
                } else {
                    __with_func__! { (PlNumArithmetic::wrapping_floor_div) }
                }
            },
        }
    })
}

pub(super) use with_match_pl_num_arith;

#[derive(Debug)]
pub(super) enum BinaryOpApplyType {
    ListToList,
    ListToPrimitive,
    PrimitiveToList,
}

#[derive(Debug)]
pub(super) enum Broadcast {
    Left,
    Right,
    #[allow(clippy::enum_variant_names)]
    NoBroadcast,
}
