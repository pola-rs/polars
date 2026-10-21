use polars_compute::arithmetic::{ArithmeticKernel, DivRounding};
use polars_core::chunked_array::ops::arity::apply_binary_kernel_broadcast;
use polars_core::prelude::*;
#[cfg(feature = "dtype-struct")]
use polars_core::series::arithmetic::_struct_arithmetic;
use polars_core::series::arithmetic::NumericListOp;
use polars_core::with_match_physical_numeric_polars_type;

fn int_div_ca<T: PolarsNumericType>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<T>,
    rounding: DivRounding,
) -> ChunkedArray<T> {
    match rounding {
        DivRounding::Floor => apply_binary_kernel_broadcast(
            lhs,
            rhs,
            |l, r| ArithmeticKernel::wrapping_floor_div(l.clone(), r.clone()),
            |l, r| ArithmeticKernel::wrapping_floor_div_scalar_lhs(l, r.clone()),
            |l, r| ArithmeticKernel::wrapping_floor_div_scalar(l.clone(), r),
        ),
        DivRounding::Trunc => apply_binary_kernel_broadcast(
            lhs,
            rhs,
            |l, r| ArithmeticKernel::wrapping_trunc_div(l.clone(), r.clone()),
            |l, r| ArithmeticKernel::wrapping_trunc_div_scalar_lhs(l, r.clone()),
            |l, r| ArithmeticKernel::wrapping_trunc_div_scalar(l.clone(), r),
        ),
    }
}

pub fn floor_div_series(a: &Series, b: &Series) -> PolarsResult<Series> {
    int_div_series(a, b, DivRounding::Floor)
}

pub fn trunc_div_series(a: &Series, b: &Series) -> PolarsResult<Series> {
    int_div_series(a, b, DivRounding::Trunc)
}

fn int_div_series(a: &Series, b: &Series, rounding: DivRounding) -> PolarsResult<Series> {
    match (a.dtype(), b.dtype()) {
        #[cfg(feature = "dtype-struct")]
        (DataType::Struct(_), DataType::Struct(_)) => {
            let recurse = match rounding {
                DivRounding::Floor => floor_div_series,
                DivRounding::Trunc => trunc_div_series,
            };
            return _struct_arithmetic(a, b, recurse);
        },
        (DataType::List(_), _) | (_, DataType::List(_)) => {
            let op = match rounding {
                DivRounding::Floor => NumericListOp::floor_div(),
                DivRounding::Trunc => NumericListOp::trunc_div(),
            };
            return op.execute(a, b);
        },
        #[cfg(feature = "dtype-array")]
        (DataType::Array(..), _) | (_, DataType::Array(..)) => {
            use polars_core::series::arithmetic::NumericFixedSizeListOp;
            let op = match rounding {
                DivRounding::Floor => NumericFixedSizeListOp::floor_div(),
                DivRounding::Trunc => NumericFixedSizeListOp::trunc_div(),
            };
            return op.execute(a, b);
        },
        _ => {},
    }

    #[cfg(feature = "dtype-decimal")]
    if a.dtype().is_decimal() || b.dtype().is_decimal() {
        let (a, b) = polars_core::series::arithmetic::coerce_lhs_rhs_numeric_op(a, b)?;
        if a.dtype().is_decimal() && b.dtype().is_decimal() {
            return Ok(a.decimal()?.int_div(b.decimal()?, rounding)?.into_series());
        }
    }

    if !a.dtype().is_primitive_numeric() {
        let op = match rounding {
            DivRounding::Floor => "floor_div",
            DivRounding::Trunc => "trunc_div",
        };
        polars_bail!(op = op, a.dtype());
    }

    let logical_type = a.dtype();

    let a = a.to_physical_repr();
    let b = b.to_physical_repr();

    let out = with_match_physical_numeric_polars_type!(a.dtype(), |$T| {
        let a: &ChunkedArray<$T> = a.as_ref().as_ref().as_ref();
        let b: &ChunkedArray<$T> = b.as_ref().as_ref().as_ref();

        int_div_ca(a, b, rounding).into_series()
    });

    unsafe { out.from_physical_unchecked(logical_type) }
}
