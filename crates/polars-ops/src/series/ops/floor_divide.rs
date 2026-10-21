use polars_compute::arithmetic::ArithmeticKernel;
use polars_core::chunked_array::ops::arity::apply_binary_kernel_broadcast;
use polars_core::prelude::*;
#[cfg(feature = "dtype-struct")]
use polars_core::series::arithmetic::_struct_arithmetic;
use polars_core::series::arithmetic::NumericListOp;
use polars_core::with_match_physical_numeric_polars_type;

fn int_div_ca<T: PolarsNumericType>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<T>,
    floor: bool,
) -> ChunkedArray<T> {
    if floor {
        apply_binary_kernel_broadcast(
            lhs,
            rhs,
            |l, r| ArithmeticKernel::wrapping_floor_div(l.clone(), r.clone()),
            |l, r| ArithmeticKernel::wrapping_floor_div_scalar_lhs(l, r.clone()),
            |l, r| ArithmeticKernel::wrapping_floor_div_scalar(l.clone(), r),
        )
    } else {
        apply_binary_kernel_broadcast(
            lhs,
            rhs,
            |l, r| ArithmeticKernel::wrapping_trunc_div(l.clone(), r.clone()),
            |l, r| ArithmeticKernel::wrapping_trunc_div_scalar_lhs(l, r.clone()),
            |l, r| ArithmeticKernel::wrapping_trunc_div_scalar(l.clone(), r),
        )
    }
}

pub fn floor_div_series(a: &Series, b: &Series) -> PolarsResult<Series> {
    int_div_series(a, b, true)
}

pub fn trunc_div_series(a: &Series, b: &Series) -> PolarsResult<Series> {
    int_div_series(a, b, false)
}

fn int_div_series(a: &Series, b: &Series, floor: bool) -> PolarsResult<Series> {
    match (a.dtype(), b.dtype()) {
        #[cfg(feature = "dtype-struct")]
        (DataType::Struct(_), DataType::Struct(_)) => {
            let recurse = if floor {
                floor_div_series
            } else {
                trunc_div_series
            };
            return _struct_arithmetic(a, b, recurse);
        },
        (DataType::List(_), _) | (_, DataType::List(_)) => {
            let op = if floor {
                NumericListOp::floor_div()
            } else {
                NumericListOp::trunc_div()
            };
            return op.execute(a, b);
        },
        #[cfg(feature = "dtype-array")]
        (DataType::Array(..), _) | (_, DataType::Array(..)) => {
            use polars_core::series::arithmetic::NumericFixedSizeListOp;
            let op = if floor {
                NumericFixedSizeListOp::floor_div()
            } else {
                NumericFixedSizeListOp::trunc_div()
            };
            return op.execute(a, b);
        },
        _ => {},
    }

    #[cfg(feature = "dtype-decimal")]
    if a.dtype().is_decimal() || b.dtype().is_decimal() {
        let (a, b) = polars_core::series::arithmetic::coerce_lhs_rhs_numeric_op(a, b)?;
        if a.dtype().is_decimal() && b.dtype().is_decimal() {
            return Ok(a.decimal()?.int_div(b.decimal()?, floor)?.into_series());
        }
    }

    if !a.dtype().is_primitive_numeric() {
        let op = if floor { "floor_div" } else { "trunc_div" };
        polars_bail!(op = op, a.dtype());
    }

    let logical_type = a.dtype();

    let a = a.to_physical_repr();
    let b = b.to_physical_repr();

    let out = with_match_physical_numeric_polars_type!(a.dtype(), |$T| {
        let a: &ChunkedArray<$T> = a.as_ref().as_ref().as_ref();
        let b: &ChunkedArray<$T> = b.as_ref().as_ref().as_ref();

        int_div_ca(a, b, floor).into_series()
    });

    unsafe { out.from_physical_unchecked(logical_type) }
}
