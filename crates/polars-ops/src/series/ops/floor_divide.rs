use polars_compute::arithmetic::ArithmeticKernel;
use polars_core::chunked_array::arrow_bridge::{chunk_from_arrow, flat_to_arrow};
use polars_core::chunked_array::ops::arity::apply_binary_kernel_broadcast;
use polars_core::prelude::*;
#[cfg(feature = "dtype-struct")]
use polars_core::series::arithmetic::_struct_arithmetic;
use polars_core::series::arithmetic::NumericListOp;
use polars_core::with_match_physical_numeric_polars_type;

fn floor_div_ca<T: PolarsNumericType>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<T>,
) -> ChunkedArray<T> {
    apply_binary_kernel_broadcast(
        lhs,
        rhs,
        // TODO(polars-array-scalar): the arithmetic kernels are Arrow ones, so a chunk crosses
        // over as a flat Arrow array rather than a repeated value being divided once.
        |l, r| {
            chunk_from_arrow(&ArithmeticKernel::wrapping_floor_div(
                flat_to_arrow(l),
                flat_to_arrow(r),
            ))
        },
        |l, r| {
            chunk_from_arrow(&ArithmeticKernel::wrapping_floor_div_scalar_lhs(
                l,
                flat_to_arrow(r),
            ))
        },
        |l, r| {
            chunk_from_arrow(&ArithmeticKernel::wrapping_floor_div_scalar(
                flat_to_arrow(l),
                r,
            ))
        },
    )
}

pub fn floor_div_series(a: &Series, b: &Series) -> PolarsResult<Series> {
    match (a.dtype(), b.dtype()) {
        #[cfg(feature = "dtype-struct")]
        (DataType::Struct(_), DataType::Struct(_)) => {
            return _struct_arithmetic(a, b, floor_div_series);
        },
        (DataType::List(_), _) | (_, DataType::List(_)) => {
            return NumericListOp::floor_div().execute(a, b);
        },
        #[cfg(feature = "dtype-array")]
        (DataType::Array(..), _) | (_, DataType::Array(..)) => {
            return polars_core::series::arithmetic::NumericFixedSizeListOp::floor_div()
                .execute(a, b);
        },
        _ => {},
    }

    if !a.dtype().is_primitive_numeric() {
        polars_bail!(op = "floor_div", a.dtype());
    }

    let logical_type = a.dtype();

    let a = a.to_physical_repr();
    let b = b.to_physical_repr();

    let out = with_match_physical_numeric_polars_type!(a.dtype(), |$T| {
        let a: &ChunkedArray<$T> = a.as_ref().as_ref().as_ref();
        let b: &ChunkedArray<$T> = b.as_ref().as_ref().as_ref();

        floor_div_ca(a, b).into_series()
    });

    unsafe { out.from_physical_unchecked(logical_type) }
}
