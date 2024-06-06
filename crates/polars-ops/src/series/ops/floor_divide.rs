use polars_compute::arithmetic::ArithmeticKernel;
use polars_core::chunked_array::ops::arity::apply_binary_kernel_broadcast;
use polars_core::prelude::*;
#[cfg(feature = "dtype-struct")]
use polars_core::series::arithmetic::_struct_arithmetic;
use polars_core::with_match_physical_numeric_polars_type;

fn floor_div_ca<T: PolarsNumericType>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<T>,
) -> ChunkedArray<T> {
    apply_binary_kernel_broadcast(
        lhs,
        rhs,
        |l, r| ArithmeticKernel::wrapping_floor_div(l.clone(), r.clone()),
        |l, r| ArithmeticKernel::wrapping_floor_div_scalar_lhs(l, r.clone()),
        |l, r| ArithmeticKernel::wrapping_floor_div_scalar(l.clone(), r),
    )
}

pub fn floor_div_series(a: &Series, b: &Series) -> PolarsResult<Series> {
    match (a.dtype(), b.dtype()) {
        #[cfg(feature = "dtype-struct")]
        (DataType::Struct(_), DataType::Struct(_)) => {
            return _struct_arithmetic(a, b, floor_div_series);
        },
        _ => {},
    }

    if !a.dtype().is_numeric() {
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

    out.cast(logical_type)
}
