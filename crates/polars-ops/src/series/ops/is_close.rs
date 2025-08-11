use std::cmp::max_by;

use arrow::array::BooleanArray;
use arrow::compute::utils::combine_validities_and;
use num_traits::AsPrimitive;
use polars_core::prelude::arity::apply_binary_kernel_broadcast;
use polars_core::prelude::*;

pub fn is_close(
    s: &Series,
    other: &Series,
    abs_tol: f64,
    rel_tol: f64,
    nans_equal: bool,
) -> PolarsResult<BooleanChunked> {
    if abs_tol < 0.0 {
        polars_bail!(ComputeError: "`abs_tol` must be non-negative but got {}", abs_tol);
    }
    if rel_tol < 0.0 {
        polars_bail!(ComputeError: "`rel_tol` must be non-negative but got {}", rel_tol);
    }
    validate_numeric(s.dtype())?;
    validate_numeric(other.dtype())?;

    let ca = match (s.dtype(), other.dtype()) {
        (DataType::Float32, DataType::Float32) => apply_binary_kernel_broadcast(
            s.f32().unwrap(),
            other.f32().unwrap(),
            |l, r| is_close_kernel::<Float32Type>(l, r, abs_tol, rel_tol, nans_equal),
            |v, ca| is_close_kernel_unary::<Float32Type>(ca, v.as_(), abs_tol, rel_tol, nans_equal),
            |ca, v| is_close_kernel_unary::<Float32Type>(ca, v.as_(), abs_tol, rel_tol, nans_equal),
        ),
        (DataType::Float64, DataType::Float64) => apply_binary_kernel_broadcast(
            s.f64().unwrap(),
            other.f64().unwrap(),
            |l, r| is_close_kernel::<Float64Type>(l, r, abs_tol, rel_tol, nans_equal),
            |v, ca| is_close_kernel_unary::<Float64Type>(ca, v.as_(), abs_tol, rel_tol, nans_equal),
            |ca, v| is_close_kernel_unary::<Float64Type>(ca, v.as_(), abs_tol, rel_tol, nans_equal),
        ),
        _ => apply_binary_kernel_broadcast(
            s.cast(&DataType::Float64)?.f64().unwrap(),
            other.cast(&DataType::Float64)?.f64().unwrap(),
            |l, r| is_close_kernel::<Float64Type>(l, r, abs_tol, rel_tol, nans_equal),
            |v, ca| is_close_kernel_unary::<Float64Type>(ca, v.as_(), abs_tol, rel_tol, nans_equal),
            |ca, v| is_close_kernel_unary::<Float64Type>(ca, v.as_(), abs_tol, rel_tol, nans_equal),
        ),
    };
    Ok(ca)
}

fn validate_numeric(dtype: &DataType) -> PolarsResult<()> {
    if !dtype.is_primitive_numeric() && !dtype.is_decimal() {
        polars_bail!(
            op = "is_close",
            dtype,
            hint = "`is_close` is only supported for numeric types"
        );
    }
    Ok(())
}

/* ------------------------------------------- KERNEL ------------------------------------------ */

fn is_close_kernel<T>(
    lhs_arr: &T::Array,
    rhs_arr: &T::Array,
    abs_tol: f64,
    rel_tol: f64,
    nans_equal: bool,
) -> BooleanArray
where
    T: PolarsNumericType,
{
    let validity = combine_validities_and(lhs_arr.validity(), rhs_arr.validity());
    let element_iter = lhs_arr
        .values_iter()
        .zip(rhs_arr.values_iter())
        .map(|(x, y)| is_close_scalar(x.as_(), y.as_(), abs_tol, rel_tol, nans_equal));
    let result: BooleanArray = element_iter.collect_arr();
    result.with_validity_typed(validity)
}

fn is_close_kernel_unary<T>(
    arr: &T::Array,
    value: f64,
    abs_tol: f64,
    rel_tol: f64,
    nans_equal: bool,
) -> BooleanArray
where
    T: PolarsNumericType,
{
    let validity = arr.validity().cloned();
    let element_iter = arr
        .values_iter()
        .map(|x| is_close_scalar(x.as_(), value, abs_tol, rel_tol, nans_equal));
    let result: BooleanArray = element_iter.collect_arr();
    result.with_validity_typed(validity)
}

/* ---------------------------------------- SCALAR LOGIC --------------------------------------- */

#[inline(always)]
fn is_close_scalar(x: f64, y: f64, abs_tol: f64, rel_tol: f64, nans_equal: bool) -> bool {
    // The logic in this function is taken from https://peps.python.org/pep-0485/.
    let cmp = (x - y).abs()
        <= max_by(
            rel_tol * max_by(x.abs(), y.abs(), f64::total_cmp),
            abs_tol,
            f64::total_cmp,
        );
    (x.is_finite() && y.is_finite() && cmp)
        || (x.is_nan() && y.is_nan() && nans_equal)
        || (x.is_infinite() && y.is_infinite() && x.signum() == y.signum())
}
