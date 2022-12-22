use polars_core::export::num;
use DataType::*;

use super::*;

pub(super) fn sign(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        Float32 => {
            let ca = s.f32().unwrap();
            sign_float(ca)
        }
        Float64 => {
            let ca = s.f64().unwrap();
            sign_float(ca)
        }
        dt if dt.is_numeric() => {
            let s = s.cast(&Float64)?;
            sign(&s)
        }
        dt => Err(PolarsError::ComputeError(
            format!("cannot use 'sign' on Series of dtype: {dt:?}").into(),
        )),
    }
}

fn sign_float<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    ca.apply(signum_improved).into_series().cast(&Int64)
}

// Wrapper for the signum function that handles +/-0.0 inputs differently
// See discussion here: https://github.com/rust-lang/rust/issues/57543
fn signum_improved<F: num::Float>(v: F) -> F {
    if v.is_zero() {
        v
    } else {
        v.signum()
    }
}
