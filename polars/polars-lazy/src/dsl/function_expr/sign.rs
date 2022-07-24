use super::*;
use polars_core::export::num;
use DataType::*;

pub(super) fn sign(s: &Series) -> Result<Series> {
    match s.dtype() {
        Float32 => {
            let ca = s.f32().unwrap();
            sign_float(ca)
        }
        Float64 => {
            let ca = s.f64().unwrap();
            sign_float(ca)
        }
        _ => {
            let s = s.cast(&Float64)?;
            sign(&s)
        }
    }
}

fn sign_float<T>(ca: &ChunkedArray<T>) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    ca.apply(sign_single_float).into_series().cast(&Int64)
}

fn sign_single_float<F: num::Float>(v: F) -> F {
    if v.is_zero() {
        v
    } else {
        v.signum()
    }
}
