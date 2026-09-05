use num_traits::AsPrimitive;
use polars_compute::moment::{CovState, PearsonState};
use polars_core::prelude::*;
use polars_core::utils::align_chunks_binary;

/// Compute the covariance between two columns.
pub fn cov<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>, ddof: u8) -> Option<f64>
where
    T: PolarsNumericType,
    T::Native: AsPrimitive<f64>,
    ChunkedArray<T>: ChunkVar,
{
    if a.len() == 1 || b.len() == 1 {
        return Some(0.0); // (Broadcasted) constant -> zero covariance.
    }
    let (a, b) = align_chunks_binary(a, b);
    let mut out = CovState::default();
    for (a, b) in a.downcast_iter().zip(b.downcast_iter()) {
        out.combine(&polars_compute::moment::cov(a, b))
    }
    out.finalize(ddof)
}

/// Compute the two-variable moment state of two columns, pairing only rows
/// where both values are non-null. `x` is the independent and `y` the
/// dependent variable.
pub fn regression_state<T>(x: &ChunkedArray<T>, y: &ChunkedArray<T>) -> PearsonState
where
    T: PolarsNumericType,
    T::Native: AsPrimitive<f64>,
{
    let (x, y) = align_chunks_binary(x, y);
    let mut out = PearsonState::default();
    for (x, y) in x.downcast_iter().zip(y.downcast_iter()) {
        out.combine(&polars_compute::moment::pearson_corr(x, y))
    }
    out
}

/// Compute the pearson correlation between two columns.
pub fn pearson_corr<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> Option<f64>
where
    T: PolarsNumericType,
    T::Native: AsPrimitive<f64>,
    ChunkedArray<T>: ChunkVar,
{
    if a.len() == 1 || b.len() == 1 {
        return Some(f64::NAN); // (Broadcasted) constant -> NaN correlation.
    }
    let (a, b) = align_chunks_binary(a, b);
    let mut out = PearsonState::default();
    for (a, b) in a.downcast_iter().zip(b.downcast_iter()) {
        out.combine(&polars_compute::moment::pearson_corr(a, b))
    }
    Some(out.finalize())
}
