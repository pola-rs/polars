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
    let (a, b) = align_chunks_binary(a, b);
    let mut out = CovState::default();
    for (a, b) in a.downcast_iter().zip(b.downcast_iter()) {
        out.combine(&polars_compute::moment::cov(a, b))
    }
    out.finalize(ddof)
}

/// Compute the pearson correlation between two columns.
pub fn pearson_corr<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> Option<f64>
where
    T: PolarsNumericType,
    T::Native: AsPrimitive<f64>,
    ChunkedArray<T>: ChunkVar,
{
    let (a, b) = align_chunks_binary(a, b);
    let mut out = PearsonState::default();
    for (a, b) in a.downcast_iter().zip(b.downcast_iter()) {
        out.combine(&polars_compute::moment::pearson_corr(a, b))
    }
    Some(out.finalize())
}
