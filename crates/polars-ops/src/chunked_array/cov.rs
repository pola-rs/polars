use num_traits::AsPrimitive;
use polars_array::arrow::bridge::chunk_to_arrow;
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
        // TODO(polars-array-scalar): the kernel is an Arrow one, so a scalar chunk is written out
        // here rather than its one value being folded in with the weight of the whole chunk.
        out.combine(&polars_compute::moment::cov(
            &chunk_to_arrow(a),
            &chunk_to_arrow(b),
        ))
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
    if a.len() == 1 || b.len() == 1 {
        return Some(f64::NAN); // (Broadcasted) constant -> NaN correlation.
    }
    let (a, b) = align_chunks_binary(a, b);
    let mut out = PearsonState::default();
    for (a, b) in a.downcast_iter().zip(b.downcast_iter()) {
        // TODO(polars-array-scalar): as in `cov`, scalar chunks are written out here.
        out.combine(&polars_compute::moment::pearson_corr(
            &chunk_to_arrow(a),
            &chunk_to_arrow(b),
        ))
    }
    Some(out.finalize())
}
