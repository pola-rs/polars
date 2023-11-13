use std::ops::Add;

use arrow::compute;
use arrow::types::simd::Simd;
use num_traits::ToPrimitive;
use polars_core::prelude::*;
use polars_core::utils::coalesce_nulls;

/// Compute the covariance between two columns.
pub fn cov<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> Option<f64>
where
    T: PolarsNumericType,
    T::Native: ToPrimitive,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
{
    if a.len() != b.len() {
        None
    } else {
        let a_mean = a.mean()?;
        let b_mean = b.mean()?;
        let a: Float64Chunked = a.apply_values_generic(|a| a.to_f64().unwrap() - a_mean);
        let b: Float64Chunked = b.apply_values_generic(|b| b.to_f64().unwrap() - b_mean);

        let tmp = a * b;
        let n = tmp.len() - tmp.null_count();
        Some(tmp.sum()? / (n - 1) as f64)
    }
}

/// Compute the pearson correlation between two columns.
pub fn pearson_corr<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>, ddof: u8) -> Option<f64>
where
    T: PolarsNumericType,
    T::Native: ToPrimitive,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
    ChunkedArray<T>: ChunkVar,
{
    let (a, b) = coalesce_nulls(a, b);
    let a = a.as_ref();
    let b = b.as_ref();

    Some(cov(a, b)? / (a.std(ddof)? * b.std(ddof)?))
}
