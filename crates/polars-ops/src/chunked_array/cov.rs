use std::ops::Add;

use arrow::compute;
use arrow::types::simd::Simd;
use num_traits::ToPrimitive;
use polars_core::prelude::*;
use polars_core::utils::align_chunks_binary;

/// Compute the covariance between two columns.
pub fn cov<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>, ddof: u8) -> Option<f64>
where
    T: PolarsNumericType,
    T::Native: ToPrimitive,
{
    if a.len() != b.len() {
        None
    } else {
        let (a, b) = align_chunks_binary(a, b);

        let out = if a.null_count() > 0 || b.null_count() > 0 {
            let iters = a
                .downcast_iter()
                .zip(b.downcast_iter())
                .map(|(a, b)| {
                    a.into_iter().zip(b).filter_map(|(a, b)| match (a, b) {
                        (Some(a), Some(b)) => Some((*a, *b)),
                        _ => None,
                    })
                })
                .collect::<Vec<_>>();
            online_cov(iters.as_slice(), ddof)
        } else {
            let iters = a
                .downcast_iter()
                .zip(b.downcast_iter())
                .map(|(a, b)| a.values_iter().copied().zip(b.values_iter().copied()))
                .collect::<Vec<_>>();
            online_cov(iters.as_slice(), ddof)
        };
        Some(out)
    }
}

/// # Arguments
/// `iter` - Iterator over `T` tuple where any `Option<T>` would skip the tuple.
fn online_cov<I, T>(iters: &[I], ddof: u8) -> f64
where
    I: IntoIterator<Item = (T, T)> + Clone,
    T: ToPrimitive,
{
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut c = 0.0;
    let mut n = 0.0;

    for iter in iters {
        let iter = iter.clone().into_iter();
        for (x, y) in iter {
            let x = x.to_f64().unwrap();
            let y = y.to_f64().unwrap();

            n += 1.0;

            let dx = x - mean_x;
            mean_x += dx / n;
            mean_y += (y - mean_y) / n;
            c += dx * (y - mean_y)
        }
    }

    c / (n - ddof as f64)
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
    let (a, b) = align_chunks_binary(a, b);

    let out = if a.null_count() > 0 || b.null_count() > 0 {
        let iters = a
            .downcast_iter()
            .zip(b.downcast_iter())
            .map(|(a, b)| {
                a.into_iter().zip(b).filter_map(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => Some((*a, *b)),
                    _ => None,
                })
            })
            .collect::<Vec<_>>();
        online_pearson_corr(iters.as_slice(), ddof)
    } else {
        let iters = a
            .downcast_iter()
            .zip(b.downcast_iter())
            .map(|(a, b)| a.values_iter().copied().zip(b.values_iter().copied()))
            .collect::<Vec<_>>();
        online_pearson_corr(iters.as_slice(), ddof)
    };
    Some(out)
}

/// # Arguments
/// `iter` - Iterator over `T` tuple where any `Option<T>` would skip the tuple.
fn online_pearson_corr<I, T>(iters: &[I], ddof: u8) -> f64
where
    I: IntoIterator<Item = (T, T)> + Clone,
    T: ToPrimitive,
{
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut c = 0.0;
    let mut n = 0.0;

    let mut m2x = 0.0;
    let mut m2y = 0.0;

    for iter in iters {
        let iter = iter.clone().into_iter();
        for (x, y) in iter {
            let x = x.to_f64().unwrap();
            let y = y.to_f64().unwrap();

            n += 1.0;

            let dx = x - mean_x;
            let dy = y - mean_y;
            mean_x += dx / n;
            mean_y += dy / n;

            let d2x = x - mean_x;
            let d2y = y - mean_y;

            m2x += dx * d2x;
            m2y += dy * d2y;
            c += dx * (y - mean_y)
        }
    }

    let sample_n = n - ddof as f64;
    let sample_cov = c / sample_n;
    let sample_std_x = (m2x / sample_n).sqrt();
    let sample_std_y = (m2y / sample_n).sqrt();

    sample_cov / (sample_std_x * sample_std_y)
}
