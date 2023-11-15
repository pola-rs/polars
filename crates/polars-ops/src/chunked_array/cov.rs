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
            let iters = a.downcast_iter().zip(b.downcast_iter()).map(|(a, b)| {
                a.into_iter().zip(b).filter_map(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => Some((*a, *b)),
                    _ => None,
                })
            });
            online_cov(iters, ddof)
        } else {
            let iters = a
                .downcast_iter()
                .zip(b.downcast_iter())
                .map(|(a, b)| a.values_iter().copied().zip(b.values_iter().copied()));
            online_cov(iters, ddof)
        };
        Some(out)
    }
}

/// # Arguments
/// `iter` - Iterator over `T` tuple where any `Option<T>` would skip the tuple.
fn online_cov<I, J, T>(iters: I, ddof: u8) -> f64
where
    I: Iterator<Item = J>,
    J: IntoIterator<Item = (T, T)> + Clone,
    T: ToPrimitive,
{
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut c = 0.0;
    let mut n = 0.0;

    const BUF_SIZE: usize = 128;

    let mut x_vec = [0.0; BUF_SIZE];
    let mut c_vec = [0.0; BUF_SIZE];
    let mut y_vec = [0.0; BUF_SIZE];

    for iter in iters {
        let mut iter = iter.clone().into_iter();

        let mut offset = BUF_SIZE;
        loop {
            for i in 0..BUF_SIZE {
                if let Some((x, y)) = iter.next() {
                    let x = x.to_f64().unwrap();
                    let y = y.to_f64().unwrap();

                    x_vec[i] = x;
                    y_vec[i] = y;
                } else {
                    offset = i;
                    break;
                }
            }
            n += offset as f64;
            mean_x += sum_f64::sum(&x_vec[..offset]) / n;
            mean_y += sum_f64::sum(&y_vec[..offset]) / n;

            // Update x_vec to hold c values;
            for i in 0..BUF_SIZE {
                c_vec[i] = (x_vec[i] - mean_x) * (y_vec[i] - mean_y)
            }
            c += sum_f64::sum(&c_vec[..offset]);
            if offset == 0 {
                break;
            }
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
        let iters = a.downcast_iter().zip(b.downcast_iter()).map(|(a, b)| {
            a.into_iter().zip(b).filter_map(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => Some((*a, *b)),
                _ => None,
            })
        });
        online_pearson_corr(iters, ddof)
    } else {
        let iters = a
            .downcast_iter()
            .zip(b.downcast_iter())
            .map(|(a, b)| a.values_iter().copied().zip(b.values_iter().copied()));
        online_pearson_corr(iters, ddof)
    };
    Some(out)
}

/// # Arguments
/// `iter` - Iterator over `T` tuple where any `Option<T>` would skip the tuple.
fn online_pearson_corr<I, J, T>(iters: I, ddof: u8) -> f64
where
    I: Iterator<Item = J>,
    J: IntoIterator<Item = (T, T)> + Clone,
    T: ToPrimitive,
{
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut c = 0.0;
    let mut n = 0.0;

    let mut m2x = 0.0;
    let mut m2y = 0.0;

    const BUF_SIZE: usize = 128;

    let mut x_vec = [0.0; BUF_SIZE];
    let mut y_vec = [0.0; BUF_SIZE];
    let mut c_vec = [0.0; BUF_SIZE];

    for iter in iters {
        let mut iter = iter.clone().into_iter();

        let mut offset = BUF_SIZE;
        loop {
            for i in 0..BUF_SIZE {
                if let Some((x, y)) = iter.next() {
                    let x = x.to_f64().unwrap();
                    let y = y.to_f64().unwrap();

                    x_vec[i] = x;
                    y_vec[i] = y;
                } else {
                    offset = i;
                    break;
                }
            }
            n += offset as f64;
            let mean_x_old = mean_x;
            let mean_y_old = mean_y;
            mean_x += sum_f64::sum(&x_vec[..offset]) / n;
            mean_y += sum_f64::sum(&y_vec[..offset]) / n;

            for i in 0..BUF_SIZE {
                let dx_new = x_vec[i] - mean_x;
                let dy_new = y_vec[i] - mean_y;
                c_vec[i] = dx_new * dy_new;
                x_vec[i] = (x_vec[i] - mean_x_old) * dx_new;
                y_vec[i] = (y_vec[i] - mean_y_old) * dy_new;
            }
            c += sum_f64::sum(&c_vec[..offset]);
            m2x += sum_f64::sum(&x_vec[..offset]);
            m2y += sum_f64::sum(&y_vec[..offset]);

            if offset == 0 {
                break;
            }
        }
    }

    let sample_n = n - ddof as f64;
    let sample_cov = c / sample_n;
    let sample_std_x = (m2x / sample_n).sqrt();
    let sample_std_y = (m2y / sample_n).sqrt();

    sample_cov / (sample_std_x * sample_std_y)
}
