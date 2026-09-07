use std::fmt;

use polars_compute::approx_quantile::{ApproxQuantileMethod, Sketch};
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::utils::_split_offsets;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::total_ord::TotalOrd;
use rayon::prelude::*;

/// Estimate every quantile in `quantiles` from a single sketch over the input.
#[inline(never)]
fn sketch_quantile<T>(
    len: usize,
    quantiles: &Float64Chunked,
    error: f64,
    method: &ApproxQuantileMethod,
    fill: impl Fn(usize, usize, &mut Sketch<T>) + Sync,
) -> Vec<Option<T>>
where
    T: fmt::Debug + Clone + TotalOrd + Send,
{
    const THREAD_BOUNDARY: usize = if cfg!(debug_assertions) { 0 } else { 100_000 };

    let build = |offset, len| {
        let mut sketch = Sketch::new(method, error);
        fill(offset, len, &mut sketch);
        sketch
    };

    let mut sketch = if len < THREAD_BOUNDARY
        || RAYON.current_num_threads() == 1
        || RAYON.current_thread_has_pending_tasks().unwrap_or(false)
    {
        build(0, len)
    } else {
        let splits = _split_offsets(len, RAYON.current_num_threads());
        RAYON
            .install(|| {
                splits
                    .into_par_iter()
                    .map(|(offset, len)| build(offset, len))
                    .reduce_with(|mut acc, sketch| {
                        acc.merge(sketch);
                        acc
                    })
            })
            .unwrap()
    };

    sketch.finalize();
    quantiles
        .iter()
        .map(|q| q.and_then(|q| sketch.estimate_quantile(q).cloned()))
        .collect()
}

/// Estimate the quantiles of `s`, one output element per requested quantile.
pub fn approx_quantile(
    s: &Column,
    quantiles: &Series,
    error: f64,
    method: &ApproxQuantileMethod,
) -> PolarsResult<Series> {
    let quantiles = quantiles.strict_cast(&DataType::Float64)?;
    let quantiles: &Float64Chunked = quantiles.f64()?;
    polars_ensure!(
        quantiles.null_count() == 0,
        ComputeError: "`quantile` should not contain null values",
    );
    polars_ensure!(
        quantiles.iter().flatten().all(|q| (0.0..=1.0).contains(&q)),
        ComputeError: "`quantile` should be between 0.0 and 1.0",
    );

    let s = s.as_materialized_series();
    let dtype = s.dtype();

    let out = match dtype {
        _ if dtype.is_primitive_numeric() || dtype.is_temporal() || dtype.is_decimal() => {
            let physical = s.to_physical_repr();
            let physical: &Series = physical.as_ref();
            with_match_physical_numeric_polars_type!(physical.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = physical.as_ref().as_ref();
                let v = sketch_quantile(ca.len(), quantiles, error, method, |offset, len, sketch| {
                    for value in ca.slice(offset as i64, len).iter().flatten() {
                        sketch.update(&value);
                    }
                });
                ChunkedArray::<$T>::from_iter_options(PlSmallStr::EMPTY, v.into_iter())
                    .into_series()
            })
        },
        DataType::Boolean => {
            let ca = s.bool()?;
            let v = sketch_quantile(ca.len(), quantiles, error, method, |offset, len, sketch| {
                for value in ca.slice(offset as i64, len).iter().flatten() {
                    sketch.update(&value);
                }
            });
            BooleanChunked::from_iter_options(PlSmallStr::EMPTY, v.into_iter()).into_series()
        },
        DataType::String => {
            let ca = s.str()?;
            let v = sketch_quantile(ca.len(), quantiles, error, method, |offset, len, sketch| {
                for value in ca.slice(offset as i64, len).iter().flatten() {
                    sketch.update(&value.to_owned());
                }
            });
            StringChunked::from_iter_options(PlSmallStr::EMPTY, v.into_iter()).into_series()
        },
        _ => {
            polars_bail!(InvalidOperation: "`approx_quantile` operation not supported for dtype `{dtype}`")
        },
    };

    // SAFETY: `out` holds items taken from `s` itself.
    unsafe { out.from_physical_unchecked(dtype) }
}
