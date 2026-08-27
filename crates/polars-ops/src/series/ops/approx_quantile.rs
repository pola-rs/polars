use std::fmt;

use polars_compute::approx_quantile::{ApproxQuantileMethod, Sketch};
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::total_ord::TotalOrd;

fn sketch_quantile<T: fmt::Debug + Clone + TotalOrd>(
    values: impl Iterator<Item = Option<T>>,
    quantile: f64,
    error: f64,
    method: &ApproxQuantileMethod,
) -> Option<T> {
    let mut sketch = Sketch::new(method, error);
    for value in values.flatten() {
        sketch.update(&[value]);
    }
    sketch.finalize();
    sketch.estimate_quantile(quantile).cloned()
}

pub fn approx_quantile(
    s: &Column,
    quantile: &Series,
    error: f64,
    method: &ApproxQuantileMethod,
) -> PolarsResult<Scalar> {
    let quantile_ca: &Float64Chunked = quantile.as_ref().as_ref();
    let q = quantile_ca.no_null_iter().next().unwrap();
    polars_ensure!(
        (0.0..=1.0).contains(&q),
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
                let v = sketch_quantile(ca.iter(), q, error, method);
                ChunkedArray::<$T>::from_iter_options(PlSmallStr::EMPTY, std::iter::once(v))
                    .into_series()
            })
        },
        DataType::Boolean => {
            let v = sketch_quantile(s.bool()?.iter(), q, error, method);
            BooleanChunked::from_iter_options(PlSmallStr::EMPTY, std::iter::once(v)).into_series()
        },
        DataType::String => {
            let v = sketch_quantile(
                s.str()?.iter().map(|v| v.map(str::to_owned)),
                q,
                error,
                method,
            );
            StringChunked::from_iter_options(PlSmallStr::EMPTY, std::iter::once(v)).into_series()
        },
        _ => {
            polars_bail!(InvalidOperation: "`approx_quantile` operation not supported for dtype `{dtype}`")
        },
    };

    // SAFETY: `out` holds an item taken from `s` itself.
    let out = unsafe { out.from_physical_unchecked(dtype)? };
    Ok(Scalar::new(dtype.clone(), out.get(0)?.into_static()))
}
