use polars_compute::req;
use polars_core::prelude::*;

pub fn approx_quantile(s: &Column, quantile: &Series, error: f64) -> PolarsResult<Scalar> {
    let s = s.as_materialized_series();
    let ca: &Float64Chunked = s.as_ref().as_ref();
    let mut sketch = req::ReqSketchBounded::new(error, 1000_000);
    for item in ca.iter() {
        if let Some(item) = item {
            sketch.update(&[item]);
        }
    }
    sketch.finalize();
    let quantile_ca: &Float64Chunked = quantile.as_ref().as_ref();
    let q = quantile_ca.no_null_iter().next().unwrap();
    polars_ensure!(
        (0.0..=1.0).contains(&q),
        ComputeError: "`quantile` should be between 0.0 and 1.0",
    );
    let quantile_value = sketch.estimate_quantile(q);
    Ok(Scalar::new(
        DataType::Float64,
        AnyValue::Float64(*quantile_value),
    ))
}
