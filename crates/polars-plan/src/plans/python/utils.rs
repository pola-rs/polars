use polars_core::error::{polars_err, PolarsResult};
use polars_core::frame::DataFrame;
use polars_core::prelude::Column;
use polars_core::schema::SchemaRef;
use pyo3::intern;
use pyo3::prelude::*;

pub fn python_df_to_rust(py: Python, df: Bound<PyAny>) -> PolarsResult<DataFrame> {
    let err = |_| polars_err!(ComputeError: "expected a polars.DataFrame; got {}", df);
    let pydf = df.getattr(intern!(py, "_df")).map_err(err)?;
    let raw_parts = pydf
        .call_method0(intern!(py, "into_raw_parts"))
        .map_err(err)?;
    let raw_parts = raw_parts.extract::<(usize, usize, usize)>().unwrap();

    let (ptr, len, cap) = raw_parts;
    unsafe {
        Ok(DataFrame::new_no_checks_height_from_first(
            Vec::from_raw_parts(ptr as *mut Column, len, cap),
        ))
    }
}

pub fn python_schema_to_rust(py: Python, schema: Bound<PyAny>) -> PolarsResult<SchemaRef> {
    let err = |_| polars_err!(ComputeError: "expected a polars.Schema; got {}", schema);
    let df = schema.call_method0("to_frame").map_err(err)?;
    python_df_to_rust(py, df).map(|df| df.schema().clone())
}
