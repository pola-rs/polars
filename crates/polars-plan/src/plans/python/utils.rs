use polars_core::error::{PolarsResult, polars_err};
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_ffi::version_0::SeriesExport;
use pyo3::intern;
use pyo3::prelude::*;

pub fn python_df_to_rust(py: Python, df: Bound<PyAny>) -> PolarsResult<DataFrame> {
    let err = |_| polars_err!(ComputeError: "expected a polars.DataFrame; got {}", df);
    let pydf = df.getattr(intern!(py, "_df")).map_err(err)?;

    let width = pydf.call_method0(intern!(py, "width")).unwrap();
    let width = width.extract::<usize>().unwrap();

    // Don't resize the Vec<> so that the drop of the SeriesExport will not be caleld.
    let mut export: Vec<SeriesExport> = Vec::with_capacity(width);
    let location = export.as_mut_ptr();

    let _ = pydf
        .call_method1(intern!(py, "_export_columns"), (location as usize,))
        .unwrap();

    unsafe { polars_ffi::version_0::import_df(location, width) }
}

pub(crate) fn python_schema_to_rust(py: Python, schema: Bound<PyAny>) -> PolarsResult<SchemaRef> {
    let err = |_| polars_err!(ComputeError: "expected a polars.Schema; got {}", schema);
    let df = schema.call_method0("to_frame").map_err(err)?;
    python_df_to_rust(py, df).map(|df| df.schema().clone())
}
