use polars_core::error::{PolarsResult, polars_err};
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_ffi::version_0::SeriesExport;
use polars_utils::python_convert_registry::get_python_convert_registry;
use pyo3::intern;
use pyo3::prelude::*;

/// Convert a Python Polars DataFrame into a Rust DataFrame.
///
/// # Safety
/// `df._df` must be a genuine Polars `PyDataFrame` whose `_export_columns` method
/// initializes `width()` valid `SeriesExport` values at the provided address.
pub unsafe fn python_df_to_rust(py: Python, df: Bound<PyAny>) -> PolarsResult<DataFrame> {
    let err = |_| polars_err!(ComputeError: "expected a polars.DataFrame; got {}", df);
    let pydf = df.getattr(intern!(py, "_df")).map_err(err)?;

    // Try to convert without going through FFI first.
    let converted = get_python_convert_registry().from_py.df;
    if let Ok(any_df) = converted(pydf.clone().unbind()) {
        return Ok(*any_df.downcast::<DataFrame>().unwrap());
    }

    // Might be foreign Polars, try with FFI.
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

pub(crate) unsafe fn python_schema_to_rust(
    py: Python,
    schema: Bound<PyAny>,
) -> PolarsResult<SchemaRef> {
    let err = |_| polars_err!(ComputeError: "expected a polars.Schema; got {}", schema);
    let df = schema.call_method0("to_frame").map_err(err)?;
    // SAFETY: The schema's `to_frame` method returns a genuine Polars DataFrame.
    unsafe { python_df_to_rust(py, df) }.map(|df| df.schema().clone())
}
