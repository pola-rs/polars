use polars::export::arrow;
use polars::prelude::*;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;

use super::PySeries;
use crate::error::PyPolarsErr;

// Import arrow data directly without requiring pyarrow (used in pyo3-polars)
#[pymethods]
impl PySeries {
    #[staticmethod]
    unsafe fn _import_arrow_from_c(
        name: &str,
        chunks: Vec<(Py_uintptr_t, Py_uintptr_t)>,
    ) -> PyResult<Self> {
        let chunks = chunks
            .into_iter()
            .map(|(schema_ptr, array_ptr)| {
                let schema_ptr = schema_ptr as *mut arrow::ffi::ArrowSchema;
                let array_ptr = array_ptr as *mut arrow::ffi::ArrowArray;

                // Don't take the box from raw as the other process must deallocate that memory.
                let array = std::ptr::read_unaligned(array_ptr);
                let schema = &*schema_ptr;

                let field = arrow::ffi::import_field_from_c(schema).unwrap();
                arrow::ffi::import_array_from_c(array, field.data_type).unwrap()
            })
            .collect::<Vec<_>>();

        let s = Series::try_from((name, chunks)).map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    unsafe fn _export_arrow_to_c(
        &self,
        out_ptr: Py_uintptr_t,
        out_schema_ptr: Py_uintptr_t,
    ) -> PyResult<()> {
        export_chunk(&self.series, out_ptr, out_schema_ptr).map_err(PyPolarsErr::from)?;
        Ok(())
    }
}

unsafe fn export_chunk(
    s: &Series,
    out_ptr: Py_uintptr_t,
    out_schema_ptr: Py_uintptr_t,
) -> PolarsResult<()> {
    polars_ensure!(s.chunks().len() == 1, InvalidOperation: "expect a single chunk");

    let c_array = arrow::ffi::export_array_to_c(s.chunks()[0].clone());
    let out_ptr = out_ptr as *mut arrow::ffi::ArrowArray;
    *out_ptr = c_array;

    let field = ArrowField::new(s.name(), s.dtype().to_arrow(CompatLevel::newest()), true);
    let c_schema = arrow::ffi::export_field_to_c(&field);

    let out_schema_ptr = out_schema_ptr as *mut arrow::ffi::ArrowSchema;
    *out_schema_ptr = c_schema;
    Ok(())
}
