use polars::export::arrow;
use pyo3::ffi::Py_uintptr_t;

use super::*;

// Import arrow data directly without requiring pyarrow (used in pyo3-polars)
#[pymethods]
impl PySeries {
    #[staticmethod]
    unsafe fn _import_from_c(
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
}
