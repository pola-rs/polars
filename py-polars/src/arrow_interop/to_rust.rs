use crate::error::PyPolarsEr;
use polars_core::prelude::Arc;
use polars_core::utils::arrow::{
    array::ArrayRef,
    datatypes::{Field, Schema},
    ffi,
    record_batch::RecordBatch,
};
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;

pub fn array_to_rust(obj: &PyAny) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let array = Box::new(ffi::Ffi_ArrowArray::empty());
    let schema = Box::new(ffi::Ffi_ArrowSchema::empty());

    let array_ptr = &*array as *const ffi::Ffi_ArrowArray;
    let schema_ptr = &*schema as *const ffi::Ffi_ArrowSchema;

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular, `_export_to_c` can go out of bounds
    obj.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    let field = ffi::import_field_from_c(schema.as_ref()).map_err(PyPolarsEr::from)?;
    let array = ffi::import_array_from_c(array, &field).map_err(PyPolarsEr::from)?;
    Ok(array.into())
}

pub fn to_rust_rb(rb: &[&PyAny]) -> PyResult<Vec<RecordBatch>> {
    let schema = rb
        .get(0)
        .ok_or_else(|| PyPolarsEr::Other("empty table".into()))?
        .getattr("schema")?;
    let names = schema.getattr("names")?.extract::<Vec<String>>()?;

    let arrays = rb
        .iter()
        .map(|rb| {
            let columns = (0..names.len())
                .map(|i| {
                    let array = rb.call_method1("column", (i,))?;
                    array_to_rust(array)
                })
                .collect::<PyResult<_>>()?;
            Ok(columns)
        })
        .collect::<PyResult<Vec<Vec<_>>>>()?;

    let fields = arrays[0]
        .iter()
        .zip(&names)
        .map(|(arr, name)| {
            let dtype = arr.data_type().clone();
            Field::new(name, dtype, true)
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));

    Ok(arrays
        .into_iter()
        .map(|columns| RecordBatch::try_new(schema.clone(), columns).unwrap())
        .collect())
}
