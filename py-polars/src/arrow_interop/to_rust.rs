use crate::error::PyPolarsEr;
use crate::prelude::ArrowDataType;
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::utils::arrow::{array::ArrayRef, ffi};
use polars_core::utils::arrow::{array::PrimitiveArray, bitmap::MutableBitmap};
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

    unsafe {
        let field = ffi::import_field_from_c(schema.as_ref()).map_err(PyPolarsEr::from)?;
        let array = if field.data_type == ArrowDataType::Null {
            let mut validity = MutableBitmap::new();
            validity.extend_constant(1, false);
            Box::new(PrimitiveArray::from_data(
                ArrowDataType::Float64,
                vec![1.0f64].into(),
                Some(validity.into()),
            ))
        } else {
            ffi::import_array_from_c(array, &field).map_err(PyPolarsEr::from)?
        };
        Ok(array.into())
    }
}

pub fn to_rust_df(rb: &[&PyAny]) -> PyResult<DataFrame> {
    let schema = rb
        .get(0)
        .ok_or_else(|| PyPolarsEr::Other("empty table".into()))?
        .getattr("schema")?;
    let names = schema.getattr("names")?.extract::<Vec<String>>()?;

    let dfs = rb
        .iter()
        .map(|rb| {
            let columns = (0..names.len())
                .map(|i| {
                    let array = rb.call_method1("column", (i,))?;
                    let arr = array_to_rust(array)?;
                    let s = Series::try_from((names[i].as_str(), arr)).map_err(PyPolarsEr::from)?;
                    Ok(s)
                })
                .collect::<PyResult<_>>()?;
            Ok(DataFrame::new(columns).map_err(PyPolarsEr::from)?)
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(accumulate_dataframes_vertical(dfs).map_err(PyPolarsEr::from)?)
}
