use crate::error::PyPolarsErr;
use polars_core::export::rayon::prelude::*;
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::utils::arrow::{array::ArrayRef, ffi};
use polars_core::POOL;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::PyList;

pub fn field_to_rust(obj: &PyAny) -> PyResult<Field> {
    let schema = Box::new(ffi::ArrowSchema::empty());
    let schema_ptr = &*schema as *const ffi::ArrowSchema;

    // make the conversion through PyArrow's private API
    obj.call_method1("_export_to_c", (schema_ptr as Py_uintptr_t,))?;
    let field = unsafe { ffi::import_field_from_c(schema.as_ref()).map_err(PyPolarsErr::from)? };
    Ok(Field::from(&field))
}

// PyList<Field> which you get by calling `list(schema)`
pub fn pyarrow_schema_to_rust(obj: &PyList) -> PyResult<Schema> {
    obj.into_iter().map(|fld| field_to_rust(fld)).collect()
}

pub fn array_to_rust(obj: &PyAny) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let array = Box::new(ffi::ArrowArray::empty());
    let schema = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = &*array as *const ffi::ArrowArray;
    let schema_ptr = &*schema as *const ffi::ArrowSchema;

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular, `_export_to_c` can go out of bounds
    obj.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        let field = ffi::import_field_from_c(schema.as_ref()).map_err(PyPolarsErr::from)?;
        let array = ffi::import_array_from_c(array, field.data_type).map_err(PyPolarsErr::from)?;
        Ok(array.into())
    }
}

pub fn to_rust_df(rb: &[&PyAny]) -> PyResult<DataFrame> {
    let schema = rb
        .get(0)
        .ok_or_else(|| PyPolarsErr::Other("empty table".into()))?
        .getattr("schema")?;
    let names = schema.getattr("names")?.extract::<Vec<String>>()?;

    let dfs = rb
        .iter()
        .map(|rb| {
            let mut run_parallel = false;

            let columns = (0..names.len())
                .map(|i| {
                    let array = rb.call_method1("column", (i,))?;
                    let arr = array_to_rust(array)?;
                    run_parallel |= matches!(
                        arr.data_type(),
                        ArrowDataType::Utf8 | ArrowDataType::Dictionary(_, _, _)
                    );
                    Ok(arr)
                })
                .collect::<PyResult<Vec<_>>>()?;

            // we parallelize this part because we can have dtypes that are not zero copy
            // for instance utf8 -> large-utf8
            // dict encoded to categorical
            let columns = if run_parallel {
                POOL.install(|| {
                    columns
                        .into_par_iter()
                        .enumerate()
                        .map(|(i, arr)| {
                            let s = Series::try_from((names[i].as_str(), arr))
                                .map_err(PyPolarsErr::from)?;
                            Ok(s)
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            } else {
                columns
                    .into_iter()
                    .enumerate()
                    .map(|(i, arr)| {
                        let s = Series::try_from((names[i].as_str(), arr))
                            .map_err(PyPolarsErr::from)?;
                        Ok(s)
                    })
                    .collect::<PyResult<Vec<_>>>()
            }?;

            Ok(DataFrame::new(columns).map_err(PyPolarsErr::from)?)
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(accumulate_dataframes_vertical(dfs).map_err(PyPolarsErr::from)?)
}
