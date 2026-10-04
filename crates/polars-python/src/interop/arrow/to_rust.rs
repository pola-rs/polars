use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_core::utils::polars_arrow::ffi;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::error::PyPolarsErr;
use crate::utils::EnterPolarsExt;

pub fn field_to_rust_arrow(obj: Bound<'_, PyAny>) -> PyResult<ArrowField> {
    let mut schema = Box::new(ffi::ArrowSchema::empty());
    let schema_ptr = schema.as_mut() as *mut ffi::ArrowSchema;

    // make the conversion through PyArrow's private API
    obj.call_method1("_export_to_c", (schema_ptr as Py_uintptr_t,))?;
    let field = unsafe { ffi::import_field_from_c(schema.as_ref()).map_err(PyPolarsErr::from)? };
    Ok(field)
}

pub fn field_to_rust(obj: Bound<'_, PyAny>) -> PyResult<Field> {
    field_to_rust_arrow(obj).map(|f| (&f).into())
}

// PyList<Field> which you get by calling `list(schema)`
pub fn pyarrow_schema_to_rust(obj: &Bound<'_, PyList>) -> PyResult<Schema> {
    obj.into_iter().map(field_to_rust).collect()
}

pub fn array_to_rust(obj: &Bound<PyAny>) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let mut array = Box::new(ffi::ArrowArray::empty());
    let mut schema = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = array.as_mut() as *mut ffi::ArrowArray;
    let schema_ptr = schema.as_mut() as *mut ffi::ArrowSchema;

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular, `_export_to_c` can go out of bounds
    obj.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        let field = ffi::import_field_from_c(schema.as_ref()).map_err(PyPolarsErr::from)?;
        let array = ffi::import_array_from_c(*array, field.dtype).map_err(PyPolarsErr::from)?;
        Ok(array)
    }
}

pub fn to_rust_df(
    py: Python<'_>,
    rb: &[Bound<PyAny>],
    schema: Bound<PyAny>,
) -> PyResult<DataFrame> {
    let ArrowDataType::Struct(fields) = field_to_rust_arrow(schema)?.dtype else {
        return Err(PyPolarsErr::Other("invalid top-level schema".into()).into());
    };

    let schema = ArrowSchema::from_iter(fields.iter().cloned());

    // Verify that field names are not duplicated. Arrow permits duplicate field names, we do not.
    // Required to uphold safety invariants for unsafe block below.
    if schema.len() != fields.len() {
        let mut field_map: PlHashMap<PlSmallStr, u64> = PlHashMap::with_capacity(fields.len());
        fields.iter().for_each(|field| {
            field_map
                .entry(field.name.clone())
                .and_modify(|c| {
                    *c += 1;
                })
                .or_insert(1);
        });
        let duplicate_fields: Vec<_> = field_map
            .into_iter()
            .filter_map(|(k, v)| (v > 1).then_some(k))
            .collect();

        return Err(PyPolarsErr::Polars(PolarsError::Duplicate(
            format!("column appears more than once; names must be unique: {duplicate_fields:?}")
                .into(),
        ))
        .into());
    }

    if rb.is_empty() {
        let columns = schema
            .iter_values()
            .map(|field| {
                let field = Field::from(field);
                Series::new_empty(field.name, &field.dtype).into_column()
            })
            .collect::<Vec<_>>();

        // no need to check as a record batch has the same guarantees
        return Ok(unsafe { DataFrame::new_unchecked_infer_height(columns) });
    }

    // Importing the arrays over FFI needs the GIL, but it is zero-copy and cheap; do it
    // for every batch up front so the (potentially expensive) conversion below can run
    // with the GIL released.
    let mut run_parallel = false;
    let mut batches: Vec<Vec<ArrayRef>> = Vec::with_capacity(rb.len());
    for rb in rb.iter() {
        let arrays = (0..schema.len())
            .map(|i| {
                let array = rb.call_method1("column", (i,))?;
                let mut arr = array_to_rust(&array)?;

                // Only the schema contains extension type info, restore.
                // TODO: nested?
                let dtype = schema.get_at_index(i).unwrap().1.dtype();
                if let ArrowDataType::Extension(ext) = dtype {
                    if *arr.dtype() == ext.inner {
                        *arr.dtype_mut() = dtype.clone();
                    }
                }

                run_parallel |= !is_zero_copy_to_polars(arr.dtype());
                Ok(arr)
            })
            .collect::<PyResult<Vec<_>>>()?;
        batches.push(arrays);
    }

    let column = |i: usize, arr: ArrayRef| -> PolarsResult<Column> {
        let (_, field) = schema.get_at_index(i).unwrap();
        let s = unsafe {
            Series::_try_from_arrow_unchecked_with_md(
                field.name.clone(),
                vec![arr],
                field.dtype(),
                field.metadata.as_deref(),
            )
        }?;
        Ok(s.into_column())
    };

    // Every dtype here is zero copy, so the conversion is just shuffling pointers around.
    // Releasing the GIL for that would cost more than it saves.
    if !run_parallel {
        let dfs = batches
            .into_iter()
            .map(|arrays| {
                let columns = arrays
                    .into_iter()
                    .enumerate()
                    .map(|(i, arr)| column(i, arr))
                    .collect::<PolarsResult<Vec<_>>>()?;
                // no need to check as a record batch has the same guarantees
                Ok(unsafe { DataFrame::new_unchecked_infer_height(columns) })
            })
            .collect::<PolarsResult<Vec<_>>>()
            .map_err(PyPolarsErr::from)?;

        return Ok(accumulate_dataframes_vertical_unchecked(dfs));
    }

    // Otherwise the conversion is not zero copy - for instance string -> view, binary ->
    // binview, dict encoded -> categorical, and nested types get rebuilt - so it must not
    // hold the GIL, otherwise concurrent Python threads serialize on this call.
    py.enter_polars(move || {
        let dfs = RAYON.install(|| {
            batches
                .into_par_iter()
                .map(|arrays| {
                    let columns = arrays
                        .into_par_iter()
                        .enumerate()
                        .map(|(i, arr)| column(i, arr))
                        .collect::<PolarsResult<Vec<_>>>()?;
                    // no need to check as a record batch has the same guarantees
                    Ok(unsafe { DataFrame::new_unchecked_infer_height(columns) })
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        PolarsResult::Ok(accumulate_dataframes_vertical_unchecked(dfs))
    })
}

/// Whether `Series::_try_from_arrow_unchecked_with_md` is pure pointer shuffling for this
/// dtype. Conservative: anything not listed here is assumed to do real work.
fn is_zero_copy_to_polars(dtype: &ArrowDataType) -> bool {
    use ArrowDataType as D;
    matches!(
        dtype,
        D::Null
            | D::Boolean
            | D::Utf8View
            | D::BinaryView
            | D::Int8
            | D::Int16
            | D::Int32
            | D::Int64
            | D::UInt8
            | D::UInt16
            | D::UInt32
            | D::UInt64
            | D::Float32
            | D::Float64
    )
}
