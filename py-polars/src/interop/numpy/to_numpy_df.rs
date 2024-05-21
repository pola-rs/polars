use ndarray::IntoDimension;
use numpy::npyffi::flags;
use numpy::{Element, IntoPyArray};
use polars_core::prelude::*;
use polars_core::utils::dtypes_to_supertype;
use polars_core::with_match_physical_numeric_polars_type;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::to_numpy_series::series_to_numpy;
use super::utils::create_borrowed_np_array;
use crate::conversion::Wrap;
use crate::dataframe::PyDataFrame;

#[pymethods]
impl PyDataFrame {
    /// Convert this DataFrame to a NumPy ndarray.
    fn to_numpy(
        &self,
        py: Python,
        order: Wrap<IndexOrder>,
        writable: bool,
        allow_copy: bool,
    ) -> PyResult<PyObject> {
        df_to_numpy(py, &self.df, order.0, writable, allow_copy)
    }

    /// Create a view of the data as a NumPy ndarray.
    ///
    /// WARNING: The resulting view will show the underlying value for nulls,
    /// which may be any value. The caller is responsible for handling nulls
    /// appropriately.
    fn to_numpy_view(&self, py: Python) -> Option<PyObject> {
        try_df_to_numpy_view(py, &self.df)
    }
}

fn df_to_numpy(
    py: Python,
    df: &DataFrame,
    order: IndexOrder,
    writable: bool,
    allow_copy: bool,
) -> PyResult<PyObject> {
    // TODO: Use `is_empty` when fixed:
    // https://github.com/pola-rs/polars/pull/16351
    if df.height() == 0 {
        // Take this path to ensure a writable array.
        // This does not actually copy data for an empty DataFrame.
        return df_to_numpy_with_copy(py, df, order, true);
    }

    if matches!(order, IndexOrder::Fortran) {
        if let Some(mut arr) = try_df_to_numpy_view(py, df) {
            if writable {
                if !allow_copy {
                    return Err(PyRuntimeError::new_err(
                        "copy not allowed: cannot create a writable array without copying data",
                    ));
                }
                arr = arr.call_method0(py, intern!(py, "copy"))?;
            }
            return Ok(arr);
        }
    }

    if !allow_copy {
        return Err(PyRuntimeError::new_err(
            "copy not allowed: cannot convert to a NumPy array without copying data",
        ));
    }

    df_to_numpy_with_copy(py, df, order, writable)
}

fn try_df_to_numpy_view(py: Python, df: &DataFrame) -> Option<PyObject> {
    if df.is_empty() {
        return None;
    }
    let first = df.get_columns().first().unwrap().dtype();
    // TODO: Support Datetime/Duration/Array types
    if !first.is_numeric() {
        return None;
    }
    if !df
        .get_columns()
        .iter()
        .all(|s| s.null_count() == 0 && s.dtype() == first && s.chunks().len() == 1)
    {
        return None;
    }

    let owner = PyDataFrame::from(df.clone()).into_py(py); // Keep the DataFrame memory alive.

    with_match_physical_numeric_polars_type!(first, |$T| {
        get_ptr::<$T>(py, df.get_columns(), owner)
    })
}
fn get_ptr<T>(py: Python, columns: &[Series], owner: PyObject) -> Option<PyObject>
where
    T: PolarsNumericType,
    T::Native: Element,
{
    let slices = columns
        .iter()
        .map(|s| {
            let ca: &ChunkedArray<T> = s.unpack().unwrap();
            ca.cont_slice().unwrap()
        })
        .collect::<Vec<_>>();

    let first = slices.first().unwrap();
    unsafe {
        let mut end_ptr = first.as_ptr().add(first.len());
        // Check if all arrays are from the same buffer
        let all_contiguous = slices[1..].iter().all(|slice| {
            let valid = slice.as_ptr() == end_ptr;

            end_ptr = slice.as_ptr().add(slice.len());

            valid
        });

        if all_contiguous {
            let start_ptr = first.as_ptr();
            let dtype = T::Native::get_dtype_bound(py);
            let dims = [first.len(), columns.len()].into_dimension();
            Some(create_borrowed_np_array::<_>(
                py,
                dtype,
                dims,
                flags::NPY_ARRAY_FARRAY_RO,
                start_ptr as _,
                owner,
            ))
        } else {
            None
        }
    }
}

fn df_to_numpy_with_copy(
    py: Python,
    df: &DataFrame,
    order: IndexOrder,
    writable: bool,
) -> PyResult<PyObject> {
    if let Some(arr) = try_df_to_numpy_numeric_supertype(py, df, order) {
        Ok(arr)
    } else {
        df_columns_to_numpy(py, df, writable)
    }
}
fn try_df_to_numpy_numeric_supertype(
    py: Python,
    df: &DataFrame,
    order: IndexOrder,
) -> Option<PyObject> {
    let st = dtypes_to_supertype(df.iter().map(|s| s.dtype())).ok()?;

    let np_array = match st {
        dt if dt.is_numeric() => with_match_physical_numeric_polars_type!(dt, |$T| {
            df.to_ndarray::<$T>(order).ok()?.into_pyarray_bound(py).into_py(py)
        }),
        _ => return None,
    };
    Some(np_array)
}
fn df_columns_to_numpy(py: Python, df: &DataFrame, writable: bool) -> PyResult<PyObject> {
    let np_arrays = df
        .iter()
        .map(|s| series_to_numpy(py, s, writable, true).unwrap());

    // TODO: Handle multidimensional column arrays

    let numpy = PyModule::import_bound(py, "numpy")?;
    let arr = numpy
        .getattr(intern!(py, "vstack"))
        .unwrap()
        .call1((PyList::new_bound(py, np_arrays),))?
        .getattr(intern!(py, "T"))?;
    Ok(arr.into())
}
