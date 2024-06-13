use ndarray::IntoDimension;
use numpy::npyffi::flags;
use numpy::{Element, IntoPyArray, PyArray1};
use polars_core::prelude::*;
use polars_core::utils::dtypes_to_supertype;
use polars_core::with_match_physical_numeric_polars_type;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::to_numpy_series::series_to_numpy;
use super::utils::{
    create_borrowed_np_array, dtype_supports_view, polars_dtype_to_np_temporal_dtype,
};
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
}

pub(super) fn df_to_numpy(
    py: Python,
    df: &DataFrame,
    order: IndexOrder,
    writable: bool,
    allow_copy: bool,
) -> PyResult<PyObject> {
    if df.is_empty() {
        // Take this path to ensure a writable array.
        // This does not actually copy data for an empty DataFrame.
        return df_to_numpy_with_copy(py, df, order, true);
    }

    if matches!(order, IndexOrder::Fortran) {
        if let Some(mut arr) = try_df_to_numpy_view(py, df, false) {
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

/// Create a NumPy view of the given DataFrame.
fn try_df_to_numpy_view(py: Python, df: &DataFrame, allow_nulls: bool) -> Option<PyObject> {
    let first_dtype = check_df_dtypes_support_view(df)?;

    // TODO: Check for nested nulls using `series_contains_null` util when we support Array types.
    if !allow_nulls && df.get_columns().iter().any(|s| s.null_count() > 0) {
        return None;
    }
    if !check_df_columns_contiguous(df) {
        return None;
    }

    let owner = PyDataFrame::from(df.clone()).into_py(py); // Keep the DataFrame memory alive.

    let arr = match first_dtype {
        dt if dt.is_numeric() => {
            with_match_physical_numeric_polars_type!(first_dtype, |$T| {
                numeric_df_to_numpy_view::<$T>(py, df, owner)
            })
        },
        DataType::Datetime(_, _) | DataType::Duration(_) => {
            temporal_df_to_numpy_view(py, df, owner)
        },
        _ => unreachable!(),
    };
    Some(arr)
}
/// Check whether the data types of the DataFrame allow for creating a NumPy view.
///
/// Returns the common data type if it is supported, otherwise returns `None`.
fn check_df_dtypes_support_view(df: &DataFrame) -> Option<&DataType> {
    let columns = df.get_columns();
    let first_dtype = columns.first()?.dtype();

    // TODO: Support viewing Array types
    if first_dtype.is_array() || !dtype_supports_view(first_dtype) {
        return None;
    }
    if columns.iter().any(|s| s.dtype() != first_dtype) {
        return None;
    }
    Some(first_dtype)
}
/// Returns whether all columns of the dataframe are contiguous in memory.
fn check_df_columns_contiguous(df: &DataFrame) -> bool {
    let columns = df.get_columns();

    if columns.iter().any(|s| s.n_chunks() > 1) {
        return false;
    }
    if columns.len() <= 1 {
        return true;
    }

    match columns.first().unwrap().dtype() {
        dt if dt.is_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let slices = columns
                    .iter()
                    .map(|s| {
                        let ca: &ChunkedArray<$T> = s.unpack().unwrap();
                        ca.data_views().next().unwrap()
                    })
                    .collect::<Vec<_>>();

                check_slices_contiguous::<$T>(slices)
            })
        },
        DataType::Datetime(_, _) | DataType::Duration(_) => {
            let phys: Vec<_> = columns.iter().map(|s| s.to_physical_repr()).collect();
            let slices = phys
                .iter()
                .map(|s| {
                    let ca = s.i64().unwrap();
                    ca.data_views().next().unwrap()
                })
                .collect::<Vec<_>>();

            check_slices_contiguous::<Int64Type>(slices)
        },
        _ => panic!("invalid data type"),
    }
}
/// Returns whether the end and start pointers of all consecutive slices match.
fn check_slices_contiguous<T>(slices: Vec<&[T::Native]>) -> bool
where
    T: PolarsNumericType,
{
    let first_slice = slices.first().unwrap();

    // Check whether all arrays are from the same buffer.
    let mut end_ptr = unsafe { first_slice.as_ptr().add(first_slice.len()) };
    slices[1..].iter().all(|slice| {
        let slice_ptr = slice.as_ptr();
        let valid = slice_ptr == end_ptr;

        end_ptr = unsafe { slice_ptr.add(slice.len()) };

        valid
    })
}

/// Create a NumPy view of a numeric DataFrame.
fn numeric_df_to_numpy_view<T>(py: Python, df: &DataFrame, owner: PyObject) -> PyObject
where
    T: PolarsNumericType,
    T::Native: Element,
{
    let ca: &ChunkedArray<T> = df.get_columns().first().unwrap().unpack().unwrap();
    let first_slice = ca.data_views().next().unwrap();

    let start_ptr = first_slice.as_ptr();
    let np_dtype = T::Native::get_dtype_bound(py);
    let dims = [first_slice.len(), df.width()].into_dimension();

    unsafe {
        create_borrowed_np_array::<_>(
            py,
            np_dtype,
            dims,
            flags::NPY_ARRAY_FARRAY_RO,
            start_ptr as _,
            owner,
        )
    }
}
/// Create a NumPy view of a Datetime or Duration DataFrame.
fn temporal_df_to_numpy_view(py: Python, df: &DataFrame, owner: PyObject) -> PyObject {
    let s = df.get_columns().first().unwrap();
    let phys = s.to_physical_repr();
    let ca = phys.i64().unwrap();
    let first_slice = ca.data_views().next().unwrap();

    let start_ptr = first_slice.as_ptr();
    let np_dtype = polars_dtype_to_np_temporal_dtype(py, s.dtype());
    let dims = [first_slice.len(), df.width()].into_dimension();

    unsafe {
        create_borrowed_np_array::<_>(
            py,
            np_dtype,
            dims,
            flags::NPY_ARRAY_FARRAY_RO,
            start_ptr as _,
            owner,
        )
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
        df_columns_to_numpy(py, df, order, writable)
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
fn df_columns_to_numpy(
    py: Python,
    df: &DataFrame,
    order: IndexOrder,
    writable: bool,
) -> PyResult<PyObject> {
    let np_arrays = df.iter().map(|s| {
        let mut arr = series_to_numpy(py, s, writable, true).unwrap();

        // Convert multidimensional arrays to 1D object arrays.
        let shape: Vec<usize> = arr
            .getattr(py, intern!(py, "shape"))
            .unwrap()
            .extract(py)
            .unwrap();
        if shape.len() > 1 {
            // TODO: Downcast the NumPy array to Rust and split without calling into Python.
            let subarrays = (0..shape[0]).map(|idx| {
                arr.call_method1(py, intern!(py, "__getitem__"), (idx,))
                    .unwrap()
            });
            arr = PyArray1::from_iter_bound(py, subarrays).into_py(py);
        }
        arr
    });

    let numpy = PyModule::import_bound(py, intern!(py, "numpy"))?;
    let np_array = match order {
        IndexOrder::C => numpy
            .getattr(intern!(py, "column_stack"))
            .unwrap()
            .call1((PyList::new_bound(py, np_arrays),))?,
        IndexOrder::Fortran => numpy
            .getattr(intern!(py, "vstack"))
            .unwrap()
            .call1((PyList::new_bound(py, np_arrays),))?
            .getattr(intern!(py, "T"))?,
    };

    Ok(np_array.into())
}
