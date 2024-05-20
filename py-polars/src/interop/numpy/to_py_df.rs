use ndarray::IntoDimension;
use numpy::npyffi::flags;
use numpy::{Element, IntoPyArray};
use polars_core::prelude::*;
use polars_core::utils::dtypes_to_supertype;
use polars_core::with_match_physical_numeric_polars_type;
use pyo3::prelude::*;

use super::utils::create_borrowed_np_array;
use crate::conversion::Wrap;
use crate::dataframe::PyDataFrame;

#[pymethods]
#[allow(clippy::wrong_self_convention)]
impl PyDataFrame {
    /// Create a view of the data as a NumPy ndarray.
    ///
    /// WARNING: The resulting view will show the underlying value for nulls,
    /// which may be any value. The caller is responsible for handling nulls
    /// appropriately.
    pub fn to_numpy_view(&self, py: Python) -> Option<PyObject> {
        if self.df.is_empty() {
            return None;
        }
        let first = self.df.get_columns().first().unwrap().dtype();
        // TODO: Support Datetime/Duration types
        if !first.is_numeric() {
            return None;
        }
        if !self
            .df
            .get_columns()
            .iter()
            .all(|s| s.null_count() == 0 && s.dtype() == first && s.chunks().len() == 1)
        {
            return None;
        }

        let owner = self.clone().into_py(py); // Keep the DataFrame memory alive.

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
        with_match_physical_numeric_polars_type!(first, |$T| {
            get_ptr::<$T>(py, self.df.get_columns(), owner)
        })
    }

    /// Convert this DataFrame to a NumPy ndarray.
    pub fn to_numpy(&self, py: Python, order: Wrap<IndexOrder>) -> Option<PyObject> {
        let st = dtypes_to_supertype(self.df.iter().map(|s| s.dtype())).ok()?;

        let pyarray = match st {
            dt if dt.is_numeric() => with_match_physical_numeric_polars_type!(dt, |$T| {
                self.df.to_ndarray::<$T>(order.0).ok()?.into_pyarray_bound(py).into_py(py)
            }),
            _ => return None,
        };
        Some(pyarray)
    }
}
