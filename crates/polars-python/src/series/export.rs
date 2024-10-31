use polars_core::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};

use super::PySeries;
use crate::interop;
use crate::interop::arrow::to_py::series_to_stream;
use crate::prelude::*;

#[pymethods]
impl PySeries {
    /// Convert this Series to a Python list.
    /// This operation copies data.
    pub fn to_list(&self) -> PyObject {
        Python::with_gil(|py| {
            let series = &self.series;

            fn to_list_recursive(py: Python, series: &Series) -> PyObject {
                let pylist = match series.dtype() {
                    DataType::Boolean => PyList::new_bound(py, series.bool().unwrap()),
                    DataType::UInt8 => PyList::new_bound(py, series.u8().unwrap()),
                    DataType::UInt16 => PyList::new_bound(py, series.u16().unwrap()),
                    DataType::UInt32 => PyList::new_bound(py, series.u32().unwrap()),
                    DataType::UInt64 => PyList::new_bound(py, series.u64().unwrap()),
                    DataType::Int8 => PyList::new_bound(py, series.i8().unwrap()),
                    DataType::Int16 => PyList::new_bound(py, series.i16().unwrap()),
                    DataType::Int32 => PyList::new_bound(py, series.i32().unwrap()),
                    DataType::Int64 => PyList::new_bound(py, series.i64().unwrap()),
                    DataType::Float32 => PyList::new_bound(py, series.f32().unwrap()),
                    DataType::Float64 => PyList::new_bound(py, series.f64().unwrap()),
                    DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                        PyList::new_bound(py, series.categorical().unwrap().iter_str())
                    },
                    #[cfg(feature = "object")]
                    DataType::Object(_, _) => {
                        let v = PyList::empty_bound(py);
                        for i in 0..series.len() {
                            let obj: Option<&ObjectValue> =
                                series.get_object(i).map(|any| any.into());
                            let val = obj.to_object(py);

                            v.append(val).unwrap();
                        }
                        v
                    },
                    DataType::List(_) => {
                        let v = PyList::empty_bound(py);
                        let ca = series.list().unwrap();
                        for opt_s in ca.amortized_iter() {
                            match opt_s {
                                None => {
                                    v.append(py.None()).unwrap();
                                },
                                Some(s) => {
                                    let pylst = to_list_recursive(py, s.as_ref());
                                    v.append(pylst).unwrap();
                                },
                            }
                        }
                        v
                    },
                    DataType::Array(_, _) => {
                        let v = PyList::empty_bound(py);
                        let ca = series.array().unwrap();
                        for opt_s in ca.amortized_iter() {
                            match opt_s {
                                None => {
                                    v.append(py.None()).unwrap();
                                },
                                Some(s) => {
                                    let pylst = to_list_recursive(py, s.as_ref());
                                    v.append(pylst).unwrap();
                                },
                            }
                        }
                        v
                    },
                    DataType::Date => {
                        let ca = series.date().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Time => {
                        let ca = series.time().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Datetime(_, _) => {
                        let ca = series.datetime().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Decimal(_, _) => {
                        let ca = series.decimal().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::String => {
                        let ca = series.str().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Struct(_) => {
                        let ca = series.struct_().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Duration(_) => {
                        let ca = series.duration().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Binary => {
                        let ca = series.binary().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Null => {
                        let null: Option<u8> = None;
                        let n = series.len();
                        let iter = std::iter::repeat(null).take(n);
                        use std::iter::{Repeat, Take};
                        struct NullIter {
                            iter: Take<Repeat<Option<u8>>>,
                            n: usize,
                        }
                        impl Iterator for NullIter {
                            type Item = Option<u8>;

                            fn next(&mut self) -> Option<Self::Item> {
                                self.iter.next()
                            }
                            fn size_hint(&self) -> (usize, Option<usize>) {
                                (self.n, Some(self.n))
                            }
                        }
                        impl ExactSizeIterator for NullIter {}

                        PyList::new_bound(py, NullIter { iter, n })
                    },
                    DataType::Unknown(_) => {
                        panic!("to_list not implemented for unknown")
                    },
                    DataType::BinaryOffset => {
                        unreachable!()
                    },
                };
                pylist.to_object(py)
            }

            let pylist = to_list_recursive(py, series);
            pylist.to_object(py)
        })
    }

    /// Return the underlying Arrow array.
    #[allow(clippy::wrong_self_convention)]
    fn to_arrow(&mut self, compat_level: PyCompatLevel) -> PyResult<PyObject> {
        self.rechunk(true);
        Python::with_gil(|py| {
            let pyarrow = py.import_bound("pyarrow")?;

            interop::arrow::to_py::to_py_array(
                self.series.to_arrow(0, compat_level.0),
                py,
                &pyarrow,
            )
        })
    }

    #[allow(unused_variables)]
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &'py self,
        py: Python<'py>,
        requested_schema: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        series_to_stream(&self.series, py)
    }
}
