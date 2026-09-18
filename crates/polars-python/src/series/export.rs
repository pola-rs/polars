use polars_ffi::version_0::SeriesExport;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};

use super::PySeries;
use crate::error::PyPolarsErr;
use crate::interop;
use crate::interop::arrow::to_py::series_to_stream;
use crate::prelude::*;

/// The elements of a Series as a Python list.
///
/// The one chunk of a Series that has one is walked as itself; a Series of several is collected
/// first, because one iterator across the chunks resolves the representation of the chunk it is
/// in per element.
///
/// Chaining the chunks' own iterators instead, behind an adapter that answers the length
/// `PyList::new` wants, was tried and is worse both ways: it cost the single-chunk walk its
/// `TrustedLen` iterator (1.01x -> 1.34x) and did not pay for the collect on the chunked one
/// (1.26x -> 1.32x). What the list costs is one Python object per element, and an adapter whose
/// `next` tests which chunk it is in adds to that rather than to the copy it saves.
fn elements_to_pylist<'py, T>(py: Python<'py>, ca: &ChunkedArray<T>) -> PyResult<Bound<'py, PyList>>
where
    T: PolarsDataType,
    for<'a> Option<T::Physical<'a>>: IntoPyObject<'py>,
{
    match ca.chunks().len() {
        1 => PyList::new(py, ca.downcast_iter().next().unwrap().iter()),
        _ => PyList::new(py, collect_elements(ca)),
    }
}

/// The elements of a Series, collected a chunk at a time.
///
/// A chunk walked on its own has its representation resolved once, where one iterator over the
/// chunks of the whole column resolves it per element — `next` stays out of line across the crate
/// boundary, so nothing hoists it back out.
fn collect_elements<T>(ca: &ChunkedArray<T>) -> Vec<Option<T::Physical<'_>>>
where
    T: PolarsDataType,
{
    let mut elements = Vec::with_capacity(ca.len());
    for arr in ca.downcast_iter() {
        arr.iter().for_each(|element| elements.push(element));
    }
    elements
}

#[pymethods]
impl PySeries {
    /// Convert this Series to a Python list.
    /// This operation copies data.
    pub fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let series = &self.series.read();

        fn to_list_recursive<'py>(py: Python<'py>, series: &Series) -> PyResult<Bound<'py, PyAny>> {
            let pylist = match series.dtype() {
                DataType::Boolean => {
                    elements_to_pylist(py, series.bool().map_err(PyPolarsErr::from)?)?
                },
                DataType::UInt8 => elements_to_pylist(py, series.u8().map_err(PyPolarsErr::from)?)?,
                DataType::UInt16 => {
                    elements_to_pylist(py, series.u16().map_err(PyPolarsErr::from)?)?
                },
                DataType::UInt32 => {
                    elements_to_pylist(py, series.u32().map_err(PyPolarsErr::from)?)?
                },
                DataType::UInt64 => {
                    elements_to_pylist(py, series.u64().map_err(PyPolarsErr::from)?)?
                },
                DataType::UInt128 => {
                    elements_to_pylist(py, series.u128().map_err(PyPolarsErr::from)?)?
                },
                DataType::Int8 => elements_to_pylist(py, series.i8().map_err(PyPolarsErr::from)?)?,
                DataType::Int16 => {
                    elements_to_pylist(py, series.i16().map_err(PyPolarsErr::from)?)?
                },
                DataType::Int32 => {
                    elements_to_pylist(py, series.i32().map_err(PyPolarsErr::from)?)?
                },
                DataType::Int64 => {
                    elements_to_pylist(py, series.i64().map_err(PyPolarsErr::from)?)?
                },
                DataType::Int128 => {
                    elements_to_pylist(py, series.i128().map_err(PyPolarsErr::from)?)?
                },
                DataType::Float16 => {
                    elements_to_pylist(py, series.f16().map_err(PyPolarsErr::from)?)?
                },
                DataType::Float32 => {
                    elements_to_pylist(py, series.f32().map_err(PyPolarsErr::from)?)?
                },
                DataType::Float64 => {
                    elements_to_pylist(py, series.f64().map_err(PyPolarsErr::from)?)?
                },
                DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                    with_match_categorical_physical_type!(series.dtype().cat_physical().unwrap(), |$C| {
                        PyList::new(py, series.cat::<$C>().unwrap().iter_str())?
                    })
                },
                #[cfg(feature = "object")]
                DataType::Object(_) => {
                    let v = PyList::empty(py);
                    for i in 0..series.len() {
                        let obj: Option<&ObjectValue> = series.get_object(i).map(|any| any.into());
                        v.append(obj)?;
                    }
                    v
                },
                DataType::List(_) => {
                    let v = PyList::empty(py);
                    let ca = series.list().map_err(PyPolarsErr::from)?;
                    for opt_s in ca.amortized_iter() {
                        match opt_s {
                            None => {
                                v.append(py.None())?;
                            },
                            Some(s) => {
                                let pylst = to_list_recursive(py, s.as_ref())?;
                                v.append(pylst)?;
                            },
                        }
                    }
                    v
                },
                DataType::Array(_, _) => {
                    let v = PyList::empty(py);
                    let ca = series.array().map_err(PyPolarsErr::from)?;
                    for opt_s in ca.amortized_iter() {
                        match opt_s {
                            None => {
                                v.append(py.None())?;
                            },
                            Some(s) => {
                                let pylst = to_list_recursive(py, s.as_ref())?;
                                v.append(pylst)?;
                            },
                        }
                    }
                    v
                },
                DataType::Date => {
                    let ca = series.date().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Time => {
                    let ca = series.time().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Datetime(_, _) => {
                    let ca = series.datetime().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Decimal(_, _) => {
                    let ca = series.decimal().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::String => {
                    let ca = series.str().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Struct(_) => {
                    let ca = series.struct_().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Duration(_) => {
                    let ca = series.duration().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Binary => {
                    let ca = series.binary().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Null => {
                    let null: Option<u8> = None;
                    let n = series.len();
                    let iter = std::iter::repeat_n(null, n);
                    use std::iter::RepeatN;
                    struct NullIter {
                        iter: RepeatN<Option<u8>>,
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

                    PyList::new(py, NullIter { iter, n })?
                },
                DataType::Unknown(_) => {
                    panic!("to_list not implemented for unknown")
                },
                DataType::BinaryOffset => {
                    unreachable!()
                },
                DataType::Map(_, _) => {
                    let ca = series.map().map_err(PyPolarsErr::from)?;
                    PyList::new(py, ca.any_value_iter().map(Wrap))?
                },
                DataType::Extension(_, _) => {
                    return to_list_recursive(py, series.ext().unwrap().storage());
                },
            };
            Ok(pylist.into_any())
        }

        to_list_recursive(py, series)
    }

    /// Return the underlying Arrow array.
    #[allow(clippy::wrong_self_convention)]
    fn to_arrow(&self, py: Python<'_>, compat_level: PyCompatLevel) -> PyResult<Py<PyAny>> {
        self.rechunk(py, true)?;
        let pyarrow = py.import("pyarrow")?;

        let s = self.series.read();
        interop::arrow::to_py::to_py_array(
            s.to_arrow(0, compat_level.0),
            &s.field().to_arrow(compat_level.0),
            &pyarrow,
        )
    }

    #[allow(unused_variables)]
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        series_to_stream(&self.series.read(), py)
    }

    pub fn _export(&self, _py: Python<'_>, location: usize) {
        let export = polars_ffi::version_0::export_series(&self.series.read());
        unsafe {
            (location as *mut SeriesExport).write(export);
        }
    }
}
