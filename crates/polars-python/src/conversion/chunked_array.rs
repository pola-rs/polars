use chrono::NaiveTime;
use polars_core::utils::arrow::temporal_conversions::date32_to_date;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyNone, PyTuple};
use pyo3::{intern, BoundObject};

use super::datetime::{
    datetime_to_py_object, elapsed_offset_to_timedelta, nanos_since_midnight_to_naivetime,
};
use super::{decimal_to_digits, struct_dict};
use crate::prelude::*;
use crate::py_modules::pl_utils;

impl<'py> IntoPyObject<'py> for &Wrap<&StringChunked> {
    type Target = PyList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let iter = self.0.iter();
        PyList::new(py, iter)
    }
}

impl<'py> IntoPyObject<'py> for &Wrap<&BinaryChunked> {
    type Target = PyList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let iter = self
            .0
            .iter()
            .map(|opt_bytes| opt_bytes.map(|bytes| PyBytes::new(py, bytes)));
        PyList::new(py, iter)
    }
}

impl<'py> IntoPyObject<'py> for &Wrap<&StructChunked> {
    type Target = PyList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let s = self.0.clone().into_series();
        // todo! iterate its chunks and flatten.
        // make series::iter() accept a chunk index.
        let s = s.rechunk();
        let iter = s.iter().map(|av| match av {
            AnyValue::Struct(_, _, flds) => struct_dict(py, av._iter_struct_av(), flds)
                .unwrap()
                .into_any(),
            AnyValue::Null => PyNone::get(py).into_bound().into_any(),
            _ => unreachable!(),
        });

        PyList::new(py, iter)
    }
}

impl<'py> IntoPyObject<'py> for &Wrap<&DurationChunked> {
    type Target = PyList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let time_unit = self.0.time_unit();
        let iter = self
            .0
            .iter()
            .map(|opt_v| opt_v.map(|v| elapsed_offset_to_timedelta(v, time_unit)));
        PyList::new(py, iter)
    }
}

impl<'py> IntoPyObject<'py> for &Wrap<&DatetimeChunked> {
    type Target = PyList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let time_zone = self.0.time_zone().as_ref();
        let time_unit = self.0.time_unit();
        let iter = self.0.iter().map(|opt_v| {
            opt_v.map(|v| datetime_to_py_object(py, v, time_unit, time_zone).unwrap())
        });
        PyList::new(py, iter)
    }
}

impl<'py> IntoPyObject<'py> for &Wrap<&TimeChunked> {
    type Target = PyList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let iter = time_to_pyobject_iter(self.0);
        PyList::new(py, iter)
    }
}

pub(crate) fn time_to_pyobject_iter(
    ca: &TimeChunked,
) -> impl '_ + ExactSizeIterator<Item = Option<NaiveTime>> {
    ca.0.iter()
        .map(move |opt_v| opt_v.map(nanos_since_midnight_to_naivetime))
}

impl<'py> IntoPyObject<'py> for &Wrap<&DateChunked> {
    type Target = PyList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let iter = self.0.into_iter().map(|opt_v| opt_v.map(date32_to_date));
        PyList::new(py, iter)
    }
}

impl<'py> IntoPyObject<'py> for &Wrap<&DecimalChunked> {
    type Target = PyList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let iter = decimal_to_pyobject_iter(py, self.0)?;
        PyList::new(py, iter)
    }
}

pub(crate) fn decimal_to_pyobject_iter<'py, 'a>(
    py: Python<'py>,
    ca: &'a DecimalChunked,
) -> PyResult<impl ExactSizeIterator<Item = Option<Bound<'py, PyAny>>> + use<'py, 'a>> {
    let utils = pl_utils(py).bind(py);
    let convert = utils.getattr(intern!(py, "to_py_decimal"))?;
    let py_scale = (-(ca.scale() as i32)).into_pyobject(py)?;
    // if we don't know precision, the only safe bet is to set it to 39
    let py_precision = ca.precision().unwrap_or(39).into_pyobject(py)?;
    Ok(ca.iter().map(move |opt_v| {
        opt_v.map(|v| {
            // TODO! use AnyValue so that we have a single impl.
            const N: usize = 3;
            let mut buf = [0_u128; N];
            let n_digits = decimal_to_digits(v.abs(), &mut buf);
            let buf = unsafe {
                std::slice::from_raw_parts(
                    buf.as_slice().as_ptr() as *const u8,
                    N * size_of::<u128>(),
                )
            };
            let digits = PyTuple::new(py, buf.iter().take(n_digits)).unwrap();
            convert
                .call1((v.is_negative() as u8, digits, &py_precision, &py_scale))
                .unwrap()
        })
    }))
}
