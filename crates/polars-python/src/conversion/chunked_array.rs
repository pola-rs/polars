use polars_core::export::chrono::NaiveTime;
use polars_core::utils::arrow::temporal_conversions::date32_to_date;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyNone, PyTuple};

use super::datetime::{
    elapsed_offset_to_timedelta, nanos_since_midnight_to_naivetime, timestamp_to_naive_datetime,
};
use super::{decimal_to_digits, struct_dict};
use crate::prelude::*;
use crate::py_modules::UTILS;

impl ToPyObject for Wrap<&StringChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let iter = self.0.iter();
        PyList::new_bound(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&BinaryChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let iter = self
            .0
            .iter()
            .map(|opt_bytes| opt_bytes.map(|bytes| PyBytes::new_bound(py, bytes)));
        PyList::new_bound(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&StructChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let s = self.0.clone().into_series();
        // todo! iterate its chunks and flatten.
        // make series::iter() accept a chunk index.
        let s = s.rechunk();
        let iter = s.iter().map(|av| match av {
            AnyValue::Struct(_, _, flds) => struct_dict(py, av._iter_struct_av(), flds),
            AnyValue::Null => PyNone::get_bound(py).into_py(py),
            _ => unreachable!(),
        });

        PyList::new_bound(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DurationChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let time_unit = self.0.time_unit();
        let iter = self
            .0
            .iter()
            .map(|opt_v| opt_v.map(|v| elapsed_offset_to_timedelta(v, time_unit)));
        PyList::new_bound(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DatetimeChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let time_zone = self.0.time_zone();
        if time_zone.is_some() {
            // Switch to more efficient code path in
            // https://github.com/pola-rs/polars/issues/16199
            let utils = UTILS.bind(py);
            let convert = utils.getattr(intern!(py, "to_py_datetime")).unwrap();
            let time_unit = self.0.time_unit().to_ascii();
            let time_zone = time_zone.to_object(py);
            let iter = self
                .0
                .iter()
                .map(|opt_v| opt_v.map(|v| convert.call1((v, time_unit, &time_zone)).unwrap()));
            PyList::new_bound(py, iter).into_py(py)
        } else {
            let time_unit = self.0.time_unit();
            let iter = self
                .0
                .iter()
                .map(|opt_v| opt_v.map(|v| timestamp_to_naive_datetime(v, time_unit)));
            PyList::new_bound(py, iter).into_py(py)
        }
    }
}

impl ToPyObject for Wrap<&TimeChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let iter = time_to_pyobject_iter(self.0);
        PyList::new_bound(py, iter).into_py(py)
    }
}

pub(crate) fn time_to_pyobject_iter(
    ca: &TimeChunked,
) -> impl '_ + ExactSizeIterator<Item = Option<NaiveTime>> {
    ca.0.iter()
        .map(move |opt_v| opt_v.map(nanos_since_midnight_to_naivetime))
}

impl ToPyObject for Wrap<&DateChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let iter = self.0.into_iter().map(|opt_v| opt_v.map(date32_to_date));
        PyList::new_bound(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DecimalChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let iter = decimal_to_pyobject_iter(py, self.0);
        PyList::new_bound(py, iter).into_py(py)
    }
}

pub(crate) fn decimal_to_pyobject_iter<'a>(
    py: Python<'a>,
    ca: &'a DecimalChunked,
) -> impl ExactSizeIterator<Item = Option<Bound<'a, PyAny>>> {
    let utils = UTILS.bind(py);
    let convert = utils.getattr(intern!(py, "to_py_decimal")).unwrap();
    let py_scale = (-(ca.scale() as i32)).to_object(py);
    // if we don't know precision, the only safe bet is to set it to 39
    let py_precision = ca.precision().unwrap_or(39).to_object(py);
    ca.iter().map(move |opt_v| {
        opt_v.map(|v| {
            // TODO! use AnyValue so that we have a single impl.
            const N: usize = 3;
            let mut buf = [0_u128; N];
            let n_digits = decimal_to_digits(v.abs(), &mut buf);
            let buf = unsafe {
                std::slice::from_raw_parts(
                    buf.as_slice().as_ptr() as *const u8,
                    N * std::mem::size_of::<u128>(),
                )
            };
            let digits = PyTuple::new_bound(py, buf.iter().take(n_digits));
            convert
                .call1((v.is_negative() as u8, digits, &py_precision, &py_scale))
                .unwrap()
        })
    })
}
