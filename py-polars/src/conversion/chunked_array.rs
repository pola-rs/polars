use polars::prelude::AnyValue;
#[cfg(feature = "cloud")]
use pyo3::conversion::{FromPyObject, IntoPy};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};
use pyo3::{intern, PyAny, PyResult};

use super::{decimal_to_digits, struct_dict};
use crate::prelude::*;
use crate::py_modules::UTILS;

impl<'a, T> FromPyObject<'a> for Wrap<ChunkedArray<T>>
where
    T: PyPolarsNumericType,
    T::Native: FromPyObject<'a>,
{
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let len = obj.len()?;
        let mut builder = PrimitiveChunkedBuilder::new("", len);

        for res in obj.iter()? {
            let item = res?;
            match item.extract::<T::Native>() {
                Ok(val) => builder.append_value(val),
                Err(_) => builder.append_null(),
            };
        }
        Ok(Wrap(builder.finish()))
    }
}

impl<'a> FromPyObject<'a> for Wrap<BooleanChunked> {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let len = obj.len()?;
        let mut builder = BooleanChunkedBuilder::new("", len);

        for res in obj.iter()? {
            let item = res?;
            match item.extract::<bool>() {
                Ok(val) => builder.append_value(val),
                Err(_) => builder.append_null(),
            }
        }
        Ok(Wrap(builder.finish()))
    }
}

impl<'a> FromPyObject<'a> for Wrap<StringChunked> {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let len = obj.len()?;
        let mut builder = StringChunkedBuilder::new("", len);

        for res in obj.iter()? {
            let item = res?;
            match item.extract::<&str>() {
                Ok(val) => builder.append_value(val),
                Err(_) => builder.append_null(),
            }
        }
        Ok(Wrap(builder.finish()))
    }
}

impl<'a> FromPyObject<'a> for Wrap<BinaryChunked> {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let len = obj.len()?;
        let mut builder = BinaryChunkedBuilder::new("", len);

        for res in obj.iter()? {
            let item = res?;
            match item.extract::<&[u8]>() {
                Ok(val) => builder.append_value(val),
                Err(_) => builder.append_null(),
            }
        }
        Ok(Wrap(builder.finish()))
    }
}

impl ToPyObject for Wrap<&StringChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let iter = self.0.into_iter();
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&BinaryChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let iter = self
            .0
            .into_iter()
            .map(|opt_bytes| opt_bytes.map(|bytes| PyBytes::new(py, bytes)));
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&StructChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let s = self.0.clone().into_series();
        // todo! iterate its chunks and flatten.
        // make series::iter() accept a chunk index.
        let s = s.rechunk();
        let iter = s.iter().map(|av| {
            if let AnyValue::Struct(_, _, flds) = av {
                struct_dict(py, av._iter_struct_av(), flds)
            } else {
                unreachable!()
            }
        });

        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DurationChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let utils = UTILS.as_ref(py);
        let convert = utils.getattr(intern!(py, "_to_python_timedelta")).unwrap();
        let time_unit = Wrap(self.0.time_unit()).to_object(py);
        let iter = self
            .0
            .into_iter()
            .map(|opt_v| opt_v.map(|v| convert.call1((v, &time_unit)).unwrap()));
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DatetimeChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let utils = UTILS.as_ref(py);
        let convert = utils.getattr(intern!(py, "_to_python_datetime")).unwrap();
        let time_unit = Wrap(self.0.time_unit()).to_object(py);
        let time_zone = self.0.time_zone().to_object(py);
        let iter = self
            .0
            .into_iter()
            .map(|opt_v| opt_v.map(|v| convert.call1((v, &time_unit, &time_zone)).unwrap()));
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&TimeChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let utils = UTILS.as_ref(py);
        let convert = utils.getattr(intern!(py, "_to_python_time")).unwrap();
        let iter = self
            .0
            .into_iter()
            .map(|opt_v| opt_v.map(|v| convert.call1((v,)).unwrap()));
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DateChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let utils = UTILS.as_ref(py);
        let convert = utils.getattr(intern!(py, "_to_python_date")).unwrap();
        let iter = self
            .0
            .into_iter()
            .map(|opt_v| opt_v.map(|v| convert.call1((v,)).unwrap()));
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DecimalChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let utils = UTILS.as_ref(py);
        let convert = utils.getattr(intern!(py, "_to_python_decimal")).unwrap();
        let py_scale = (-(self.0.scale() as i32)).to_object(py);
        // if we don't know precision, the only safe bet is to set it to 39
        let py_precision = self.0.precision().unwrap_or(39).to_object(py);
        let iter = self.0.into_iter().map(|opt_v| {
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
                let digits = PyTuple::new(py, buf.iter().take(n_digits));
                convert
                    .call1((v.is_negative() as u8, digits, &py_precision, &py_scale))
                    .unwrap()
            })
        });
        PyList::new(py, iter).into_py(py)
    }
}
