use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

#[cfg(feature = "object")]
use polars::chunked_array::object::PolarsObjectSafe;
use polars::frame::row::Row;
use polars::frame::NullStrategy;
#[cfg(feature = "avro")]
use polars::io::avro::AvroCompression;
#[cfg(feature = "ipc")]
use polars::io::ipc::IpcCompression;
use polars::prelude::AnyValue;
use polars::series::ops::NullBehavior;
use polars_core::frame::row::any_values_to_dtype;
use polars_core::prelude::{IndexOrder, QuantileInterpolOptions};
use polars_core::utils::arrow::types::NativeType;
use polars_utils::total_ord::TotalEq;
use polars_lazy::prelude::*;
#[cfg(feature = "cloud")]
use polars_rs::io::cloud::CloudOptions;
use pyo3::basic::CompareOp;
use pyo3::conversion::{FromPyObject, IntoPy};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyList, PySequence, PyString, PyTuple, PyType,
};
use pyo3::{intern, PyAny, PyResult};
use smartstring::alias::String as SmartString;

use crate::error::PyPolarsErr;
#[cfg(feature = "object")]
use crate::object::OBJECT_NAME;
use crate::prelude::*;
use crate::py_modules::{POLARS, SERIES, UTILS};
use crate::series::PySeries;
use crate::{PyDataFrame, PyLazyFrame};

pub(crate) fn slice_to_wrapped<T>(slice: &[T]) -> &[Wrap<T>] {
    // Safety:
    // Wrap is transparent.
    unsafe { std::mem::transmute(slice) }
}

pub(crate) fn slice_extract_wrapped<T>(slice: &[Wrap<T>]) -> &[T] {
    // Safety:
    // Wrap is transparent.
    unsafe { std::mem::transmute(slice) }
}

pub(crate) fn vec_extract_wrapped<T>(buf: Vec<Wrap<T>>) -> Vec<T> {
    // Safety:
    // Wrap is transparent.
    unsafe { std::mem::transmute(buf) }
}

#[repr(transparent)]
pub struct Wrap<T>(pub T);

impl<T> Clone for Wrap<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Wrap(self.0.clone())
    }
}
impl<T> From<T> for Wrap<T> {
    fn from(t: T) -> Self {
        Wrap(t)
    }
}

// extract a Rust DataFrame from a python DataFrame, that is DataFrame<PyDataFrame<RustDataFrame>>
pub(crate) fn get_df(obj: &PyAny) -> PyResult<DataFrame> {
    let pydf = obj.getattr(intern!(obj.py(), "_df"))?;
    Ok(pydf.extract::<PyDataFrame>()?.df)
}

pub(crate) fn get_lf(obj: &PyAny) -> PyResult<LazyFrame> {
    let pydf = obj.getattr(intern!(obj.py(), "_ldf"))?;
    Ok(pydf.extract::<PyLazyFrame>()?.ldf)
}

pub(crate) fn get_series(obj: &PyAny) -> PyResult<Series> {
    let pydf = obj.getattr(intern!(obj.py(), "_s"))?;
    Ok(pydf.extract::<PySeries>()?.series)
}

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
            }
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

impl<'a> FromPyObject<'a> for Wrap<Utf8Chunked> {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let len = obj.len()?;
        let mut builder = Utf8ChunkedBuilder::new("", len, len * 25);

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
        let mut builder = BinaryChunkedBuilder::new("", len, len * 25);

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

#[cfg(feature = "csv")]
impl<'a> FromPyObject<'a> for Wrap<NullValues> {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        if let Ok(s) = ob.extract::<String>() {
            Ok(Wrap(NullValues::AllColumnsSingle(s)))
        } else if let Ok(s) = ob.extract::<Vec<String>>() {
            Ok(Wrap(NullValues::AllColumns(s)))
        } else if let Ok(s) = ob.extract::<Vec<(String, String)>>() {
            Ok(Wrap(NullValues::Named(s)))
        } else {
            Err(
                PyPolarsErr::Other("could not extract value from null_values argument".into())
                    .into(),
            )
        }
    }
}

fn struct_dict<'a>(
    py: Python,
    vals: impl Iterator<Item = AnyValue<'a>>,
    flds: &[Field],
) -> PyObject {
    let dict = PyDict::new(py);
    for (fld, val) in flds.iter().zip(vals) {
        dict.set_item(fld.name().as_str(), Wrap(val)).unwrap()
    }
    dict.into_py(py)
}

// accept u128 array to ensure alignment is correct
fn decimal_to_digits(v: i128, buf: &mut [u128; 3]) -> usize {
    const ZEROS: i128 = 0x3030_3030_3030_3030_3030_3030_3030_3030;
    // safety: transmute is safe as there are 48 bytes in 3 128bit ints
    // and the minimal alignment of u8 fits u16
    let buf = unsafe { std::mem::transmute::<&mut [u128; 3], &mut [u8; 48]>(buf) };
    let mut buffer = itoa::Buffer::new();
    let value = buffer.format(v);
    let len = value.len();
    for (dst, src) in buf.iter_mut().zip(value.as_bytes().iter()) {
        *dst = *src
    }

    let ptr = buf.as_mut_ptr() as *mut i128;
    unsafe {
        // this is safe because we know that the buffer is exactly 48 bytes long
        *ptr -= ZEROS;
        *ptr.add(1) -= ZEROS;
        *ptr.add(2) -= ZEROS;
    }
    len
}

impl IntoPy<PyObject> for Wrap<AnyValue<'_>> {
    fn into_py(self, py: Python) -> PyObject {
        let utils = UTILS.as_ref(py);
        match self.0 {
            AnyValue::UInt8(v) => v.into_py(py),
            AnyValue::UInt16(v) => v.into_py(py),
            AnyValue::UInt32(v) => v.into_py(py),
            AnyValue::UInt64(v) => v.into_py(py),
            AnyValue::Int8(v) => v.into_py(py),
            AnyValue::Int16(v) => v.into_py(py),
            AnyValue::Int32(v) => v.into_py(py),
            AnyValue::Int64(v) => v.into_py(py),
            AnyValue::Float32(v) => v.into_py(py),
            AnyValue::Float64(v) => v.into_py(py),
            AnyValue::Null => py.None(),
            AnyValue::Boolean(v) => v.into_py(py),
            AnyValue::Utf8(v) => v.into_py(py),
            AnyValue::Utf8Owned(v) => v.into_py(py),
            AnyValue::Categorical(idx, rev, arr) => {
                let s = if arr.is_null() {
                    rev.get(idx)
                } else {
                    unsafe { arr.deref_unchecked().value(idx as usize) }
                };
                s.into_py(py)
            },
            AnyValue::Date(v) => {
                let convert = utils.getattr(intern!(py, "_to_python_date")).unwrap();
                convert.call1((v,)).unwrap().into_py(py)
            },
            AnyValue::Datetime(v, time_unit, time_zone) => {
                let convert = utils.getattr(intern!(py, "_to_python_datetime")).unwrap();
                let time_unit = time_unit.to_ascii();
                convert
                    .call1((v, time_unit, time_zone.as_ref().map(|s| s.as_str())))
                    .unwrap()
                    .into_py(py)
            },
            AnyValue::Duration(v, time_unit) => {
                let convert = utils.getattr(intern!(py, "_to_python_timedelta")).unwrap();
                let time_unit = time_unit.to_ascii();
                convert.call1((v, time_unit)).unwrap().into_py(py)
            },
            AnyValue::Time(v) => {
                let convert = utils.getattr(intern!(py, "_to_python_time")).unwrap();
                convert.call1((v,)).unwrap().into_py(py)
            },
            AnyValue::Array(v, _) | AnyValue::List(v) => PySeries::new(v).to_list(),
            ref av @ AnyValue::Struct(_, _, flds) => struct_dict(py, av._iter_struct_av(), flds),
            AnyValue::StructOwned(payload) => struct_dict(py, payload.0.into_iter(), &payload.1),
            #[cfg(feature = "object")]
            AnyValue::Object(v) => {
                let object = v.as_any().downcast_ref::<ObjectValue>().unwrap();
                object.inner.clone()
            },
            #[cfg(feature = "object")]
            AnyValue::ObjectOwned(v) => {
                let object = v.0.as_any().downcast_ref::<ObjectValue>().unwrap();
                object.inner.clone()
            },
            AnyValue::Binary(v) => v.into_py(py),
            AnyValue::BinaryOwned(v) => v.into_py(py),
            AnyValue::Decimal(v, scale) => {
                let convert = utils.getattr(intern!(py, "_to_python_decimal")).unwrap();
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
                    .call1((v.is_negative() as u8, digits, n_digits, -(scale as i32)))
                    .unwrap()
                    .into_py(py)
            },
        }
    }
}

impl ToPyObject for Wrap<DataType> {
    fn to_object(&self, py: Python) -> PyObject {
        let pl = POLARS.as_ref(py);

        match &self.0 {
            DataType::Int8 => pl.getattr(intern!(py, "Int8")).unwrap().into(),
            DataType::Int16 => pl.getattr(intern!(py, "Int16")).unwrap().into(),
            DataType::Int32 => pl.getattr(intern!(py, "Int32")).unwrap().into(),
            DataType::Int64 => pl.getattr(intern!(py, "Int64")).unwrap().into(),
            DataType::UInt8 => pl.getattr(intern!(py, "UInt8")).unwrap().into(),
            DataType::UInt16 => pl.getattr(intern!(py, "UInt16")).unwrap().into(),
            DataType::UInt32 => pl.getattr(intern!(py, "UInt32")).unwrap().into(),
            DataType::UInt64 => pl.getattr(intern!(py, "UInt64")).unwrap().into(),
            DataType::Float32 => pl.getattr(intern!(py, "Float32")).unwrap().into(),
            DataType::Float64 => pl.getattr(intern!(py, "Float64")).unwrap().into(),
            DataType::Decimal(precision, scale) => {
                let kwargs = PyDict::new(py);
                kwargs.set_item("precision", *precision).unwrap();
                kwargs.set_item("scale", *scale).unwrap();
                pl.getattr(intern!(py, "Decimal"))
                    .unwrap()
                    .call((), Some(kwargs))
                    .unwrap()
                    .into()
            },
            DataType::Boolean => pl.getattr(intern!(py, "Boolean")).unwrap().into(),
            DataType::Utf8 => pl.getattr(intern!(py, "Utf8")).unwrap().into(),
            DataType::Binary => pl.getattr(intern!(py, "Binary")).unwrap().into(),
            DataType::Array(inner, size) => {
                let inner = Wrap(*inner.clone()).to_object(py);
                let list_class = pl.getattr(intern!(py, "Array")).unwrap();
                let kwargs = PyDict::new(py);
                kwargs.set_item("inner", inner).unwrap();
                kwargs.set_item("width", size).unwrap();
                list_class.call((), Some(kwargs)).unwrap().into()
            },
            DataType::List(inner) => {
                let inner = Wrap(*inner.clone()).to_object(py);
                let list_class = pl.getattr(intern!(py, "List")).unwrap();
                list_class.call1((inner,)).unwrap().into()
            },
            DataType::Date => pl.getattr(intern!(py, "Date")).unwrap().into(),
            DataType::Datetime(tu, tz) => {
                let datetime_class = pl.getattr(intern!(py, "Datetime")).unwrap();
                datetime_class
                    .call1((tu.to_ascii(), tz.clone()))
                    .unwrap()
                    .into()
            },
            DataType::Duration(tu) => {
                let duration_class = pl.getattr(intern!(py, "Duration")).unwrap();
                duration_class.call1((tu.to_ascii(),)).unwrap().into()
            },
            #[cfg(feature = "object")]
            DataType::Object(_) => pl.getattr(intern!(py, "Object")).unwrap().into(),
            DataType::Categorical(_) => pl.getattr(intern!(py, "Categorical")).unwrap().into(),
            DataType::Time => pl.getattr(intern!(py, "Time")).unwrap().into(),
            DataType::Struct(fields) => {
                let field_class = pl.getattr(intern!(py, "Field")).unwrap();
                let iter = fields.iter().map(|fld| {
                    let name = fld.name().as_str();
                    let dtype = Wrap(fld.data_type().clone()).to_object(py);
                    field_class.call1((name, dtype)).unwrap()
                });
                let fields = PyList::new(py, iter);
                let struct_class = pl.getattr(intern!(py, "Struct")).unwrap();
                struct_class.call1((fields,)).unwrap().into()
            },
            DataType::Null => pl.getattr(intern!(py, "Null")).unwrap().into(),
            DataType::Unknown => pl.getattr(intern!(py, "Unknown")).unwrap().into(),
        }
    }
}

impl FromPyObject<'_> for Wrap<Field> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let py = ob.py();
        let name = ob.getattr(intern!(py, "name"))?.str()?.to_str()?;
        let dtype = ob
            .getattr(intern!(py, "dtype"))?
            .extract::<Wrap<DataType>>()?;
        Ok(Wrap(Field::new(name, dtype.0)))
    }
}

impl FromPyObject<'_> for Wrap<DataType> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let py = ob.py();
        let type_name = ob.get_type().name()?;

        let dtype = match type_name {
            "DataTypeClass" => {
                // just the class, not an object
                let name = ob.getattr(intern!(py, "__name__"))?.str()?.to_str()?;
                match name {
                    "UInt8" => DataType::UInt8,
                    "UInt16" => DataType::UInt16,
                    "UInt32" => DataType::UInt32,
                    "UInt64" => DataType::UInt64,
                    "Int8" => DataType::Int8,
                    "Int16" => DataType::Int16,
                    "Int32" => DataType::Int32,
                    "Int64" => DataType::Int64,
                    "Utf8" => DataType::Utf8,
                    "Binary" => DataType::Binary,
                    "Boolean" => DataType::Boolean,
                    "Categorical" => DataType::Categorical(None),
                    "Date" => DataType::Date,
                    "Datetime" => DataType::Datetime(TimeUnit::Microseconds, None),
                    "Time" => DataType::Time,
                    "Duration" => DataType::Duration(TimeUnit::Microseconds),
                    "Decimal" => DataType::Decimal(None, None), // "none" scale => "infer"
                    "Float32" => DataType::Float32,
                    "Float64" => DataType::Float64,
                    #[cfg(feature = "object")]
                    "Object" => DataType::Object(OBJECT_NAME),
                    "Array" => DataType::Array(Box::new(DataType::Null), 0),
                    "List" => DataType::List(Box::new(DataType::Null)),
                    "Struct" => DataType::Struct(vec![]),
                    "Null" => DataType::Null,
                    "Unknown" => DataType::Unknown,
                    dt => {
                        return Err(PyValueError::new_err(format!(
                            "{dt} is not a recognised polars DataType.",
                        )))
                    },
                }
            },
            "Duration" => {
                let time_unit = ob.getattr(intern!(py, "time_unit")).unwrap();
                let time_unit = time_unit.extract::<Wrap<TimeUnit>>()?.0;
                DataType::Duration(time_unit)
            },
            "Datetime" => {
                let time_unit = ob.getattr(intern!(py, "time_unit")).unwrap();
                let time_unit = time_unit.extract::<Wrap<TimeUnit>>()?.0;
                let time_zone = ob.getattr(intern!(py, "time_zone")).unwrap();
                let time_zone = time_zone.extract()?;
                DataType::Datetime(time_unit, time_zone)
            },
            "Decimal" => {
                let precision = ob.getattr(intern!(py, "precision"))?.extract()?;
                let scale = ob.getattr(intern!(py, "scale"))?.extract()?;
                DataType::Decimal(precision, Some(scale))
            },
            "List" => {
                let inner = ob.getattr(intern!(py, "inner")).unwrap();
                let inner = inner.extract::<Wrap<DataType>>()?;
                DataType::List(Box::new(inner.0))
            },
            "Array" => {
                let inner = ob.getattr(intern!(py, "inner")).unwrap();
                let width = ob.getattr(intern!(py, "width")).unwrap();
                let inner = inner.extract::<Wrap<DataType>>()?;
                let width = width.extract::<usize>()?;
                DataType::Array(Box::new(inner.0), width)
            },
            "Struct" => {
                let fields = ob.getattr(intern!(py, "fields"))?;
                let fields = fields
                    .extract::<Vec<Wrap<Field>>>()?
                    .into_iter()
                    .map(|f| f.0)
                    .collect::<Vec<Field>>();
                DataType::Struct(fields)
            },
            dt => {
                return Err(PyTypeError::new_err(format!(
                    "A {dt} object is not a recognised polars DataType. \
                    Hint: use the class without instantiating it.",
                )))
            },
        };
        Ok(Wrap(dtype))
    }
}

impl ToPyObject for Wrap<AnyValue<'_>> {
    fn to_object(&self, py: Python) -> PyObject {
        self.clone().into_py(py)
    }
}

impl ToPyObject for Wrap<TimeUnit> {
    fn to_object(&self, py: Python) -> PyObject {
        let time_unit = match self.0 {
            TimeUnit::Nanoseconds => "ns",
            TimeUnit::Microseconds => "us",
            TimeUnit::Milliseconds => "ms",
        };
        time_unit.into_py(py)
    }
}

impl ToPyObject for Wrap<&Utf8Chunked> {
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
                // TODO! use anyvalue so that we have a single impl.
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

fn abs_decimal_from_digits(
    digits: impl IntoIterator<Item = u8>,
    exp: i32,
) -> Option<(i128, usize)> {
    const MAX_ABS_DEC: i128 = 10_i128.pow(38) - 1;
    let mut v = 0_i128;
    for (i, d) in digits.into_iter().map(i128::from).enumerate() {
        if i < 38 {
            v = v * 10 + d;
        } else {
            v = v.checked_mul(10).and_then(|v| v.checked_add(d))?;
        }
    }
    // we only support non-negative scale (=> non-positive exponent)
    let scale = if exp > 0 {
        // the decimal may be in a non-canonical representation, try to fix it first
        v = 10_i128
            .checked_pow(exp as u32)
            .and_then(|factor| v.checked_mul(factor))?;
        0
    } else {
        (-exp) as usize
    };
    // TODO: do we care for checking if it fits in MAX_ABS_DEC? (if we set precision to None anyway?)
    (v <= MAX_ABS_DEC).then_some((v, scale))
}

fn convert_date(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
    Python::with_gil(|py| {
        let date = UTILS
            .as_ref(py)
            .getattr(intern!(py, "_date_to_pl_date"))
            .unwrap()
            .call1((ob,))
            .unwrap();
        let v = date.extract::<i32>().unwrap();
        Ok(Wrap(AnyValue::Date(v)))
    })
}
fn convert_datetime(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
    Python::with_gil(|py| {
        // windows
        #[cfg(target_arch = "windows")]
        let (seconds, microseconds) = {
            let convert = UTILS
                .getattr(py, intern!(py, "_datetime_for_anyvalue_windows"))
                .unwrap();
            let out = convert.call1(py, (ob,)).unwrap();
            let out: (i64, i64) = out.extract(py).unwrap();
            out
        };
        // unix
        #[cfg(not(target_arch = "windows"))]
        let (seconds, microseconds) = {
            let convert = UTILS
                .getattr(py, intern!(py, "_datetime_for_anyvalue"))
                .unwrap();
            let out = convert.call1(py, (ob,)).unwrap();
            let out: (i64, i64) = out.extract(py).unwrap();
            out
        };

        // s to us
        let mut v = seconds * 1_000_000;
        v += microseconds;

        // choose "us" as that is python's default unit
        Ok(AnyValue::Datetime(v, TimeUnit::Microseconds, &None).into())
    })
}

type TypeObjectPtr = usize;
type InitFn = fn(&PyAny) -> PyResult<Wrap<AnyValue<'_>>>;
pub(crate) static LUT: crate::gil_once_cell::GILOnceCell<PlHashMap<TypeObjectPtr, InitFn>> =
    crate::gil_once_cell::GILOnceCell::new();

impl<'s> FromPyObject<'s> for Wrap<AnyValue<'s>> {
    fn extract(ob: &'s PyAny) -> PyResult<Self> {
        // conversion functions
        fn get_bool(ob: &PyAny) -> PyResult<Wrap<AnyValue<'_>>> {
            Ok(AnyValue::Boolean(ob.extract::<bool>().unwrap()).into())
        }

        fn get_int(ob: &PyAny) -> PyResult<Wrap<AnyValue<'_>>> {
            // can overflow
            match ob.extract::<i64>() {
                Ok(v) => Ok(AnyValue::Int64(v).into()),
                Err(_) => Ok(AnyValue::UInt64(ob.extract::<u64>()?).into()),
            }
        }

        fn get_float(ob: &PyAny) -> PyResult<Wrap<AnyValue<'_>>> {
            Ok(AnyValue::Float64(ob.extract::<f64>().unwrap()).into())
        }

        fn get_str(ob: &PyAny) -> PyResult<Wrap<AnyValue<'_>>> {
            let value = ob.extract::<&str>().unwrap();
            Ok(AnyValue::Utf8(value).into())
        }

        fn get_struct(ob: &PyAny) -> PyResult<Wrap<AnyValue<'_>>> {
            let dict = ob.downcast::<PyDict>().unwrap();
            let len = dict.len();
            let mut keys = Vec::with_capacity(len);
            let mut vals = Vec::with_capacity(len);
            for (k, v) in dict.into_iter() {
                let key = k.extract::<&str>()?;
                let val = v.extract::<Wrap<AnyValue>>()?.0;
                let dtype = DataType::from(&val);
                keys.push(Field::new(key, dtype));
                vals.push(val)
            }
            Ok(Wrap(AnyValue::StructOwned(Box::new((vals, keys)))))
        }

        fn get_list(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            fn get_list_with_constructor(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
                // Use the dedicated constructor
                // this constructor is able to go via dedicated type constructors
                // so it can be much faster
                Python::with_gil(|py| {
                    let s = SERIES.call1(py, (ob,))?;
                    get_series_el(s.as_ref(py))
                })
            }

            if ob.is_empty()? {
                Ok(Wrap(AnyValue::List(Series::new_empty("", &DataType::Null))))
            } else if ob.is_instance_of::<PyList>() | ob.is_instance_of::<PyTuple>() {
                let list = ob.downcast::<PySequence>().unwrap();

                let mut avs = Vec::with_capacity(25);
                let mut iter = list.iter()?;

                for item in (&mut iter).take(25) {
                    avs.push(item?.extract::<Wrap<AnyValue>>()?.0)
                }

                let (dtype, n_types) = any_values_to_dtype(&avs).map_err(PyPolarsErr::from)?;

                // we only take this path if there is no question of the data-type
                if dtype.is_primitive() && n_types == 1 {
                    get_list_with_constructor(ob)
                } else {
                    // push the rest
                    avs.reserve(list.len()?);
                    for item in iter {
                        avs.push(item?.extract::<Wrap<AnyValue>>()?.0)
                    }

                    let s = Series::from_any_values_and_dtype("", &avs, &dtype, true)
                        .map_err(PyPolarsErr::from)?;
                    Ok(Wrap(AnyValue::List(s)))
                }
            } else {
                // range will take this branch
                get_list_with_constructor(ob)
            }
        }

        fn get_series_el(ob: &PyAny) -> PyResult<Wrap<AnyValue<'static>>> {
            let py_pyseries = ob.getattr(intern!(ob.py(), "_s")).unwrap();
            let series = py_pyseries.extract::<PySeries>().unwrap().series;
            Ok(Wrap(AnyValue::List(series)))
        }

        fn get_bin(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            let value = ob.extract::<&[u8]>().unwrap();
            Ok(AnyValue::Binary(value).into())
        }

        fn get_null(_ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            Ok(AnyValue::Null.into())
        }

        fn get_timedelta(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            Python::with_gil(|py| {
                let td = UTILS
                    .as_ref(py)
                    .getattr(intern!(py, "_timedelta_to_pl_timedelta"))
                    .unwrap()
                    .call1((ob, intern!(py, "us")))
                    .unwrap();
                let v = td.extract::<i64>().unwrap();
                Ok(Wrap(AnyValue::Duration(v, TimeUnit::Microseconds)))
            })
        }

        fn get_time(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            Python::with_gil(|py| {
                let time = UTILS
                    .as_ref(py)
                    .getattr(intern!(py, "_time_to_pl_time"))
                    .unwrap()
                    .call1((ob,))
                    .unwrap();
                let v = time.extract::<i64>().unwrap();
                Ok(Wrap(AnyValue::Time(v)))
            })
        }

        fn get_decimal(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            let (sign, digits, exp): (i8, Vec<u8>, i32) = ob
                .call_method0(intern!(ob.py(), "as_tuple"))
                .unwrap()
                .extract()
                .unwrap();
            // note: using Vec<u8> is not the most efficient thing here (input is a tuple)
            let (mut v, scale) = abs_decimal_from_digits(digits, exp).ok_or_else(|| {
                PyErr::from(PyPolarsErr::Other(
                    "Decimal is too large to fit in Decimal128".into(),
                ))
            })?;
            if sign > 0 {
                v = -v; // won't overflow since -i128::MAX > i128::MIN
            }
            Ok(Wrap(AnyValue::Decimal(v, scale)))
        }

        fn get_object(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            #[cfg(feature = "object")]
            {
                // this is slow, but hey don't use objects
                let v = &ObjectValue { inner: ob.into() };
                Ok(Wrap(AnyValue::ObjectOwned(OwnedObject(v.to_boxed()))))
            }
            #[cfg(not(feature = "object"))]
            {
                panic!("activate object")
            }
        }

        // TYPE key
        let type_object_ptr = PyType::as_type_ptr(ob.get_type()) as usize;

        Python::with_gil(|py| {
            LUT.with_gil(py, |lut| {
                // get the conversion function
                let convert_fn = lut.entry(type_object_ptr).or_insert_with(
                    // This only runs if type is not in LUT
                    || {
                        if ob.is_instance_of::<PyBool>() {
                            get_bool
                            // TODO: this heap allocs on failure
                        } else if ob.extract::<i64>().is_ok() || ob.extract::<u64>().is_ok() {
                            get_int
                        } else if ob.is_instance_of::<PyFloat>() {
                            get_float
                        } else if ob.is_instance_of::<PyString>() {
                            get_str
                        } else if ob.is_instance_of::<PyDict>() {
                            get_struct
                        } else if ob.is_instance_of::<PyList>() || ob.is_instance_of::<PyTuple>() {
                            get_list
                        } else if ob.hasattr(intern!(py, "_s")).unwrap() {
                            get_series_el
                        }
                        // TODO: this heap allocs on failure
                        else if ob.extract::<&'s [u8]>().is_ok() {
                            get_bin
                        } else if ob.is_none() {
                            get_null
                        } else {
                            let type_name = ob.get_type().name().unwrap();
                            match type_name {
                                "datetime" => convert_datetime,
                                "date" => convert_date,
                                "timedelta" => get_timedelta,
                                "time" => get_time,
                                "Decimal" => get_decimal,
                                "range" => get_list,
                                _ => {
                                    // special branch for np.float as this fails isinstance float
                                    if ob.extract::<f64>().is_ok() {
                                        return get_float;
                                    }

                                    // Can't use pyo3::types::PyDateTime with abi3-py37 feature,
                                    // so need this workaround instead of `isinstance(ob, datetime)`.
                                    let bases = ob
                                        .get_type()
                                        .getattr(intern!(py, "__bases__"))
                                        .unwrap()
                                        .iter()
                                        .unwrap();
                                    for base in bases {
                                        let parent_type =
                                            base.unwrap().str().unwrap().to_str().unwrap();
                                        match parent_type {
                                            "<class 'datetime.datetime'>" => {
                                                // `datetime.datetime` is a subclass of `datetime.date`,
                                                // so need to check `datetime.datetime` first
                                                return convert_datetime;
                                            },
                                            "<class 'datetime.date'>" => {
                                                return convert_date;
                                            },
                                            _ => (),
                                        }
                                    }

                                    get_object
                                },
                            }
                        }
                    },
                );

                convert_fn(ob)
            })
        })
    }
}

impl<'s> FromPyObject<'s> for Wrap<Row<'s>> {
    fn extract(ob: &'s PyAny) -> PyResult<Self> {
        let vals = ob.extract::<Vec<Wrap<AnyValue<'s>>>>()?;
        let vals: Vec<AnyValue> = unsafe { std::mem::transmute(vals) };
        Ok(Wrap(Row(vals)))
    }
}

impl FromPyObject<'_> for Wrap<Schema> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let dict = ob.extract::<&PyDict>()?;

        Ok(Wrap(
            dict.iter()
                .map(|(key, val)| {
                    let key = key.extract::<&str>()?;
                    let val = val.extract::<Wrap<DataType>>()?;

                    Ok(Field::new(key, val.0))
                })
                .collect::<PyResult<Schema>>()?,
        ))
    }
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct ObjectValue {
    pub inner: PyObject,
}

impl Hash for ObjectValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let h = Python::with_gil(|py| self.inner.as_ref(py).hash().expect("should be hashable"));
        state.write_isize(h)
    }
}

impl Eq for ObjectValue {}

impl PartialEq for ObjectValue {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| {
            match self
                .inner
                .as_ref(py)
                .rich_compare(other.inner.as_ref(py), CompareOp::Eq)
            {
                Ok(result) => result.is_true().unwrap(),
                Err(_) => false,
            }
        })
    }
}

impl TotalEq for ObjectValue {
    fn tot_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl Display for ObjectValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

#[cfg(feature = "object")]
impl PolarsObject for ObjectValue {
    fn type_name() -> &'static str {
        "object"
    }
}

impl From<PyObject> for ObjectValue {
    fn from(p: PyObject) -> Self {
        Self { inner: p }
    }
}

impl<'a> FromPyObject<'a> for ObjectValue {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        Python::with_gil(|py| {
            Ok(ObjectValue {
                inner: ob.to_object(py),
            })
        })
    }
}

/// # Safety
///
/// The caller is responsible for checking that val is Object otherwise UB
#[cfg(feature = "object")]
impl From<&dyn PolarsObjectSafe> for &ObjectValue {
    fn from(val: &dyn PolarsObjectSafe) -> Self {
        unsafe { &*(val as *const dyn PolarsObjectSafe as *const ObjectValue) }
    }
}

impl ToPyObject for ObjectValue {
    fn to_object(&self, _py: Python) -> PyObject {
        self.inner.clone()
    }
}

impl Default for ObjectValue {
    fn default() -> Self {
        Python::with_gil(|py| ObjectValue { inner: py.None() })
    }
}

impl<'a, T: NativeType + FromPyObject<'a>> FromPyObject<'a> for Wrap<Vec<T>> {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let seq = <PySequence as PyTryFrom>::try_from(obj)?;
        let mut v = Vec::with_capacity(seq.len().unwrap_or(0));
        for item in seq.iter()? {
            v.push(item?.extract::<T>()?);
        }
        Ok(Wrap(v))
    }
}

pub(crate) fn dicts_to_rows(
    records: &PyAny,
    infer_schema_len: Option<usize>,
    schema_columns: PlIndexSet<String>,
) -> PyResult<(Vec<Row>, Vec<String>)> {
    let infer_schema_len = infer_schema_len.map(|n| std::cmp::max(1, n));
    let len = records.len()?;

    let key_names = {
        if !schema_columns.is_empty() {
            schema_columns
        } else {
            let mut inferred_keys = PlIndexSet::new();
            for d in records.iter()?.take(infer_schema_len.unwrap_or(usize::MAX)) {
                let d = d?;
                let d = d.downcast::<PyDict>()?;
                let keys = d.keys();
                for name in keys {
                    let name = name.extract::<String>()?;
                    inferred_keys.insert(name);
                }
            }
            inferred_keys
        }
    };
    let mut rows = Vec::with_capacity(len);

    for d in records.iter()? {
        let d = d?;
        let d = d.downcast::<PyDict>()?;

        let mut row = Vec::with_capacity(key_names.len());
        for k in key_names.iter() {
            let val = match d.get_item(k)? {
                None => AnyValue::Null,
                Some(val) => val.extract::<Wrap<AnyValue>>()?.0,
            };
            row.push(val)
        }
        rows.push(Row(row))
    }
    Ok((rows, key_names.into_iter().collect()))
}

#[cfg(feature = "asof_join")]
impl FromPyObject<'_> for Wrap<AsofStrategy> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "backward" => AsofStrategy::Backward,
            "forward" => AsofStrategy::Forward,
            "nearest" => AsofStrategy::Nearest,
            v => {
                return Err(PyValueError::new_err(format!(
                    "asof `strategy` must be one of {{'backward', 'forward', 'nearest'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<InterpolationMethod> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "linear" => InterpolationMethod::Linear,
            "nearest" => InterpolationMethod::Nearest,
            v => {
                return Err(PyValueError::new_err(format!(
                    "interpolation `method` must be one of {{'linear', 'nearest'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "avro")]
impl FromPyObject<'_> for Wrap<Option<AvroCompression>> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "uncompressed" => None,
            "snappy" => Some(AvroCompression::Snappy),
            "deflate" => Some(AvroCompression::Deflate),
            v => {
                return Err(PyValueError::new_err(format!(
                "avro `compression` must be one of {{'uncompressed', 'snappy', 'deflate'}}, got {v}",
            )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<CategoricalOrdering> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "physical" => CategoricalOrdering::Physical,
            "lexical" => CategoricalOrdering::Lexical,
            v => {
                return Err(PyValueError::new_err(format!(
                    "categorical `ordering` must be one of {{'physical', 'lexical'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<StartBy> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "window" => StartBy::WindowBound,
            "datapoint" => StartBy::DataPoint,
            "monday" => StartBy::Monday,
            "tuesday" => StartBy::Tuesday,
            "wednesday" => StartBy::Wednesday,
            "thursday" => StartBy::Thursday,
            "friday" => StartBy::Friday,
            "saturday" => StartBy::Saturday,
            "sunday" => StartBy::Sunday,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`start_by` must be one of {{'window', 'datapoint', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}}, got {v}",
                )))
            }
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<ClosedWindow> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "left" => ClosedWindow::Left,
            "right" => ClosedWindow::Right,
            "both" => ClosedWindow::Both,
            "none" => ClosedWindow::None,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`closed` must be one of {{'left', 'right', 'both', 'none'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "csv")]
impl FromPyObject<'_> for Wrap<CsvEncoding> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "utf8" => CsvEncoding::Utf8,
            "utf8-lossy" => CsvEncoding::LossyUtf8,
            v => {
                return Err(PyValueError::new_err(format!(
                    "csv `encoding` must be one of {{'utf8', 'utf8-lossy'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "ipc")]
impl FromPyObject<'_> for Wrap<Option<IpcCompression>> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "uncompressed" => None,
            "lz4" => Some(IpcCompression::LZ4),
            "zstd" => Some(IpcCompression::ZSTD),
            v => {
                return Err(PyValueError::new_err(format!(
                    "ipc `compression` must be one of {{'uncompressed', 'lz4', 'zstd'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<JoinType> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "inner" => JoinType::Inner,
            "left" => JoinType::Left,
            "outer" => JoinType::Outer,
            "semi" => JoinType::Semi,
            "anti" => JoinType::Anti,
            #[cfg(feature = "cross_join")]
            "cross" => JoinType::Cross,
            v => {
                return Err(PyValueError::new_err(format!(
                "`how` must be one of {{'inner', 'left', 'outer', 'semi', 'anti', 'cross'}}, got {v}",
            )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<Label> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "left" => Label::Left,
            "right" => Label::Right,
            "datapoint" => Label::DataPoint,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`label` must be one of {{'left', 'right', 'datapoint'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<ListToStructWidthStrategy> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "first_non_null" => ListToStructWidthStrategy::FirstNonNull,
            "max_width" => ListToStructWidthStrategy::MaxWidth,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`n_field_strategy` must be one of {{'first_non_null', 'max_width'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<NullBehavior> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "drop" => NullBehavior::Drop,
            "ignore" => NullBehavior::Ignore,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`null_behavior` must be one of {{'drop', 'ignore'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<NullStrategy> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "ignore" => NullStrategy::Ignore,
            "propagate" => NullStrategy::Propagate,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`null_strategy` must be one of {{'ignore', 'propagate'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "parquet")]
impl FromPyObject<'_> for Wrap<ParallelStrategy> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "auto" => ParallelStrategy::Auto,
            "columns" => ParallelStrategy::Columns,
            "row_groups" => ParallelStrategy::RowGroups,
            "none" => ParallelStrategy::None,
            v => {
                return Err(PyValueError::new_err(format!(
                "`parallel` must be one of {{'auto', 'columns', 'row_groups', 'none'}}, got {v}",
            )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<IndexOrder> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "fortran" => IndexOrder::Fortran,
            "c" => IndexOrder::C,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`order` must be one of {{'fortran', 'c'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<QuantileInterpolOptions> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "lower" => QuantileInterpolOptions::Lower,
            "higher" => QuantileInterpolOptions::Higher,
            "nearest" => QuantileInterpolOptions::Nearest,
            "linear" => QuantileInterpolOptions::Linear,
            "midpoint" => QuantileInterpolOptions::Midpoint,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`interpolation` must be one of {{'lower', 'higher', 'nearest', 'linear', 'midpoint'}}, got {v}",
                )))
            }
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<RankMethod> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "min" => RankMethod::Min,
            "max" => RankMethod::Max,
            "average" => RankMethod::Average,
            "dense" => RankMethod::Dense,
            "ordinal" => RankMethod::Ordinal,
            "random" => RankMethod::Random,
            v => {
                return Err(PyValueError::new_err(format!(
                    "rank `method` must be one of {{'min', 'max', 'average', 'dense', 'ordinal', 'random'}}, got {v}",
                )))
            }
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<TimeUnit> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "ns" => TimeUnit::Nanoseconds,
            "us" => TimeUnit::Microseconds,
            "ms" => TimeUnit::Milliseconds,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`time_unit` must be one of {{'ns', 'us', 'ms'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<UniqueKeepStrategy> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "first" => UniqueKeepStrategy::First,
            "last" => UniqueKeepStrategy::Last,
            "none" => UniqueKeepStrategy::None,
            "any" => UniqueKeepStrategy::Any,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`keep` must be one of {{'first', 'last', 'any', 'none'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "ipc")]
impl FromPyObject<'_> for Wrap<IpcCompression> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "zstd" => IpcCompression::ZSTD,
            "lz4" => IpcCompression::LZ4,
            v => {
                return Err(PyValueError::new_err(format!(
                    "ipc `compression` must be one of {{'zstd', 'lz4'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "search_sorted")]
impl FromPyObject<'_> for Wrap<SearchSortedSide> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "any" => SearchSortedSide::Any,
            "left" => SearchSortedSide::Left,
            "right" => SearchSortedSide::Right,
            v => {
                return Err(PyValueError::new_err(format!(
                    "sorted `side` must be one of {{'any', 'left', 'right'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<WindowMapping> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "group_to_rows" => WindowMapping::GroupsToRows,
            "join" => WindowMapping::Join,
            "explode" => WindowMapping::Explode,
            v => {
                return Err(PyValueError::new_err(format!(
                "`mapping_strategy` must be one of {{'group_to_rows', 'join', 'explode'}}, got {v}",
            )))
            },
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<JoinValidation> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "1:1" => JoinValidation::OneToOne,
            "1:m" => JoinValidation::OneToMany,
            "m:m" => JoinValidation::ManyToMany,
            "m:1" => JoinValidation::ManyToOne,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`validate` must be one of {{'m:m', 'm:1', '1:m', '1:1'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "csv")]
impl FromPyObject<'_> for Wrap<QuoteStyle> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "always" => QuoteStyle::Always,
            "necessary" => QuoteStyle::Necessary,
            "non_numeric" => QuoteStyle::NonNumeric,
            "never" => QuoteStyle::Never,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`quote_style` must be one of {{'always', 'necessary', 'non_numeric', 'never'}}, got {v}",
                )))
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "cloud")]
pub(crate) fn parse_cloud_options(uri: &str, kv: Vec<(String, String)>) -> PyResult<CloudOptions> {
    let out = CloudOptions::from_untyped_config(uri, kv).map_err(PyPolarsErr::from)?;
    Ok(out)
}

#[cfg(feature = "list_sets")]
impl FromPyObject<'_> for Wrap<SetOperation> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "union" => SetOperation::Union,
            "difference" => SetOperation::Difference,
            "intersection" => SetOperation::Intersection,
            "symmetric_difference" => SetOperation::SymmetricDifference,
            v => {
                return Err(PyValueError::new_err(format!(
                    "set operation must be one of {{'union', 'difference', 'intersection', 'symmetric_difference'}}, got {v}",
                )))
            }
        };
        Ok(Wrap(parsed))
    }
}

pub(crate) fn parse_fill_null_strategy(
    strategy: &str,
    limit: FillNullLimit,
) -> PyResult<FillNullStrategy> {
    let parsed = match strategy {
        "forward" => FillNullStrategy::Forward(limit),
        "backward" => FillNullStrategy::Backward(limit),
        "min" => FillNullStrategy::Min,
        "max" => FillNullStrategy::Max,
        "mean" => FillNullStrategy::Mean,
        "zero" => FillNullStrategy::Zero,
        "one" => FillNullStrategy::One,
        e => {
            return Err(PyValueError::new_err(format!(
                "`strategy` must be one of {{'forward', 'backward', 'min', 'max', 'mean', 'zero', 'one'}}, got {e}",
            )))
        }
    };
    Ok(parsed)
}

#[cfg(feature = "parquet")]
pub(crate) fn parse_parquet_compression(
    compression: &str,
    compression_level: Option<i32>,
) -> PyResult<ParquetCompression> {
    let parsed = match compression {
        "uncompressed" => ParquetCompression::Uncompressed,
        "snappy" => ParquetCompression::Snappy,
        "gzip" => ParquetCompression::Gzip(
            compression_level
                .map(|lvl| {
                    GzipLevel::try_new(lvl as u8)
                        .map_err(|e| PyValueError::new_err(format!("{e:?}")))
                })
                .transpose()?,
        ),
        "lzo" => ParquetCompression::Lzo,
        "brotli" => ParquetCompression::Brotli(
            compression_level
                .map(|lvl| {
                    BrotliLevel::try_new(lvl as u32)
                        .map_err(|e| PyValueError::new_err(format!("{e:?}")))
                })
                .transpose()?,
        ),
        "lz4" => ParquetCompression::Lz4Raw,
        "zstd" => ParquetCompression::Zstd(
            compression_level
                .map(|lvl| {
                    ZstdLevel::try_new(lvl)
                        .map_err(|e| PyValueError::new_err(format!("{e:?}")))
                })
                .transpose()?,
        ),
        e => {
            return Err(PyValueError::new_err(format!(
                "parquet `compression` must be one of {{'uncompressed', 'snappy', 'gzip', 'lzo', 'brotli', 'lz4', 'zstd'}}, got {e}",
            )))
        }
    };
    Ok(parsed)
}

pub(crate) fn strings_to_smartstrings<I, S>(container: I) -> Vec<SmartString>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    container.into_iter().map(|s| s.as_ref().into()).collect()
}
