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
use polars_core::prelude::QuantileInterpolOptions;
use polars_core::utils::arrow::types::NativeType;
use pyo3::basic::CompareOp;
use pyo3::conversion::{FromPyObject, IntoPy};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PySequence, PyTuple};
use pyo3::{PyAny, PyResult};
use smartstring::alias::String as SmartString;

use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::lazy::dataframe::PyLazyFrame;
#[cfg(feature = "object")]
use crate::object::OBJECT_NAME;
use crate::prelude::*;
use crate::py_modules::{POLARS, UTILS};
use crate::series::PySeries;

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
    let pydf = obj.getattr("_df")?;
    Ok(pydf.extract::<PyDataFrame>()?.df)
}

pub(crate) fn get_lf(obj: &PyAny) -> PyResult<LazyFrame> {
    let pydf = obj.getattr("_ldf")?;
    Ok(pydf.extract::<PyLazyFrame>()?.ldf)
}

pub(crate) fn get_series(obj: &PyAny) -> PyResult<Series> {
    let pydf = obj.getattr("_s")?;
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

fn decimal_to_digits(v: i128, buf: &mut [u8; 48]) -> usize {
    const ZEROS: i128 = 0x3030_3030_3030_3030_3030_3030_3030_3030;
    if buf.len() < 48 {
        panic!("decimal_to_digits: buffer size < 48");
    }
    let len = lexical_core::write(v, buf).len();
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
        let pl = POLARS.as_ref(py);
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
            }
            AnyValue::Date(v) => {
                let convert = utils.getattr("_to_python_datetime").unwrap();
                let py_date_dtype = pl.getattr("Date").unwrap();
                convert.call1((v, py_date_dtype)).unwrap().into_py(py)
            }
            AnyValue::Datetime(v, tu, tz) => {
                let convert = utils.getattr("_to_python_datetime").unwrap();
                let py_datetime_dtype = pl.getattr("Datetime").unwrap();
                let tu = tu.to_ascii();
                convert
                    .call1((v, py_datetime_dtype, tu, tz.as_ref().map(|s| s.as_str())))
                    .unwrap()
                    .into_py(py)
            }
            AnyValue::Duration(v, tu) => {
                let convert = utils.getattr("_to_python_timedelta").unwrap();
                let tu = tu.to_ascii();
                convert.call1((v, tu)).unwrap().into_py(py)
            }
            AnyValue::Time(v) => {
                let convert = utils.getattr("_to_python_time").unwrap();
                convert.call1((v,)).unwrap().into_py(py)
            }
            AnyValue::List(v) => PySeries::new(v).to_list(),
            ref av @ AnyValue::Struct(_, _, flds) => struct_dict(py, av._iter_struct_av(), flds),
            AnyValue::StructOwned(payload) => struct_dict(py, payload.0.into_iter(), &payload.1),
            #[cfg(feature = "object")]
            AnyValue::Object(v) => {
                let object = v.as_any().downcast_ref::<ObjectValue>().unwrap();
                object.inner.clone()
            }
            #[cfg(feature = "object")]
            AnyValue::ObjectOwned(v) => {
                let object = v.0.as_any().downcast_ref::<ObjectValue>().unwrap();
                object.inner.clone()
            }
            AnyValue::Binary(v) => v.into_py(py),
            AnyValue::BinaryOwned(v) => v.into_py(py),
            AnyValue::Decimal(v, scale) => {
                let convert = utils.getattr("_to_python_decimal").unwrap();
                let mut buf = [0_u8; 48];
                let n_digits = decimal_to_digits(v.abs(), &mut buf);
                let digits = PyTuple::new(py, buf.into_iter().take(n_digits));
                convert
                    .call1((v.is_negative() as u8, digits, n_digits, -(scale as i32)))
                    .unwrap()
                    .into_py(py)
            }
        }
    }
}

impl ToPyObject for Wrap<DataType> {
    fn to_object(&self, py: Python) -> PyObject {
        let pl = POLARS.as_ref(py);

        match &self.0 {
            DataType::Int8 => pl.getattr("Int8").unwrap().into(),
            DataType::Int16 => pl.getattr("Int16").unwrap().into(),
            DataType::Int32 => pl.getattr("Int32").unwrap().into(),
            DataType::Int64 => pl.getattr("Int64").unwrap().into(),
            DataType::UInt8 => pl.getattr("UInt8").unwrap().into(),
            DataType::UInt16 => pl.getattr("UInt16").unwrap().into(),
            DataType::UInt32 => pl.getattr("UInt32").unwrap().into(),
            DataType::UInt64 => pl.getattr("UInt64").unwrap().into(),
            DataType::Float32 => pl.getattr("Float32").unwrap().into(),
            DataType::Float64 => pl.getattr("Float64").unwrap().into(),
            DataType::Decimal(precision, scale) => pl
                .getattr("Decimal")
                .unwrap()
                .call1((*precision, *scale))
                .unwrap()
                .into(),
            DataType::Boolean => pl.getattr("Boolean").unwrap().into(),
            DataType::Utf8 => pl.getattr("Utf8").unwrap().into(),
            DataType::Binary => pl.getattr("Binary").unwrap().into(),
            DataType::List(inner) => {
                let inner = Wrap(*inner.clone()).to_object(py);
                let list_class = pl.getattr("List").unwrap();
                list_class.call1((inner,)).unwrap().into()
            }
            DataType::Date => pl.getattr("Date").unwrap().into(),
            DataType::Datetime(tu, tz) => {
                let datetime_class = pl.getattr("Datetime").unwrap();
                datetime_class
                    .call1((tu.to_ascii(), tz.clone()))
                    .unwrap()
                    .into()
            }
            DataType::Duration(tu) => {
                let duration_class = pl.getattr("Duration").unwrap();
                duration_class.call1((tu.to_ascii(),)).unwrap().into()
            }
            #[cfg(feature = "object")]
            DataType::Object(_) => pl.getattr("Object").unwrap().into(),
            DataType::Categorical(_) => pl.getattr("Categorical").unwrap().into(),
            DataType::Time => pl.getattr("Time").unwrap().into(),
            DataType::Struct(fields) => {
                let field_class = pl.getattr("Field").unwrap();
                let iter = fields.iter().map(|fld| {
                    let name = fld.name().as_str();
                    let dtype = Wrap(fld.data_type().clone()).to_object(py);
                    field_class.call1((name, dtype)).unwrap()
                });
                let fields = PyList::new(py, iter);
                let struct_class = pl.getattr("Struct").unwrap();
                struct_class.call1((fields,)).unwrap().into()
            }
            DataType::Null => pl.getattr("Null").unwrap().into(),
            DataType::Unknown => pl.getattr("Unknown").unwrap().into(),
        }
    }
}

impl FromPyObject<'_> for Wrap<Field> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let name = ob.getattr("name")?.str()?.to_str()?;
        let dtype = ob.getattr("dtype")?.extract::<Wrap<DataType>>()?;
        Ok(Wrap(Field::new(name, dtype.0)))
    }
}

impl FromPyObject<'_> for Wrap<DataType> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let type_name = ob.get_type().name()?;

        let dtype = match type_name {
            "DataTypeClass" => {
                // just the class, not an object
                let name = ob.getattr("__name__")?.str()?.to_str()?;
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
                    "List" => DataType::List(Box::new(DataType::Boolean)),
                    "Null" => DataType::Null,
                    "Unknown" => DataType::Unknown,
                    dt => {
                        return Err(PyValueError::new_err(format!(
                            "{dt} is not a correct polars DataType.",
                        )))
                    }
                }
            }
            "Duration" => {
                let tu = ob.getattr("tu").unwrap();
                let tu = tu.extract::<Wrap<TimeUnit>>()?.0;
                DataType::Duration(tu)
            }
            "Datetime" => {
                let tu = ob.getattr("tu").unwrap();
                let tu = tu.extract::<Wrap<TimeUnit>>()?.0;
                let tz = ob.getattr("tz").unwrap();
                let tz = tz.extract()?;
                DataType::Datetime(tu, tz)
            }
            "Decimal" => {
                let precision = ob.getattr("precision")?.extract()?;
                let scale = ob.getattr("scale")?.extract()?;
                DataType::Decimal(precision, Some(scale))
            }
            "List" => {
                let inner = ob.getattr("inner").unwrap();
                let inner = inner.extract::<Wrap<DataType>>()?;
                DataType::List(Box::new(inner.0))
            }
            "Struct" => {
                let fields = ob.getattr("fields")?;
                let fields = fields
                    .extract::<Vec<Wrap<Field>>>()?
                    .into_iter()
                    .map(|f| f.0)
                    .collect::<Vec<Field>>();
                DataType::Struct(fields)
            }
            dt => {
                return Err(PyTypeError::new_err(format!(
                    "A {dt} object is not a correct polars DataType. \
                    Hint: use the class without instantiating it.",
                )))
            }
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
        let tu = match self.0 {
            TimeUnit::Nanoseconds => "ns",
            TimeUnit::Microseconds => "us",
            TimeUnit::Milliseconds => "ms",
        };
        tu.into_py(py)
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
        let convert = utils.getattr("_to_python_timedelta").unwrap();
        let tu = Wrap(self.0.time_unit()).to_object(py);
        let iter = self
            .0
            .into_iter()
            .map(|opt_v| opt_v.map(|v| convert.call1((v, &tu)).unwrap()));
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DatetimeChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let (pl, utils) = (POLARS.as_ref(py), UTILS.as_ref(py));
        let convert = utils.getattr("_to_python_datetime").unwrap();
        let py_date_dtype = pl.getattr("Datetime").unwrap();
        let tu = Wrap(self.0.time_unit()).to_object(py);
        let tz = self.0.time_zone().to_object(py);
        let iter = self
            .0
            .into_iter()
            .map(|opt_v| opt_v.map(|v| convert.call1((v, py_date_dtype, &tu, &tz)).unwrap()));
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&TimeChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let utils = UTILS.as_ref(py);
        let convert = utils.getattr("_to_python_time").unwrap();
        let iter = self
            .0
            .into_iter()
            .map(|opt_v| opt_v.map(|v| convert.call1((v,)).unwrap()));
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DateChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let (pl, utils) = (POLARS.as_ref(py), UTILS.as_ref(py));
        let convert = utils.getattr("_to_python_datetime").unwrap();
        let py_date_dtype = pl.getattr("Date").unwrap();
        let iter = self
            .0
            .into_iter()
            .map(|opt_v| opt_v.map(|v| convert.call1((v, py_date_dtype)).unwrap()));
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DecimalChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let utils = UTILS.as_ref(py);
        let convert = utils.getattr("_to_python_decimal").unwrap();
        let py_scale = self.0.scale().to_object(py);
        // if we don't know precision, the only safe bet is to set it to 39
        let py_precision = self.0.precision().unwrap_or(39).to_object(py);
        let iter = self.0.into_iter().map(|opt_v| {
            opt_v.map(|v| {
                // TODO! use anyvalue so that we have a single impl.
                let mut buf = [0_u8; 48];
                let n_digits = decimal_to_digits(v.abs(), &mut buf);
                let digits = PyTuple::new(py, buf.into_iter().take(n_digits));
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

fn materialize_list(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
    if ob.is_empty()? {
        Ok(Wrap(AnyValue::List(Series::new_empty("", &DataType::Null))))
    } else {
        let avs = ob.extract::<Wrap<Row>>()?.0 .0;
        // use first `n` values to infer datatype
        // this value is not too large as this will be done with every
        // anyvalue that has to be converted, which can be many
        let n = 25;
        let dtype =
            any_values_to_dtype(&avs[..std::cmp::min(avs.len(), n)]).map_err(PyPolarsErr::from)?;
        let s =
            Series::from_any_values_and_dtype("", &avs, &dtype, true).map_err(PyPolarsErr::from)?;
        Ok(Wrap(AnyValue::List(s)))
    }
}

impl<'s> FromPyObject<'s> for Wrap<AnyValue<'s>> {
    fn extract(ob: &'s PyAny) -> PyResult<Self> {
        let type_name = ob.get_type().name()?;
        if ob.is_instance_of::<PyBool>().unwrap() {
            Ok(AnyValue::Boolean(ob.extract::<bool>().unwrap()).into())
        } else if let Ok(v) = ob.extract::<i64>() {
            Ok(AnyValue::Int64(v).into())
        } else if let Some(Ok(v)) = (type_name != "Decimal").then_some(ob.extract::<f64>()) {
            Ok(AnyValue::Float64(v).into())
        } else if let Ok(v) = ob.extract::<&'s str>() {
            Ok(AnyValue::Utf8(v).into())
        } else if ob.is_none() {
            Ok(AnyValue::Null.into())
        } else if ob.is_instance_of::<PyDict>()? {
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
        } else if ob.is_instance_of::<PyList>()? {
            materialize_list(ob)
        } else if ob.hasattr("_s")? {
            let py_pyseries = ob.getattr("_s").unwrap();
            let series = py_pyseries.extract::<PySeries>().unwrap().series;
            Ok(Wrap(AnyValue::List(series)))
        } else if let Ok(v) = ob.extract::<&'s [u8]>() {
            Ok(AnyValue::Binary(v).into())
        } else {
            match type_name {
                "datetime" => {
                    Python::with_gil(|py| {
                        // windows
                        #[cfg(target_arch = "windows")]
                        let (seconds, microseconds) = {
                            let tzinfo = ob.getattr("tzinfo")?;
                            let dt = if tzinfo.is_none() {
                                let kwargs = PyDict::new(py);
                                kwargs.set_item("tzinfo", py.None())?;
                                let dt = ob.call_method("replace", (), Some(kwargs))?;
                                let localize = UTILS.getattr("_localize").unwrap();
                                localize.call1((dt, "UTC"))
                            } else {
                                ob
                            };
                            let kwargs = PyDict::new(py);
                            kwargs.set_item("microsecond", 0)?;
                            let seconds = dt
                                .call_method("replace", (), Some(kwargs))?
                                .call_method0("timestamp")?;
                            let microseconds = dt.getattr("microsecond")?.extract::<i64>()?;
                            (seconds, microseconds)
                        };
                        // unix
                        #[cfg(not(target_arch = "windows"))]
                        let (seconds, microseconds) = {
                            let datetime = PyModule::import(py, "datetime")?;
                            let timezone = datetime.getattr("timezone")?;
                            let tzinfo = ob.getattr("tzinfo")?;
                            let dt = if tzinfo.is_none() {
                                let kwargs = PyDict::new(py);
                                kwargs.set_item("tzinfo", timezone.getattr("utc")?)?;
                                ob.call_method("replace", (), Some(kwargs))?
                            } else {
                                ob
                            };
                            let kwargs = PyDict::new(py);
                            kwargs.set_item("microsecond", 0)?;
                            let seconds = dt
                                .call_method("replace", (), Some(kwargs))?
                                .call_method0("timestamp")?;
                            let microseconds = dt.getattr("microsecond")?.extract::<i64>()?;
                            (seconds, microseconds)
                        };

                        // s to us
                        let mut v = (seconds.extract::<f64>()? as i64) * 1_000_000;
                        v += microseconds;

                        // choose "us" as that is python's default unit
                        Ok(AnyValue::Datetime(v, TimeUnit::Microseconds, &None).into())
                    })
                }
                "date" => Python::with_gil(|py| {
                    let date = UTILS
                        .as_ref(py)
                        .getattr("_date_to_pl_date")
                        .unwrap()
                        .call1((ob,))
                        .unwrap();
                    let v = date.extract::<i32>().unwrap();
                    Ok(Wrap(AnyValue::Date(v)))
                }),
                "timedelta" => Python::with_gil(|py| {
                    let td = UTILS
                        .as_ref(py)
                        .getattr("_timedelta_to_pl_timedelta")
                        .unwrap()
                        .call1((ob, "us"))
                        .unwrap();
                    let v = td.extract::<i64>().unwrap();
                    Ok(Wrap(AnyValue::Duration(v, TimeUnit::Microseconds)))
                }),
                "time" => Python::with_gil(|py| {
                    let time = UTILS
                        .as_ref(py)
                        .getattr("_time_to_pl_time")
                        .unwrap()
                        .call1((ob,))
                        .unwrap();
                    let v = time.extract::<i64>().unwrap();
                    Ok(Wrap(AnyValue::Time(v)))
                }),
                "Decimal" => {
                    let (sign, digits, exp): (i8, Vec<u8>, i32) =
                        ob.call_method0("as_tuple").unwrap().extract().unwrap();
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
                "range" => materialize_list(ob),
                _ => Err(PyErr::from(PyPolarsErr::Other(format!(
                    "object type not supported {ob:?}",
                )))),
            }
        }
    }
}

impl<'s> FromPyObject<'s> for Wrap<Row<'s>> {
    fn extract(ob: &'s PyAny) -> PyResult<Self> {
        let vals = ob.extract::<Vec<Wrap<AnyValue<'s>>>>()?;
        let vals: Vec<AnyValue> = unsafe { std::mem::transmute(vals) };
        Ok(Wrap(Row(vals)))
    }
}

pub(crate) trait ToSeries {
    fn to_series(self) -> Vec<Series>;
}

impl ToSeries for Vec<PySeries> {
    fn to_series(self) -> Vec<Series> {
        // Safety:
        // transparent repr
        unsafe { std::mem::transmute(self) }
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
            let val = match d.get_item(k) {
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
            v => {
                return Err(PyValueError::new_err(format!(
                    "strategy must be one of {{'backward', 'forward'}}, got {v}",
                )))
            }
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
                    "method must be one of {{'linear', 'nearest'}}, got {v}",
                )))
            }
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
                    "compression must be one of {{'uncompressed', 'snappy', 'deflate'}}, got {v}",
                )))
            }
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
                    "ordering must be one of {{'physical', 'lexical'}}, got {v}",
                )))
            }
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
            v => {
                return Err(PyValueError::new_err(format!(
                    "closed must be one of {{'window', 'datapoint', 'monday'}}, got {v}",
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
                    "closed must be one of {{'left', 'right', 'both', 'none'}}, got {v}",
                )))
            }
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<CsvEncoding> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "utf8" => CsvEncoding::Utf8,
            "utf8-lossy" => CsvEncoding::LossyUtf8,
            v => {
                return Err(PyValueError::new_err(format!(
                    "encoding must be one of {{'utf8', 'utf8-lossy'}}, got {v}",
                )))
            }
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
                    "compression must be one of {{'uncompressed', 'lz4', 'zstd'}}, got {v}",
                )))
            }
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
                "how must be one of {{'inner', 'left', 'outer', 'semi', 'anti', 'cross'}}, got {v}",
            )))
            }
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
                    "n_field_strategy must be one of {{'first_non_null', 'max_width'}}, got {v}",
                )))
            }
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
                    "null behavior must be one of {{'drop', 'ignore'}}, got {v}",
                )))
            }
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
                    "null strategy must be one of {{'ignore', 'propagate'}}, got {v}",
                )))
            }
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
                    "parallel must be one of {{'auto', 'columns', 'row_groups', 'none'}}, got {v}",
                )))
            }
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
                    "interpolation must be one of {{'lower', 'higher', 'nearest', 'linear', 'midpoint'}}, got {v}",
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
                    "method must be one of {{'min', 'max', 'average', 'dense', 'ordinal', 'random'}}, got {v}",
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
                    "time unit must be one of {{'ns', 'us', 'ms'}}, got {v}",
                )))
            }
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
                    "keep must be one of {{'first', 'last', 'any', 'none'}}, got {v}",
                )))
            }
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<IpcCompression> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "zstd" => IpcCompression::ZSTD,
            "lz4" => IpcCompression::LZ4,
            v => {
                return Err(PyValueError::new_err(format!(
                    "compression must be one of {{'zstd', 'lz4'}}, got {v}",
                )))
            }
        };
        Ok(Wrap(parsed))
    }
}

impl FromPyObject<'_> for Wrap<SearchSortedSide> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "any" => SearchSortedSide::Any,
            "left" => SearchSortedSide::Left,
            "right" => SearchSortedSide::Right,
            v => {
                return Err(PyValueError::new_err(format!(
                    "side must be one of {{'any', 'left', 'right'}}, got {v}",
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
                "strategy must be one of {{'forward', 'backward', 'min', 'max', 'mean', 'zero', 'one'}}, got {e}",
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
                "compression must be one of {{'uncompressed', 'snappy', 'gzip', 'lzo', 'brotli', 'lz4', 'zstd'}}, got {e}",
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
