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
use polars_core::prelude::QuantileInterpolOptions;
use polars_core::utils::arrow::types::NativeType;
use pyo3::basic::CompareOp;
use pyo3::conversion::{FromPyObject, IntoPy};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PySequence};
use pyo3::{PyAny, PyResult};

use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::lazy::dataframe::PyLazyFrame;
use crate::prelude::*;
use crate::py_modules::POLARS;
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

pub(crate) fn get_pyseq(obj: &PyAny) -> PyResult<(&PySequence, usize)> {
    let seq = <PySequence as PyTryFrom>::try_from(obj)?;
    let len = seq.len()? as usize;
    Ok((seq, len))
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
        let (seq, len) = get_pyseq(obj)?;
        let mut builder = PrimitiveChunkedBuilder::new("", len);

        for res in seq.iter()? {
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
        let (seq, len) = get_pyseq(obj)?;
        let mut builder = BooleanChunkedBuilder::new("", len);

        for res in seq.iter()? {
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
        let (seq, len) = get_pyseq(obj)?;
        let mut builder = Utf8ChunkedBuilder::new("", len, len * 25);

        for res in seq.iter()? {
            let item = res?;
            match item.extract::<&str>() {
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

fn struct_dict(py: Python, vals: Vec<AnyValue>, flds: &[Field]) -> PyObject {
    let dict = PyDict::new(py);
    for (fld, val) in flds.iter().zip(vals) {
        dict.set_item(fld.name(), Wrap(val)).unwrap()
    }
    dict.into_py(py)
}

impl IntoPy<PyObject> for Wrap<AnyValue<'_>> {
    fn into_py(self, py: Python) -> PyObject {
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
            AnyValue::Categorical(idx, rev) => {
                let s = rev.get(idx);
                s.into_py(py)
            }
            AnyValue::Date(v) => {
                let pl = PyModule::import(py, "polars").unwrap();
                let utils = pl.getattr("utils").unwrap();
                let convert = utils.getattr("_to_python_datetime").unwrap();
                let py_date_dtype = pl.getattr("Date").unwrap();
                convert.call1((v, py_date_dtype)).unwrap().into_py(py)
            }
            AnyValue::Datetime(v, tu, tz) => {
                let pl = PyModule::import(py, "polars").unwrap();
                let utils = pl.getattr("utils").unwrap();
                let convert = utils.getattr("_to_python_datetime").unwrap();
                let py_datetime_dtype = pl.getattr("Datetime").unwrap();
                let tu = match tu {
                    TimeUnit::Nanoseconds => "ns",
                    TimeUnit::Microseconds => "us",
                    TimeUnit::Milliseconds => "ms",
                };
                convert
                    .call1((v, py_datetime_dtype, tu, tz.as_ref().map(|s| s.as_str())))
                    .unwrap()
                    .into_py(py)
            }
            AnyValue::Duration(v, tu) => {
                let pl = PyModule::import(py, "polars").unwrap();
                let utils = pl.getattr("utils").unwrap();
                let convert = utils.getattr("_to_python_timedelta").unwrap();
                match tu {
                    TimeUnit::Nanoseconds => convert.call1((v, "ns")).unwrap().into_py(py),
                    TimeUnit::Microseconds => convert.call1((v, "us")).unwrap().into_py(py),
                    TimeUnit::Milliseconds => convert.call1((v, "ms")).unwrap().into_py(py),
                }
            }
            AnyValue::Time(v) => v.into_py(py),
            AnyValue::List(v) => PySeries::new(v).to_list(),
            AnyValue::Struct(vals, flds) => struct_dict(py, vals, flds),
            AnyValue::StructOwned(payload) => struct_dict(py, payload.0, &payload.1),
            #[cfg(feature = "object")]
            AnyValue::Object(v) => {
                let s = format!("{}", v);
                s.into_py(py)
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
            DataType::Boolean => pl.getattr("Boolean").unwrap().into(),
            DataType::Utf8 => pl.getattr("Utf8").unwrap().into(),
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
                    let name = fld.name().clone();
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
            "type" => {
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
                    "Boolean" => DataType::Boolean,
                    "Categorical" => DataType::Categorical(None),
                    "Date" => DataType::Date,
                    "Datetime" => DataType::Datetime(TimeUnit::Microseconds, None),
                    "Time" => DataType::Time,
                    "Duration" => DataType::Duration(TimeUnit::Microseconds),
                    "Float32" => DataType::Float32,
                    "Float64" => DataType::Float64,
                    #[cfg(feature = "object")]
                    "Object" => DataType::Object("unknown"),
                    "List" => DataType::List(Box::new(DataType::Boolean)),
                    "Null" => DataType::Null,
                    "Unknown" => DataType::Unknown,
                    dt => {
                        return Err(PyValueError::new_err(format!(
                            "{} is not a correct polars DataType.",
                            dt
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
                return Err(PyValueError::new_err(format!(
                    "A {} object is not a correct polars DataType.\
                 Hint: use the class without instantiating it.",
                    dt
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

impl ToPyObject for Wrap<&StructChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let s = self.0.clone().into_series();
        // todo! iterate its chunks and flatten.
        // make series::iter() accept a chunk index.
        let s = s.rechunk();
        let iter = s.iter().map(|av| {
            if let AnyValue::Struct(vals, flds) = av {
                struct_dict(py, vals, flds)
            } else {
                unreachable!()
            }
        });

        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DurationChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let pl = PyModule::import(py, "polars").unwrap();
        let pl_utils = pl.getattr("utils").unwrap();
        let convert = pl_utils.getattr("_to_python_timedelta").unwrap();

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
        let pl = PyModule::import(py, "polars").unwrap();
        let utils = pl.getattr("utils").unwrap();
        let convert = utils.getattr("_to_python_datetime").unwrap();
        let py_date_dtype = pl.getattr("Datetime").unwrap();

        let tu = Wrap(self.0.time_unit()).to_object(py);

        let iter = self
            .0
            .into_iter()
            .map(|opt_v| opt_v.map(|v| convert.call1((v, py_date_dtype, &tu)).unwrap()));
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&TimeChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let pl = PyModule::import(py, "polars").unwrap();
        let pl_utils = pl.getattr("utils").unwrap();
        let convert = pl_utils.getattr("_to_python_time").unwrap();
        let iter = self
            .0
            .into_iter()
            .map(|opt_v| opt_v.map(|v| convert.call1((v,)).unwrap()));
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DateChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let pl = PyModule::import(py, "polars").unwrap();
        let utils = pl.getattr("utils").unwrap();
        let convert = utils.getattr("_to_python_datetime").unwrap();
        let py_date_dtype = pl.getattr("Date").unwrap();

        let iter = self
            .0
            .into_iter()
            .map(|opt_v| opt_v.map(|v| convert.call1((v, py_date_dtype)).unwrap()));
        PyList::new(py, iter).into_py(py)
    }
}

impl<'s> FromPyObject<'s> for Wrap<AnyValue<'s>> {
    fn extract(ob: &'s PyAny) -> PyResult<Self> {
        if ob.is_instance_of::<PyBool>().unwrap() {
            Ok(AnyValue::Boolean(ob.extract::<bool>().unwrap()).into())
        } else if let Ok(v) = ob.extract::<i64>() {
            Ok(AnyValue::Int64(v).into())
        } else if let Ok(v) = ob.extract::<f64>() {
            Ok(AnyValue::Float64(v).into())
        } else if let Ok(v) = ob.extract::<&'s str>() {
            Ok(AnyValue::Utf8(v).into())
        } else if ob.get_type().name()?.contains("datetime") {
            Python::with_gil(|py| {
                // windows
                #[cfg(target_arch = "windows")]
                {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("tzinfo", py.None())?;
                    let dt = ob.call_method("replace", (), Some(kwargs))?;

                    let pypolars = PyModule::import(py, "polars").unwrap();
                    let localize = pypolars
                        .getattr("utils")
                        .unwrap()
                        .getattr("_localize")
                        .unwrap();
                    let loc_tz = localize.call1((dt, "UTC"));

                    loc_tz.call_method0("timestamp")?;
                    // s to us
                    let v = (ts.extract::<f64>()? * 1000_000.0) as i64;
                    Ok(AnyValue::Datetime(v, TimeUnit::Microseconds, &None).into())
                }
                // unix
                #[cfg(not(target_arch = "windows"))]
                {
                    let datetime = PyModule::import(py, "datetime")?;
                    let timezone = datetime.getattr("timezone")?;
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("tzinfo", timezone.getattr("utc")?)?;
                    let dt = ob.call_method("replace", (), Some(kwargs))?;
                    let ts = dt.call_method0("timestamp")?;
                    // s to us
                    let v = (ts.extract::<f64>()? * 1_000_000.0) as i64;
                    // we choose us as that is pythons default unit
                    Ok(AnyValue::Datetime(v, TimeUnit::Microseconds, &None).into())
                }
            })
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
            if ob.is_empty()? {
                Ok(Wrap(AnyValue::List(Series::new_empty("", &DataType::Null))))
            } else {
                let avs = ob.extract::<Wrap<Row>>()?.0;
                let s = Series::new("", &avs.0);
                Ok(Wrap(AnyValue::List(s)))
            }
        } else if ob.hasattr("_s")? {
            let py_pyseries = ob.getattr("_s").unwrap();
            let series = py_pyseries.extract::<PySeries>().unwrap().series;
            Ok(Wrap(AnyValue::List(series)))
        } else if ob.get_type().name()?.contains("date") {
            Python::with_gil(|py| {
                let date = py_modules::UTILS
                    .getattr(py, "_date_to_pl_date")
                    .unwrap()
                    .call1(py, (ob,))
                    .unwrap();
                let v = date.extract::<i32>(py).unwrap();
                Ok(Wrap(AnyValue::Date(v)))
            })
        } else if ob.get_type().name()?.contains("timedelta") {
            Python::with_gil(|py| {
                let td = py_modules::UTILS
                    .getattr(py, "_timedelta_to_pl_timedelta")
                    .unwrap()
                    .call1(py, (ob, "us"))
                    .unwrap();
                let v = td.extract::<i64>(py).unwrap();
                Ok(Wrap(AnyValue::Duration(v, TimeUnit::Microseconds)))
            })
        } else {
            Err(PyErr::from(PyPolarsErr::Other(format!(
                "row type not supported {:?}",
                ob
            ))))
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
        let mut v = Vec::with_capacity(seq.len().unwrap_or(0) as usize);
        for item in seq.iter()? {
            v.push(item?.extract::<T>()?);
        }
        Ok(Wrap(v))
    }
}

pub(crate) fn dicts_to_rows(records: &PyAny) -> PyResult<(Vec<Row>, Vec<String>)> {
    let (dicts, len) = get_pyseq(records)?;
    let mut rows = Vec::with_capacity(len);

    let mut iter = dicts.iter()?;
    let d = iter.next().unwrap()?;
    let d = d.downcast::<PyDict>()?;
    let vals = d.values();
    let keys_first = d.keys().extract::<Vec<String>>()?;
    let row = vals.extract::<Wrap<Row>>()?.0;
    rows.push(row);

    let keys = d.keys();
    let width = keys.len();

    for d in iter {
        let d = d?;
        let d = d.downcast::<PyDict>()?;

        let mut row = Vec::with_capacity(width);

        for k in keys {
            let val = d.get_item(k).unwrap();
            let val = val.extract::<Wrap<AnyValue>>()?.0;
            row.push(val)
        }
        rows.push(Row(row))
    }
    Ok((rows, keys_first))
}

#[cfg(feature = "asof_join")]
impl FromPyObject<'_> for Wrap<AsofStrategy> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let parsed = match ob.extract::<&str>()? {
            "backward" => AsofStrategy::Backward,
            "forward" => AsofStrategy::Forward,
            v => {
                return Err(PyValueError::new_err(format!(
                    "strategy must be one of {{'backward', 'forward'}}, got {}",
                    v
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
                    "compression must be one of {{'uncompressed', 'snappy', 'deflate'}}, got {}",
                    v
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
                    "ordering must be one of {{'physical', 'lexical'}}, got {}",
                    v
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
                    "closed must be one of {{'left', 'right', 'both', 'none'}}, got {}",
                    v
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
                    "encoding must be one of {{'utf8', 'utf8-lossy'}}, got {}",
                    v
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
                    "compression must be one of {{'uncompressed', 'lz4', 'zstd'}}, got {}",
                    v
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
                "how must be one of {{'inner', 'left', 'outer', 'semi', 'anti', 'cross'}}, got {}",
                v
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
                    "n_field_strategy must be one of {{'first_non_null', 'max_width'}}, got {}",
                    v
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
                    "null behavior must be one of {{'drop', 'ignore'}}, got {}",
                    v
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
                    "null strategy must be one of {{'ignore', 'propagate'}}, got {}",
                    v
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
                    "parallel must be one of {{'auto', 'columns', 'row_groups', 'none'}}, got {}",
                    v
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
                    "interpolation must be one of {{'lower', 'higher', 'nearest', 'linear', 'midpoint'}}, got {}",
                    v
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
                    "method must be one of {{'min', 'max', 'average', 'dense', 'ordinal', 'random'}}, got {}",
                    v
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
                    "time unit must be one of {{'ns', 'us', 'ms'}}, got {}",
                    v
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
            v => {
                return Err(PyValueError::new_err(format!(
                    "keep must be one of {{'first', 'last'}}, got {}",
                    v
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
                "strategy must be one of {{'forward', 'backward', 'min', 'max', 'mean', 'zero', 'one'}}, got {}",
                e,
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
                        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
                })
                .transpose()?,
        ),
        "lzo" => ParquetCompression::Lzo,
        "brotli" => ParquetCompression::Brotli(
            compression_level
                .map(|lvl| {
                    BrotliLevel::try_new(lvl as u32)
                        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
                })
                .transpose()?,
        ),
        "lz4" => ParquetCompression::Lz4Raw,
        "zstd" => ParquetCompression::Zstd(
            compression_level
                .map(|lvl| {
                    ZstdLevel::try_new(lvl as i32)
                        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
                })
                .transpose()?,
        ),
        e => {
            return Err(PyValueError::new_err(format!(
                "compression must be one of {{'uncompressed', 'snappy', 'gzip', 'lzo', 'brotli', 'lz4', 'zstd'}}, got {}",
                e
            )))
        }
    };
    Ok(parsed)
}
