use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsEr;
use crate::lazy::dataframe::PyLazyFrame;
use crate::prelude::*;
use crate::series::PySeries;
use polars::chunked_array::object::PolarsObjectSafe;
use polars::frame::row::Row;
use polars::frame::NullStrategy;
use polars::prelude::AnyValue;
use polars::series::ops::NullBehavior;
use polars_core::utils::arrow::types::NativeType;
use pyo3::basic::CompareOp;
use pyo3::conversion::{FromPyObject, IntoPy};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySequence};
use pyo3::{PyAny, PyResult};
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

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
            Ok(Wrap(NullValues::AllColumns(s)))
        } else if let Ok(s) = ob.extract::<Vec<String>>() {
            Ok(Wrap(NullValues::Columns(s)))
        } else if let Ok(s) = ob.extract::<Vec<(String, String)>>() {
            Ok(Wrap(NullValues::Named(s)))
        } else {
            Err(
                PyPolarsEr::Other("could not extract value from null_values argument".into())
                    .into(),
            )
        }
    }
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
            AnyValue::Categorical(idx, rev) => {
                let s = rev.get(idx);
                s.into_py(py)
            }
            AnyValue::Date(v) => {
                let pl = PyModule::import(py, "polars").unwrap();
                let pli = pl.getattr("internals").unwrap();
                let m_series = pli.getattr("series").unwrap();
                let convert = m_series.getattr("_to_python_datetime").unwrap();
                let py_date_dtype = pl.getattr("Date").unwrap();
                convert.call1((v, py_date_dtype)).unwrap().into_py(py)
            }
            AnyValue::Datetime(v) => {
                let pl = PyModule::import(py, "polars").unwrap();
                let pli = pl.getattr("internals").unwrap();
                let m_series = pli.getattr("series").unwrap();
                let convert = m_series.getattr("_to_python_datetime").unwrap();
                let py_datetime_dtype = pl.getattr("Datetime").unwrap();
                convert.call1((v, py_datetime_dtype)).unwrap().into_py(py)
            }
            AnyValue::Time(v) => v.into_py(py),
            AnyValue::List(v) => {
                let pypolars = PyModule::import(py, "polars").unwrap();
                let pyseries = PySeries::new(v);
                let python_series_wrapper = pypolars
                    .getattr("wrap_s")
                    .unwrap()
                    .call1((pyseries,))
                    .unwrap();
                python_series_wrapper.into()
            }
            AnyValue::Object(v) => {
                let s = format!("{}", v);
                s.into_py(py)
            }
        }
    }
}

impl ToPyObject for Wrap<DataType> {
    fn to_object(&self, py: Python) -> PyObject {
        let pl = PyModule::import(py, "polars").unwrap();

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
            DataType::List(_) => pl.getattr("List").unwrap().into(),
            DataType::Date => pl.getattr("Date").unwrap().into(),
            DataType::Datetime => pl.getattr("Datetime").unwrap().into(),
            DataType::Object(_) => pl.getattr("Object").unwrap().into(),
            DataType::Categorical => pl.getattr("Categorical").unwrap().into(),
            DataType::Time => pl.getattr("Time").unwrap().into(),
            dt => panic!("{} not supported", dt),
        }
    }
}

impl FromPyObject<'_> for Wrap<ClosedWindow> {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        let s = ob.extract::<&str>()?;
        Ok(Wrap(match s {
            "none" => ClosedWindow::None,
            "both" => ClosedWindow::Both,
            "left" => ClosedWindow::Left,
            "right" => ClosedWindow::Right,
            _ => panic!("{}", "closed should be any of {'none', 'left', 'right'}"),
        }))
    }
}

impl FromPyObject<'_> for Wrap<DataType> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let dtype = match ob.repr().unwrap().to_str().unwrap() {
            "<class 'polars.datatypes.UInt8'>" => DataType::UInt8,
            "<class 'polars.datatypes.UInt16'>" => DataType::UInt16,
            "<class 'polars.datatypes.UInt32'>" => DataType::UInt32,
            "<class 'polars.datatypes.UInt64'>" => DataType::UInt64,
            "<class 'polars.datatypes.Int8'>" => DataType::Int8,
            "<class 'polars.datatypes.Int16'>" => DataType::Int16,
            "<class 'polars.datatypes.Int32'>" => DataType::Int32,
            "<class 'polars.datatypes.Int64'>" => DataType::Int64,
            "<class 'polars.datatypes.Utf8'>" => DataType::Utf8,
            "<class 'polars.datatypes.List'>" => DataType::List(Box::new(DataType::Boolean)),
            "<class 'polars.datatypes.Boolean'>" => DataType::Boolean,
            "<class 'polars.datatypes.Categorical'>" => DataType::Categorical,
            "<class 'polars.datatypes.Date'>" => DataType::Date,
            "<class 'polars.datatypes.Datetime'>" => DataType::Datetime,
            "<class 'polars.datatypes.Float32'>" => DataType::Float32,
            "<class 'polars.datatypes.Float64'>" => DataType::Float64,
            "<class 'polars.datatypes.Object'>" => DataType::Object("unknown"),
            dt => panic!(
                "{} not expected in python dtype to rust dtype conversion",
                dt
            ),
        };
        Ok(Wrap(dtype))
    }
}

impl ToPyObject for Wrap<AnyValue<'_>> {
    fn to_object(&self, py: Python) -> PyObject {
        self.clone().into_py(py)
    }
}

impl ToPyObject for Wrap<&DatetimeChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let pl = PyModule::import(py, "polars").unwrap();
        let pli = pl.getattr("internals").unwrap();
        let m_series = pli.getattr("series").unwrap();
        let convert = m_series.getattr("_to_python_datetime").unwrap();
        let py_date_dtype = pl.getattr("Datetime").unwrap();

        let iter = self
            .0
            .into_iter()
            .map(|opt_v| opt_v.map(|v| convert.call1((v, py_date_dtype)).unwrap()));
        PyList::new(py, iter).into_py(py)
    }
}

impl ToPyObject for Wrap<&DateChunked> {
    fn to_object(&self, py: Python) -> PyObject {
        let pl = PyModule::import(py, "polars").unwrap();
        let pli = pl.getattr("internals").unwrap();
        let m_series = pli.getattr("series").unwrap();
        let convert = m_series.getattr("_to_python_datetime").unwrap();
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
        if let Ok(v) = ob.extract::<i64>() {
            Ok(AnyValue::Int64(v).into())
        } else if let Ok(v) = ob.extract::<f64>() {
            Ok(AnyValue::Float64(v).into())
        } else if let Ok(v) = ob.extract::<&'s str>() {
            Ok(AnyValue::Utf8(v).into())
        } else if let Ok(v) = ob.extract::<bool>() {
            Ok(AnyValue::Boolean(v).into())
        } else if ob.get_type().name()?.contains("datetime") {
            let gil = Python::acquire_gil();
            let py = gil.python();

            // windows
            #[cfg(target_arch = "windows")]
            {
                let kwargs = PyDict::new(py);
                kwargs.set_item("tzinfo", py.None())?;
                let dt = ob.call_method("replace", (), Some(kwargs))?;

                let pytz = PyModule::import(py, "pytz")?;
                let tz = pytz.call_method("timezone", ("UTC",), None)?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("is_dst", py.None())?;
                let loc_tz = tz.call_method("localize", (dt,), Some(kwargs))?;
                loc_tz.call_method0("timestamp")?;
                // s to ms
                let v = ts.extract::<f64>()? as i64;
                Ok(AnyValue::Datetime(v * 1000).into())
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
                // s to ms
                let v = ts.extract::<f64>()? as i64;
                Ok(AnyValue::Datetime(v * 1000).into())
            }
        } else if ob.is_none() {
            Ok(AnyValue::Null.into())
        } else {
            Err(PyErr::from(PyPolarsEr::Other(format!(
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

#[derive(Clone, Debug)]
pub struct ObjectValue {
    pub inner: PyObject,
}

impl Hash for ObjectValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let gil = Python::acquire_gil();
        let python = gil.python();
        let h = self
            .inner
            .as_ref(python)
            .hash()
            .expect("should be hashable");
        state.write_isize(h)
    }
}

impl Eq for ObjectValue {}

impl PartialEq for ObjectValue {
    fn eq(&self, other: &Self) -> bool {
        let gil = Python::acquire_gil();
        let py = gil.python();
        match self
            .inner
            .as_ref(py)
            .rich_compare(other.inner.as_ref(py), CompareOp::Eq)
        {
            Ok(result) => result.is_true().unwrap(),
            Err(_) => false,
        }
    }
}

impl Display for ObjectValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

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
        let gil = Python::acquire_gil();
        let python = gil.python();
        Ok(ObjectValue {
            inner: ob.to_object(python),
        })
    }
}

/// # Safety
///
/// The caller is responsible for checking that val is Object otherwise UB
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
        let gil = Python::acquire_gil();
        let python = gil.python();
        ObjectValue {
            inner: python.None(),
        }
    }
}

impl<'a, T: NativeType + FromPyObject<'a>> FromPyObject<'a> for Wrap<AlignedVec<T>> {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let seq = <PySequence as PyTryFrom>::try_from(obj)?;
        let mut v = AlignedVec::with_capacity(seq.len().unwrap_or(0) as usize);
        for item in seq.iter()? {
            v.push(item?.extract::<T>()?);
        }
        Ok(Wrap(v))
    }
}

pub(crate) fn str_to_null_behavior(null_behavior: &str) -> PyResult<NullBehavior> {
    let null_behavior = match null_behavior {
        "drop" => NullBehavior::Drop,
        "ignore" => NullBehavior::Ignore,
        _ => {
            return Err(PyValueError::new_err(
                "use one of 'drop', 'ignore'".to_string(),
            ))
        }
    };
    Ok(null_behavior)
}

pub(crate) fn str_to_rankmethod(method: &str) -> PyResult<RankMethod> {
    let method = match method {
        "min" => RankMethod::Min,
        "max" => RankMethod::Max,
        "average" => RankMethod::Average,
        "dense" => RankMethod::Dense,
        "ordinal" => RankMethod::Ordinal,
        "random" => RankMethod::Random,
        _ => {
            return Err(PyValueError::new_err(
                "use one of 'avg, min, max, dense, ordinal'".to_string(),
            ))
        }
    };
    Ok(method)
}

pub(crate) fn str_to_null_strategy(strategy: &str) -> PyResult<NullStrategy> {
    let strategy = match strategy {
        "ignore" => NullStrategy::Ignore,
        "propagate" => NullStrategy::Propagate,
        _ => {
            return Err(PyValueError::new_err(
                "use one of 'ignore', 'propagate'".to_string(),
            ))
        }
    };
    Ok(strategy)
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

pub(crate) fn parse_strategy(strat: &str) -> FillNullStrategy {
    match strat {
        "backward" => FillNullStrategy::Backward,
        "forward" => FillNullStrategy::Forward,
        "min" => FillNullStrategy::Min,
        "max" => FillNullStrategy::Max,
        "mean" => FillNullStrategy::Mean,
        "zero" => FillNullStrategy::Zero,
        "one" => FillNullStrategy::One,
        s => panic!("Strategy {} not supported", s),
    }
}
