use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::lazy::dataframe::PyLazyFrame;
use crate::prelude::*;
use crate::py_modules::POLARS;
use crate::series::PySeries;
use polars::chunked_array::object::PolarsObjectSafe;
use polars::frame::row::Row;
use polars::frame::{groupby::PivotAgg, NullStrategy};
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
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

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

impl<'a> FromPyObject<'a> for Wrap<PivotAgg> {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        match ob.extract::<&str>()? {
            "sum" => Ok(Wrap(PivotAgg::Sum)),
            "min" => Ok(Wrap(PivotAgg::Min)),
            "max" => Ok(Wrap(PivotAgg::Max)),
            "first" => Ok(Wrap(PivotAgg::First)),
            "mean" => Ok(Wrap(PivotAgg::Mean)),
            "median" => Ok(Wrap(PivotAgg::Median)),
            "count" => Ok(Wrap(PivotAgg::Count)),
            "last" => Ok(Wrap(PivotAgg::Last)),
            s => panic!("aggregation {} is not supported", s),
        }
    }
}

impl<'a> FromPyObject<'a> for Wrap<UniqueKeepStrategy> {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        match ob.extract::<&str>()? {
            "first" => Ok(Wrap(UniqueKeepStrategy::First)),
            "last" => Ok(Wrap(UniqueKeepStrategy::Last)),
            s => panic!("keep strategy {} is not supported", s),
        }
    }
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
                let pli = pl.getattr("internals").unwrap();
                let m_series = pli.getattr("series").unwrap();
                let convert = m_series.getattr("_to_python_datetime").unwrap();
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

impl FromPyObject<'_> for Wrap<QuantileInterpolOptions> {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        let s = ob.extract::<&str>()?;
        Ok(Wrap(match s {
            "lower" => QuantileInterpolOptions::Lower,
            "higher" => QuantileInterpolOptions::Higher,
            "nearest" => QuantileInterpolOptions::Nearest,
            "linear" => QuantileInterpolOptions::Linear,
            "midpoint" => QuantileInterpolOptions::Midpoint,
            _ => panic!("{}", "interpolation should be any of {'lower', 'higher', 'nearest', 'linear', 'midpoint'}"),
        }))
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
                    "Object" => DataType::Object("unknown"),
                    "List" => DataType::List(Box::new(DataType::Boolean)),
                    "Null" => DataType::Null,
                    dt => panic!("{} not expected as Python type for dtype conversion", dt),
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
                panic!(
                    "{} not expected in Python dtype to Rust dtype conversion",
                    dt
                )
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

impl FromPyObject<'_> for Wrap<TimeUnit> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let unit = match ob.str()?.to_str()? {
            "ns" => TimeUnit::Nanoseconds,
            "us" => TimeUnit::Microseconds,
            "ms" => TimeUnit::Milliseconds,
            _ => return Err(PyValueError::new_err("expected one of {'ns', 'us', 'ms'}")),
        };
        Ok(Wrap(unit))
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
        let pli = pl.getattr("internals").unwrap();
        let m_series = pli.getattr("series").unwrap();
        let convert = m_series.getattr("_to_python_datetime").unwrap();
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
        if ob.is_instance_of::<PyBool>().unwrap() {
            Ok(AnyValue::Boolean(ob.extract::<bool>().unwrap()).into())
        } else if let Ok(v) = ob.extract::<i64>() {
            Ok(AnyValue::Int64(v).into())
        } else if let Ok(v) = ob.extract::<f64>() {
            Ok(AnyValue::Float64(v).into())
        } else if let Ok(v) = ob.extract::<&'s str>() {
            Ok(AnyValue::Utf8(v).into())
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
            let gil = Python::acquire_gil();
            let py = gil.python();
            let date = py_modules::UTILS
                .getattr(py, "_date_to_pl_date")
                .unwrap()
                .call1(py, (ob,))
                .unwrap();
            let v = date.extract::<i32>(py).unwrap();
            Ok(Wrap(AnyValue::Date(v)))
        } else if ob.get_type().name()?.contains("timedelta") {
            let gil = Python::acquire_gil();
            let py = gil.python();
            let td = py_modules::UTILS
                .getattr(py, "_timedelta_to_pl_timedelta")
                .unwrap()
                .call1(py, (ob, "us"))
                .unwrap();
            let v = td.extract::<i64>(py).unwrap();
            Ok(Wrap(AnyValue::Duration(v, TimeUnit::Microseconds)))
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

pub(crate) fn parse_strategy(strat: &str, limit: FillNullLimit) -> PyResult<FillNullStrategy> {
    if limit.is_some() && strat != "forward" && strat != "backward" {
        Err(PyValueError::new_err(
            "'limit' argument in 'fill_null' only allowed for {'forward', 'backward'} strategies",
        ))
    } else {
        let strat = match strat {
            "backward" => FillNullStrategy::Backward(limit),
            "forward" => FillNullStrategy::Forward(limit),
            "min" => FillNullStrategy::Min,
            "max" => FillNullStrategy::Max,
            "mean" => FillNullStrategy::Mean,
            "zero" => FillNullStrategy::Zero,
            "one" => FillNullStrategy::One,
            s => {
                return Err(PyValueError::new_err(format!(
                    "Strategy {} not supported",
                    s
                )))
            }
        };

        Ok(strat)
    }
}
#[cfg(feature = "parquet")]
impl FromPyObject<'_> for Wrap<ParallelStrategy> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let unit = match ob.str()?.to_str()? {
            "auto" => ParallelStrategy::Auto,
            "columns" => ParallelStrategy::Columns,
            "row_groups" => ParallelStrategy::RowGroups,
            "none" => ParallelStrategy::None,
            _ => {
                return Err(PyValueError::new_err(
                    "expected one of {'auto', 'columns', 'row_groups', 'none'}",
                ))
            }
        };
        Ok(Wrap(unit))
    }
}
