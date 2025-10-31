use std::convert::Infallible;

use arrow;
use polars_core::datatypes::{CompatLevel, DataType};
use polars_core::prelude::*;
use polars_core::utils::materialize_dyn_int;
#[cfg(feature = "lazy")]
use polars_lazy::frame::LazyFrame;
#[cfg(feature = "lazy")]
use polars_plan::dsl::DslPlan;
#[cfg(feature = "lazy")]
use polars_plan::dsl::Expr;
#[cfg(feature = "lazy")]
use polars_utils::pl_serialize;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::ffi::Py_uintptr_t;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
#[cfg(feature = "lazy")]
use pyo3::types::PyBytes;
#[cfg(feature = "dtype-struct")]
use pyo3::types::PyList;
use pyo3::types::{PyDict, PyString};

use super::*;
use crate::error::PyPolarsErr;
use crate::ffi::to_py::to_py_array;

#[cfg(feature = "dtype-categorical")]
pub(crate) fn get_series(obj: &Bound<'_, PyAny>) -> PyResult<Series> {
    let s = obj.getattr(intern!(obj.py(), "_s"))?;
    Ok(s.extract::<PySeries>()?.0)
}

#[repr(transparent)]
#[derive(Debug, Clone)]
/// A wrapper around a [`Series`] that can be converted to and from python with `pyo3`.
pub struct PySeries(pub Series);

#[repr(transparent)]
#[derive(Debug, Clone)]
/// A wrapper around a [`DataFrame`] that can be converted to and from python with `pyo3`.
pub struct PyDataFrame(pub DataFrame);

#[cfg(feature = "lazy")]
#[repr(transparent)]
#[derive(Clone)]
/// A wrapper around a [`DataFrame`] that can be converted to and from python with `pyo3`.
/// # Warning
/// If the [`LazyFrame`] contains in memory data,
/// such as a [`DataFrame`] this will be serialized/deserialized.
///
/// It is recommended to only have `LazyFrame`s that scan data
/// from disk
pub struct PyLazyFrame(pub LazyFrame);

#[cfg(feature = "lazy")]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyExpr(pub Expr);

#[repr(transparent)]
#[derive(Clone)]
pub struct PySchema(pub SchemaRef);

#[repr(transparent)]
#[derive(Clone)]
pub struct PyDataType(pub DataType);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct PyTimeUnit(TimeUnit);

#[repr(transparent)]
#[derive(Clone)]
pub struct PyField(Field);

impl<'py> FromPyObject<'py> for PyField {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        let name = ob
            .getattr(intern!(py, "name"))?
            .str()?
            .extract::<PyBackedStr>()?;
        let dtype = ob.getattr(intern!(py, "dtype"))?.extract::<PyDataType>()?;
        let name: &str = name.as_ref();
        Ok(PyField(Field::new(name.into(), dtype.0)))
    }
}

impl<'py> FromPyObject<'py> for PyTimeUnit {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "ns" => TimeUnit::Nanoseconds,
            "us" => TimeUnit::Microseconds,
            "ms" => TimeUnit::Milliseconds,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`time_unit` must be one of {{'ns', 'us', 'ms'}}, got {v}",
                )));
            },
        };
        Ok(PyTimeUnit(parsed))
    }
}

impl<'py> IntoPyObject<'py> for PyTimeUnit {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let time_unit = match self.0 {
            TimeUnit::Nanoseconds => "ns",
            TimeUnit::Microseconds => "us",
            TimeUnit::Milliseconds => "ms",
        };
        time_unit.into_pyobject(py)
    }
}

impl From<PyDataFrame> for DataFrame {
    fn from(value: PyDataFrame) -> Self {
        value.0
    }
}

impl From<PySeries> for Series {
    fn from(value: PySeries) -> Self {
        value.0
    }
}

#[cfg(feature = "lazy")]
impl From<PyLazyFrame> for LazyFrame {
    fn from(value: PyLazyFrame) -> Self {
        value.0
    }
}

impl From<PySchema> for SchemaRef {
    fn from(value: PySchema) -> Self {
        value.0
    }
}

impl AsRef<Series> for PySeries {
    fn as_ref(&self) -> &Series {
        &self.0
    }
}

impl AsRef<DataFrame> for PyDataFrame {
    fn as_ref(&self) -> &DataFrame {
        &self.0
    }
}

#[cfg(feature = "lazy")]
impl AsRef<LazyFrame> for PyLazyFrame {
    fn as_ref(&self) -> &LazyFrame {
        &self.0
    }
}

impl AsRef<Schema> for PySchema {
    fn as_ref(&self) -> &Schema {
        self.0.as_ref()
    }
}

impl<'a> FromPyObject<'a> for PySeries {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let ob = ob.call_method0("rechunk")?;

        let name = ob.getattr("name")?;
        let py_name = name.str()?;
        let name = py_name.to_cow()?;

        let kwargs = PyDict::new(ob.py());
        if let Ok(compat_level) = ob.call_method0("_newest_compat_level") {
            // Choose the maximum supported between both us and Python's compatibility level.
            let compat_level = compat_level.extract().unwrap();
            let compat_level =
                CompatLevel::with_level(compat_level).unwrap_or(CompatLevel::newest());
            let compat_level_type = POLARS_INTERCHANGE
                .bind(ob.py())
                .getattr("CompatLevel")
                .unwrap();
            let py_compat_level =
                compat_level_type.call_method1("_with_version", (compat_level.get_level(),))?;
            kwargs.set_item("compat_level", py_compat_level)?;
        }
        let arr = ob.call_method("to_arrow", (), Some(&kwargs))?;
        let arr = ffi::to_rust::array_to_rust(&arr)?;
        let name = name.as_ref();
        Ok(PySeries(
            Series::try_from((PlSmallStr::from(name), arr)).map_err(PyPolarsErr::from)?,
        ))
    }
}

impl<'a> FromPyObject<'a> for PyDataFrame {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let series = ob.call_method0("get_columns")?;
        let n = ob.getattr("width")?.extract::<usize>()?;
        let mut columns = Vec::with_capacity(n);
        for pyseries in series.try_iter()? {
            let pyseries = pyseries?;
            let s = pyseries.extract::<PySeries>()?.0;
            columns.push(s.into_column());
        }
        unsafe {
            Ok(PyDataFrame(DataFrame::new_no_checks_height_from_first(
                columns,
            )))
        }
    }
}

#[cfg(feature = "lazy")]
impl<'a> FromPyObject<'a> for PyLazyFrame {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let s = ob.call_method0("__getstate__")?;
        let b = s.extract::<Bound<'_, PyBytes>>()?;
        let b = b.as_bytes();

        let lp = DslPlan::deserialize_versioned(b).map_err(
            |e| PyPolarsErr::Other(
                format!("Error when deserializing LazyFrame. This may be due to mismatched polars versions. {e}")
            ))
            ?;

        Ok(PyLazyFrame(LazyFrame::from(lp)))
    }
}

#[cfg(feature = "lazy")]
impl<'a> FromPyObject<'a> for PyExpr {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let s = ob.call_method0("__getstate__")?.extract::<Vec<u8>>()?;

        let e: Expr = pl_serialize::SerializeOptions::default()
            .deserialize_from_reader::<Expr, &[u8], false>(&*s)
            .map_err(
            |e| PyPolarsErr::Other(
                format!("Error when deserializing 'Expr'. This may be due to mismatched polars versions. {e}")
            )
        )?;

        Ok(PyExpr(e))
    }
}

impl<'py> IntoPyObject<'py> for PySeries {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let polars = POLARS.bind(py);
        let s = SERIES.bind(py);
        match s
            .getattr("_import_arrow_from_c")
            .or_else(|_| s.getattr("_import_from_c"))
        {
            // Go via polars
            Ok(import_arrow_from_c) => {
                // Get supported compatibility level
                let compat_level = CompatLevel::with_level(
                    s.getattr("_newest_compat_level")
                        .map_or(1, |newest_compat_level| {
                            newest_compat_level.call0().unwrap().extract().unwrap()
                        }),
                )
                .unwrap_or(CompatLevel::newest());
                // Prepare pointers on the heap.
                let mut chunk_ptrs = Vec::with_capacity(self.0.n_chunks());
                for i in 0..self.0.n_chunks() {
                    let array = self.0.to_arrow(i, compat_level);
                    let schema = Box::new(arrow::ffi::export_field_to_c(&ArrowField::new(
                        "".into(),
                        array.dtype().clone(),
                        true,
                    )));
                    let array = Box::new(arrow::ffi::export_array_to_c(array.clone()));

                    let schema_ptr: *const arrow::ffi::ArrowSchema = Box::leak(schema);
                    let array_ptr: *const arrow::ffi::ArrowArray = Box::leak(array);

                    chunk_ptrs.push((schema_ptr as Py_uintptr_t, array_ptr as Py_uintptr_t))
                }

                // Somehow we need to clone the Vec, because pyo3 doesn't accept a slice here.
                let pyseries = import_arrow_from_c
                    .call1((self.0.name().as_str(), chunk_ptrs.clone()))
                    .unwrap();
                // Deallocate boxes
                for (schema_ptr, array_ptr) in chunk_ptrs {
                    let schema_ptr = schema_ptr as *mut arrow::ffi::ArrowSchema;
                    let array_ptr = array_ptr as *mut arrow::ffi::ArrowArray;
                    unsafe {
                        // We can drop both because the `schema` isn't read in an owned matter on the other side.
                        let _ = Box::from_raw(schema_ptr);

                        // The array is `ptr::read_unaligned` so there are two owners.
                        // We drop the box, and forget the content so the other process is the owner.
                        let array = Box::from_raw(array_ptr);
                        // We must forget because the other process will call the release callback.
                        // Read *array as Box::into_inner
                        let array = *array;
                        std::mem::forget(array);
                    }
                }

                Ok(pyseries)
            },
            // Go via pyarrow
            Err(_) => {
                let s = self.0.rechunk();
                let name = s.name().as_str();
                let arr = s.to_arrow(0, CompatLevel::oldest());
                let pyarrow = py.import("pyarrow").expect("pyarrow not installed");

                let arg = to_py_array(arr, pyarrow).unwrap();
                let s = polars.call_method1("from_arrow", (arg,)).unwrap();
                let s = s.call_method1("rename", (name,)).unwrap();
                Ok(s)
            },
        }
    }
}

impl<'py> IntoPyObject<'py> for PyDataFrame {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let pyseries = self
            .0
            .get_columns()
            .iter()
            .map(|s| PySeries(s.as_materialized_series().clone()).into_pyobject(py))
            .collect::<PyResult<Vec<_>>>()?;

        let polars = POLARS.bind(py);
        polars.call_method1("DataFrame", (pyseries,))
    }
}

#[cfg(feature = "lazy")]
impl<'py> IntoPyObject<'py> for PyLazyFrame {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        use polars::prelude::PlanSerializationContext;

        let polars = POLARS.bind(py);
        let cls = polars.getattr("LazyFrame")?;
        let instance = cls.call_method1(intern!(py, "__new__"), (&cls,)).unwrap();

        let mut v = vec![];
        self.0
            .logical_plan
            .serialize_versioned(&mut v, PlanSerializationContext::default())
            .unwrap();
        instance.call_method1("__setstate__", (&v,))?;
        Ok(instance)
    }
}

#[cfg(feature = "lazy")]
impl<'py> IntoPyObject<'py> for PyExpr {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let polars = POLARS.bind(py);
        let cls = polars.getattr("Expr")?;
        let instance = cls.call_method1(intern!(py, "__new__"), (&cls,))?;

        let buf = pl_serialize::SerializeOptions::default()
            .serialize_to_bytes::<Expr, false>(&self.0)
            .unwrap();

        instance
            .call_method1("__setstate__", (&buf,))
            .map_err(|err| {
                let msg = format!("deserialization failed: {err}");
                PyValueError::new_err(msg)
            })
    }
}

#[cfg(feature = "dtype-categorical")]
pub(crate) fn to_series(py: Python, s: PySeries) -> Py<PyAny> {
    let series = SERIES.bind(py);
    let constructor = series
        .getattr(intern!(series.py(), "_from_pyseries"))
        .unwrap();
    constructor
        .call1((s,))
        .unwrap()
        .into_pyobject(py)
        .unwrap()
        .into()
}

impl<'py> IntoPyObject<'py> for PyDataType {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let pl = POLARS.bind(py);

        match &self.0 {
            DataType::Int8 => {
                let class = pl.getattr(intern!(py, "Int8")).unwrap();
                class.call0()
            },
            DataType::Int16 => {
                let class = pl.getattr(intern!(py, "Int16")).unwrap();
                class.call0()
            },
            DataType::Int32 => {
                let class = pl.getattr(intern!(py, "Int32")).unwrap();
                class.call0()
            },
            DataType::Int64 => {
                let class = pl.getattr(intern!(py, "Int64")).unwrap();
                class.call0()
            },
            DataType::Int128 => {
                let class = pl.getattr(intern!(py, "Int128")).unwrap();
                class.call0()
            },
            DataType::UInt8 => {
                let class = pl.getattr(intern!(py, "UInt8")).unwrap();
                class.call0()
            },
            DataType::UInt16 => {
                let class = pl.getattr(intern!(py, "UInt16")).unwrap();
                class.call0()
            },
            DataType::UInt32 => {
                let class = pl.getattr(intern!(py, "UInt32")).unwrap();
                class.call0()
            },
            DataType::UInt64 => {
                let class = pl.getattr(intern!(py, "UInt64")).unwrap();
                class.call0()
            },
            DataType::UInt128 => {
                let class = pl.getattr(intern!(py, "UInt128")).unwrap();
                class.call0()
            },
            DataType::Float32 => {
                let class = pl.getattr(intern!(py, "Float32")).unwrap();
                class.call0()
            },
            DataType::Float64 | DataType::Unknown(UnknownKind::Float) => {
                let class = pl.getattr(intern!(py, "Float64")).unwrap();
                class.call0()
            },
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(precision, scale) => {
                let class = pl.getattr(intern!(py, "Decimal")).unwrap();
                let args = (*precision, *scale);
                class.call1(args)
            },
            DataType::Boolean => {
                let class = pl.getattr(intern!(py, "Boolean")).unwrap();
                class.call0()
            },
            DataType::String | DataType::Unknown(UnknownKind::Str) => {
                let class = pl.getattr(intern!(py, "String")).unwrap();
                class.call0()
            },
            DataType::Binary => {
                let class = pl.getattr(intern!(py, "Binary")).unwrap();
                class.call0()
            },
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner, size) => {
                let class = pl.getattr(intern!(py, "Array")).unwrap();
                let inner = PyDataType(*inner.clone()).into_pyobject(py)?;
                let args = (inner, *size);
                class.call1(args)
            },
            DataType::List(inner) => {
                let class = pl.getattr(intern!(py, "List")).unwrap();
                let inner = PyDataType(*inner.clone()).into_pyobject(py)?;
                class.call1((inner,))
            },
            DataType::Date => {
                let class = pl.getattr(intern!(py, "Date")).unwrap();
                class.call0()
            },
            DataType::Datetime(tu, tz) => {
                let datetime_class = pl.getattr(intern!(py, "Datetime")).unwrap();
                datetime_class.call1((tu.to_ascii(), tz.as_ref().map(|s| s.as_str())))
            },
            DataType::Duration(tu) => {
                let duration_class = pl.getattr(intern!(py, "Duration")).unwrap();
                duration_class.call1((tu.to_ascii(),))
            },
            #[cfg(feature = "object")]
            DataType::Object(_) => {
                let class = pl.getattr(intern!(py, "Object")).unwrap();
                class.call0()
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) => {
                let class = pl.getattr(intern!(py, "Categorical")).unwrap();
                class.call1(())
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(categories, _) => {
                // we should always have an initialized rev_map coming from rust
                let class = pl.getattr(intern!(py, "Enum")).unwrap();
                let s =
                    Series::from_arrow("category".into(), categories.categories().clone().boxed())
                        .unwrap();
                let series = to_series(py, PySeries(s));
                class.call1((series,))
            },
            DataType::Time => pl.getattr(intern!(py, "Time")),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                let field_class = pl.getattr(intern!(py, "Field")).unwrap();
                let iter = fields
                    .iter()
                    .map(|fld| {
                        let name = fld.name().as_str();
                        let dtype = PyDataType(fld.dtype().clone()).into_pyobject(py)?;
                        field_class.call1((name, dtype))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let fields = PyList::new(py, iter)?;
                let struct_class = pl.getattr(intern!(py, "Struct")).unwrap();
                struct_class.call1((fields,))
            },
            DataType::Null => {
                let class = pl.getattr(intern!(py, "Null")).unwrap();
                class.call0()
            },
            DataType::Unknown(UnknownKind::Int(v)) => {
                PyDataType(materialize_dyn_int(*v).dtype()).into_pyobject(py)
            },
            DataType::Unknown(_) => {
                let class = pl.getattr(intern!(py, "Unknown")).unwrap();
                class.call0()
            },
            DataType::BinaryOffset => {
                panic!("this type isn't exposed to python")
            },
            #[allow(unreachable_patterns)]
            _ => panic!("activate dtype"),
        }
    }
}

impl<'py> IntoPyObject<'py> for PySchema {
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        for (k, v) in self.0.iter() {
            dict.set_item(k.as_str(), PyDataType(v.clone()).into_pyobject(py)?)?;
        }
        Ok(dict)
    }
}

impl<'py> FromPyObject<'py> for PyDataType {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        let type_name = ob.get_type().qualname()?.to_string();

        let dtype = match type_name.as_ref() {
            "DataTypeClass" => {
                // just the class, not an object
                let name = ob
                    .getattr(intern!(py, "__name__"))?
                    .str()?
                    .extract::<PyBackedStr>()?;
                match &*name {
                    "Int8" => DataType::Int8,
                    "Int16" => DataType::Int16,
                    "Int32" => DataType::Int32,
                    "Int64" => DataType::Int64,
                    "Int128" => DataType::Int128,
                    "UInt8" => DataType::UInt8,
                    "UInt16" => DataType::UInt16,
                    "UInt32" => DataType::UInt32,
                    "UInt64" => DataType::UInt64,
                    "UInt128" => DataType::UInt128,
                    "Float32" => DataType::Float32,
                    "Float64" => DataType::Float64,
                    "Boolean" => DataType::Boolean,
                    "String" => DataType::String,
                    "Binary" => DataType::Binary,
                    #[cfg(feature = "dtype-categorical")]
                    "Categorical" => {
                        DataType::Categorical(Categories::global(), Categories::global().mapping())
                    },
                    #[cfg(feature = "dtype-categorical")]
                    "Enum" => {
                        let categories = FrozenCategories::new([]).unwrap();
                        let mapping = categories.mapping().clone();
                        DataType::Enum(categories, mapping)
                    },
                    "Date" => DataType::Date,
                    "Time" => DataType::Time,
                    "Datetime" => DataType::Datetime(TimeUnit::Microseconds, None),
                    "Duration" => DataType::Duration(TimeUnit::Microseconds),
                    #[cfg(feature = "dtype-decimal")]
                    "Decimal" => {
                        return Err(PyTypeError::new_err("Decimal without specifying precision and scale is not a valid Polars data type".to_string()));
                    },
                    "List" => DataType::List(Box::new(DataType::Null)),
                    #[cfg(feature = "dtype-array")]
                    "Array" => DataType::Array(Box::new(DataType::Null), 0),
                    #[cfg(feature = "dtype-struct")]
                    "Struct" => DataType::Struct(vec![]),
                    "Null" => DataType::Null,
                    #[cfg(feature = "object")]
                    "Object" => todo!(),
                    "Unknown" => DataType::Unknown(Default::default()),
                    dt => {
                        return Err(PyTypeError::new_err(format!(
                            "'{dt}' is not a Polars data type, or the plugin isn't compiled with the right features",
                        )));
                    },
                }
            },
            "Int8" => DataType::Int8,
            "Int16" => DataType::Int16,
            "Int32" => DataType::Int32,
            "Int64" => DataType::Int64,
            "Int128" => DataType::Int128,
            "UInt8" => DataType::UInt8,
            "UInt16" => DataType::UInt16,
            "UInt32" => DataType::UInt32,
            "UInt64" => DataType::UInt64,
            "UInt128" => DataType::UInt128,
            "Float32" => DataType::Float32,
            "Float64" => DataType::Float64,
            "Boolean" => DataType::Boolean,
            "String" => DataType::String,
            "Binary" => DataType::Binary,
            #[cfg(feature = "dtype-categorical")]
            "Categorical" => {
                DataType::Categorical(Categories::global(), Categories::global().mapping())
            },
            #[cfg(feature = "dtype-categorical")]
            "Enum" => {
                let categories = ob.getattr(intern!(py, "categories")).unwrap();
                let s = get_series(&categories.as_borrowed())?;
                let ca = s.str().map_err(PyPolarsErr::from)?;
                let categories = ca.iter();
                let categories = FrozenCategories::new(categories.map(|v| v.unwrap())).unwrap();
                let mapping = categories.mapping().clone();
                DataType::Enum(categories, mapping)
            },
            "Date" => DataType::Date,
            "Time" => DataType::Time,
            "Datetime" => {
                let time_unit = ob.getattr(intern!(py, "time_unit")).unwrap();
                let time_unit = time_unit.extract::<PyTimeUnit>()?.0;
                let time_zone = ob.getattr(intern!(py, "time_zone")).unwrap();
                let time_zone: Option<String> = time_zone.extract()?;
                DataType::Datetime(time_unit, TimeZone::opt_try_new(time_zone).unwrap())
            },
            "Duration" => {
                let time_unit = ob.getattr(intern!(py, "time_unit")).unwrap();
                let time_unit = time_unit.extract::<PyTimeUnit>()?.0;
                DataType::Duration(time_unit)
            },
            #[cfg(feature = "dtype-decimal")]
            "Decimal" => {
                let precision = ob.getattr(intern!(py, "precision"))?.extract()?;
                let scale = ob.getattr(intern!(py, "scale"))?.extract()?;
                DataType::Decimal(precision, scale)
            },
            "List" => {
                let inner = ob.getattr(intern!(py, "inner")).unwrap();
                let inner = inner.extract::<PyDataType>()?;
                DataType::List(Box::new(inner.0))
            },
            #[cfg(feature = "dtype-array")]
            "Array" => {
                let inner = ob.getattr(intern!(py, "inner")).unwrap();
                let size = ob.getattr(intern!(py, "size")).unwrap();
                let inner = inner.extract::<PyDataType>()?;
                let size = size.extract::<usize>()?;
                DataType::Array(Box::new(inner.0), size)
            },
            #[cfg(feature = "dtype-struct")]
            "Struct" => {
                let fields = ob.getattr(intern!(py, "fields"))?;
                let fields = fields
                    .extract::<Vec<PyField>>()?
                    .into_iter()
                    .map(|f| f.0)
                    .collect::<Vec<Field>>();
                DataType::Struct(fields)
            },
            "Null" => DataType::Null,
            #[cfg(feature = "object")]
            "Object" => panic!("object not supported"),
            "Unknown" => DataType::Unknown(Default::default()),
            dt => {
                return Err(PyTypeError::new_err(format!(
                    "'{dt}' is not a Polars data type, or the plugin isn't compiled with the right features",
                )));
            },
        };
        Ok(PyDataType(dtype))
    }
}
