pub(crate) mod any_value;
mod categorical;
pub(crate) mod chunked_array;
mod datetime;

use std::convert::Infallible;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::hash::{Hash, Hasher};

pub use categorical::PyCategories;
#[cfg(feature = "object")]
use polars::chunked_array::object::PolarsObjectSafe;
use polars::frame::row::Row;
#[cfg(feature = "avro")]
use polars::io::avro::AvroCompression;
#[cfg(feature = "cloud")]
use polars::io::cloud::CloudOptions;
#[cfg(feature = "csv")]
use polars::io::csv::write::CsvCompression;
use polars::prelude::ColumnMapping;
use polars::prelude::default_values::{
    DefaultFieldValues, IcebergIdentityTransformedPartitionFields,
};
use polars::prelude::deletion::DeletionFilesList;
use polars::series::ops::NullBehavior;
use polars_compute::decimal::dec128_verify_prec_scale;
use polars_core::schema::iceberg::IcebergSchema;
use polars_core::utils::arrow::array::Array;
use polars_core::utils::arrow::types::NativeType;
use polars_core::utils::materialize_dyn_int;
use polars_lazy::prelude::*;
#[cfg(feature = "parquet")]
use polars_parquet::write::StatisticsOptions;
use polars_plan::dsl::ScanSources;
#[cfg(feature = "csv")]
use polars_utils::compression::{GzipLevel as CsvGzipLevel, ZstdLevel as CsvZstdLevel};
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::plpath::CloudScheme;
use polars_utils::total_ord::{TotalEq, TotalHash};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::sync::GILOnceCell;
use pyo3::types::{IntoPyDict, PyDict, PyList, PySequence, PyString};

use crate::error::PyPolarsErr;
use crate::expr::PyExpr;
use crate::file::{PythonScanSourceInput, get_python_scan_source_input};
use crate::interop::arrow::to_rust::field_to_rust_arrow;
#[cfg(feature = "object")]
use crate::object::OBJECT_NAME;
use crate::prelude::*;
use crate::py_modules::{pl_series, polars};
use crate::series::PySeries;
use crate::utils::to_py_err;
use crate::{PyDataFrame, PyLazyFrame};

/// # Safety
/// Should only be implemented for transparent types
pub(crate) unsafe trait Transparent {
    type Target;
}

unsafe impl Transparent for PySeries {
    type Target = Series;
}

unsafe impl<T> Transparent for Wrap<T> {
    type Target = T;
}

unsafe impl<T: Transparent> Transparent for Option<T> {
    type Target = Option<T::Target>;
}

pub(crate) fn reinterpret_vec<T: Transparent>(input: Vec<T>) -> Vec<T::Target> {
    assert_eq!(size_of::<T>(), size_of::<T::Target>());
    assert_eq!(align_of::<T>(), align_of::<T::Target>());
    let len = input.len();
    let cap = input.capacity();
    let mut manual_drop_vec = std::mem::ManuallyDrop::new(input);
    let vec_ptr: *mut T = manual_drop_vec.as_mut_ptr();
    let ptr: *mut T::Target = vec_ptr as *mut T::Target;
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

pub(crate) fn vec_extract_wrapped<T>(buf: Vec<Wrap<T>>) -> Vec<T> {
    reinterpret_vec(buf)
}

#[derive(PartialEq, Eq, Hash)]
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
pub(crate) fn get_df(obj: &Bound<'_, PyAny>) -> PyResult<DataFrame> {
    let pydf = obj.getattr(intern!(obj.py(), "_df"))?;
    Ok(pydf.extract::<PyDataFrame>()?.df.into_inner())
}

pub(crate) fn get_lf(obj: &Bound<'_, PyAny>) -> PyResult<LazyFrame> {
    let pydf = obj.getattr(intern!(obj.py(), "_ldf"))?;
    Ok(pydf.extract::<PyLazyFrame>()?.ldf.into_inner())
}

pub(crate) fn get_series(obj: &Bound<'_, PyAny>) -> PyResult<Series> {
    let s = obj.getattr(intern!(obj.py(), "_s"))?;
    Ok(s.extract::<PySeries>()?.series.into_inner())
}

pub(crate) fn to_series(py: Python<'_>, s: PySeries) -> PyResult<Bound<'_, PyAny>> {
    let series = pl_series(py).bind(py);
    let constructor = series.getattr(intern!(py, "_from_pyseries"))?;
    constructor.call1((s,))
}

impl<'py> FromPyObject<'py> for Wrap<PlSmallStr> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Wrap((&*ob.extract::<PyBackedStr>()?).into()))
    }
}

#[cfg(feature = "csv")]
impl<'py> FromPyObject<'py> for Wrap<NullValues> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = ob.extract::<PyBackedStr>() {
            Ok(Wrap(NullValues::AllColumnsSingle((&*s).into())))
        } else if let Ok(s) = ob.extract::<Vec<PyBackedStr>>() {
            Ok(Wrap(NullValues::AllColumns(
                s.into_iter().map(|x| (&*x).into()).collect(),
            )))
        } else if let Ok(s) = ob.extract::<Vec<(PyBackedStr, PyBackedStr)>>() {
            Ok(Wrap(NullValues::Named(
                s.into_iter()
                    .map(|(a, b)| ((&*a).into(), (&*b).into()))
                    .collect(),
            )))
        } else {
            Err(
                PyPolarsErr::Other("could not extract value from null_values argument".into())
                    .into(),
            )
        }
    }
}

fn struct_dict<'a, 'py>(
    py: Python<'py>,
    vals: impl Iterator<Item = AnyValue<'a>>,
    flds: &[Field],
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    flds.iter().zip(vals).try_for_each(|(fld, val)| {
        dict.set_item(fld.name().as_str(), Wrap(val).into_pyobject(py)?)
    })?;
    Ok(dict)
}

impl<'py> IntoPyObject<'py> for &Wrap<DataType> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let pl = polars(py).bind(py);

        match &self.0 {
            DataType::Int8 => {
                let class = pl.getattr(intern!(py, "Int8"))?;
                class.call0()
            },
            DataType::Int16 => {
                let class = pl.getattr(intern!(py, "Int16"))?;
                class.call0()
            },
            DataType::Int32 => {
                let class = pl.getattr(intern!(py, "Int32"))?;
                class.call0()
            },
            DataType::Int64 => {
                let class = pl.getattr(intern!(py, "Int64"))?;
                class.call0()
            },
            DataType::UInt8 => {
                let class = pl.getattr(intern!(py, "UInt8"))?;
                class.call0()
            },
            DataType::UInt16 => {
                let class = pl.getattr(intern!(py, "UInt16"))?;
                class.call0()
            },
            DataType::UInt32 => {
                let class = pl.getattr(intern!(py, "UInt32"))?;
                class.call0()
            },
            DataType::UInt64 => {
                let class = pl.getattr(intern!(py, "UInt64"))?;
                class.call0()
            },
            DataType::UInt128 => {
                let class = pl.getattr(intern!(py, "UInt128"))?;
                class.call0()
            },
            DataType::Int128 => {
                let class = pl.getattr(intern!(py, "Int128"))?;
                class.call0()
            },
            DataType::Float32 => {
                let class = pl.getattr(intern!(py, "Float32"))?;
                class.call0()
            },
            DataType::Float64 | DataType::Unknown(UnknownKind::Float) => {
                let class = pl.getattr(intern!(py, "Float64"))?;
                class.call0()
            },
            DataType::Decimal(precision, scale) => {
                let class = pl.getattr(intern!(py, "Decimal"))?;
                let args = (*precision, *scale);
                class.call1(args)
            },
            DataType::Boolean => {
                let class = pl.getattr(intern!(py, "Boolean"))?;
                class.call0()
            },
            DataType::String | DataType::Unknown(UnknownKind::Str) => {
                let class = pl.getattr(intern!(py, "String"))?;
                class.call0()
            },
            DataType::Binary => {
                let class = pl.getattr(intern!(py, "Binary"))?;
                class.call0()
            },
            DataType::Array(inner, size) => {
                let class = pl.getattr(intern!(py, "Array"))?;
                let inner = Wrap(*inner.clone());
                let args = (&inner, *size);
                class.call1(args)
            },
            DataType::List(inner) => {
                let class = pl.getattr(intern!(py, "List"))?;
                let inner = Wrap(*inner.clone());
                class.call1((&inner,))
            },
            DataType::Date => {
                let class = pl.getattr(intern!(py, "Date"))?;
                class.call0()
            },
            DataType::Datetime(tu, tz) => {
                let datetime_class = pl.getattr(intern!(py, "Datetime"))?;
                datetime_class.call1((tu.to_ascii(), tz.as_deref().map(|x| x.as_str())))
            },
            DataType::Duration(tu) => {
                let duration_class = pl.getattr(intern!(py, "Duration"))?;
                duration_class.call1((tu.to_ascii(),))
            },
            #[cfg(feature = "object")]
            DataType::Object(_) => {
                let class = pl.getattr(intern!(py, "Object"))?;
                class.call0()
            },
            DataType::Categorical(cats, _) => {
                let categories_class = pl.getattr(intern!(py, "Categories"))?;
                let categorical_class = pl.getattr(intern!(py, "Categorical"))?;
                let categories = categories_class
                    .call_method1("_from_py_categories", (PyCategories::from(cats.clone()),))?;
                let kwargs = [("categories", categories)];
                categorical_class.call((), Some(&kwargs.into_py_dict(py)?))
            },
            DataType::Enum(_, mapping) => {
                let categories = unsafe {
                    StringChunked::from_chunks(
                        PlSmallStr::from_static("category"),
                        vec![mapping.to_arrow(true)],
                    )
                };
                let class = pl.getattr(intern!(py, "Enum"))?;
                let series = to_series(py, categories.into_series().into())?;
                class.call1((series,))
            },
            DataType::Time => pl.getattr(intern!(py, "Time")).and_then(|x| x.call0()),
            DataType::Struct(fields) => {
                let field_class = pl.getattr(intern!(py, "Field"))?;
                let iter = fields.iter().map(|fld| {
                    let name = fld.name().as_str();
                    let dtype = Wrap(fld.dtype().clone());
                    field_class.call1((name, &dtype)).unwrap()
                });
                let fields = PyList::new(py, iter)?;
                let struct_class = pl.getattr(intern!(py, "Struct"))?;
                struct_class.call1((fields,))
            },
            DataType::Null => {
                let class = pl.getattr(intern!(py, "Null"))?;
                class.call0()
            },
            DataType::Unknown(UnknownKind::Int(v)) => {
                Wrap(materialize_dyn_int(*v).dtype()).into_pyobject(py)
            },
            DataType::Unknown(_) => {
                let class = pl.getattr(intern!(py, "Unknown"))?;
                class.call0()
            },
            DataType::BinaryOffset => {
                unimplemented!()
            },
        }
    }
}

impl<'py> FromPyObject<'py> for Wrap<Field> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        let name = ob
            .getattr(intern!(py, "name"))?
            .str()?
            .extract::<PyBackedStr>()?;
        let dtype = ob
            .getattr(intern!(py, "dtype"))?
            .extract::<Wrap<DataType>>()?;
        Ok(Wrap(Field::new((&*name).into(), dtype.0)))
    }
}

impl<'py> FromPyObject<'py> for Wrap<DataType> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        let type_name = ob.get_type().qualname()?.to_string();

        let dtype = match &*type_name {
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
                    "Categorical" => DataType::from_categories(Categories::global()),
                    "Enum" => DataType::from_frozen_categories(FrozenCategories::new([]).unwrap()),
                    "Date" => DataType::Date,
                    "Time" => DataType::Time,
                    "Datetime" => DataType::Datetime(TimeUnit::Microseconds, None),
                    "Duration" => DataType::Duration(TimeUnit::Microseconds),
                    "List" => DataType::List(Box::new(DataType::Null)),
                    "Array" => DataType::Array(Box::new(DataType::Null), 0),
                    "Struct" => DataType::Struct(vec![]),
                    "Null" => DataType::Null,
                    #[cfg(feature = "object")]
                    "Object" => DataType::Object(OBJECT_NAME),
                    "Unknown" => DataType::Unknown(Default::default()),
                    "Decimal" => {
                        return Err(PyTypeError::new_err(
                            "Decimal without precision/scale set is not a valid Polars datatype",
                        ));
                    },
                    dt => {
                        return Err(PyTypeError::new_err(format!(
                            "'{dt}' is not a Polars data type",
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
            "Categorical" => {
                let categories = ob.getattr(intern!(py, "categories")).unwrap();
                let py_categories = categories.getattr(intern!(py, "_categories")).unwrap();
                let py_categories = py_categories.extract::<PyCategories>()?;
                DataType::from_categories(py_categories.categories().clone())
            },
            "Enum" => {
                let categories = ob.getattr(intern!(py, "categories")).unwrap();
                let s = get_series(&categories.as_borrowed())?;
                let ca = s.str().map_err(PyPolarsErr::from)?;
                let categories = ca.downcast_iter().next().unwrap().clone();
                assert!(!categories.has_nulls());
                DataType::from_frozen_categories(
                    FrozenCategories::new(categories.values_iter()).unwrap(),
                )
            },
            "Date" => DataType::Date,
            "Time" => DataType::Time,
            "Datetime" => {
                let time_unit = ob.getattr(intern!(py, "time_unit")).unwrap();
                let time_unit = time_unit.extract::<Wrap<TimeUnit>>()?.0;
                let time_zone = ob.getattr(intern!(py, "time_zone")).unwrap();
                let time_zone = time_zone.extract::<Option<PyBackedStr>>()?;
                DataType::Datetime(
                    time_unit,
                    TimeZone::opt_try_new(time_zone.as_deref()).map_err(to_py_err)?,
                )
            },
            "Duration" => {
                let time_unit = ob.getattr(intern!(py, "time_unit")).unwrap();
                let time_unit = time_unit.extract::<Wrap<TimeUnit>>()?.0;
                DataType::Duration(time_unit)
            },
            "Decimal" => {
                let precision = ob.getattr(intern!(py, "precision"))?.extract()?;
                let scale = ob.getattr(intern!(py, "scale"))?.extract()?;
                dec128_verify_prec_scale(precision, scale).map_err(to_py_err)?;
                DataType::Decimal(precision, scale)
            },
            "List" => {
                let inner = ob.getattr(intern!(py, "inner")).unwrap();
                let inner = inner.extract::<Wrap<DataType>>()?;
                DataType::List(Box::new(inner.0))
            },
            "Array" => {
                let inner = ob.getattr(intern!(py, "inner")).unwrap();
                let size = ob.getattr(intern!(py, "size")).unwrap();
                let inner = inner.extract::<Wrap<DataType>>()?;
                let size = size.extract::<usize>()?;
                DataType::Array(Box::new(inner.0), size)
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
            "Null" => DataType::Null,
            #[cfg(feature = "object")]
            "Object" => DataType::Object(OBJECT_NAME),
            "Unknown" => DataType::Unknown(Default::default()),
            dt => {
                return Err(PyTypeError::new_err(format!(
                    "'{dt}' is not a Polars data type",
                )));
            },
        };
        Ok(Wrap(dtype))
    }
}

enum CategoricalOrdering {
    Lexical,
}

impl<'py> IntoPyObject<'py> for Wrap<CategoricalOrdering> {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        "lexical".into_pyobject(py)
    }
}

impl<'py> IntoPyObject<'py> for Wrap<TimeUnit> {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.0.to_ascii().into_pyobject(py)
    }
}

#[cfg(feature = "parquet")]
impl<'py> FromPyObject<'py> for Wrap<StatisticsOptions> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let mut statistics = StatisticsOptions::empty();

        let dict = ob.downcast::<PyDict>()?;
        for (key, val) in dict {
            let key = key.extract::<PyBackedStr>()?;
            let val = val.extract::<bool>()?;

            match key.as_ref() {
                "min" => statistics.min_value = val,
                "max" => statistics.max_value = val,
                "distinct_count" => statistics.distinct_count = val,
                "null_count" => statistics.null_count = val,
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "'{key}' is not a valid statistic option",
                    )));
                },
            }
        }

        Ok(Wrap(statistics))
    }
}

impl<'py> FromPyObject<'py> for Wrap<Row<'static>> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let vals = ob.extract::<Vec<Wrap<AnyValue<'static>>>>()?;
        let vals = reinterpret_vec(vals);
        Ok(Wrap(Row(vals)))
    }
}

impl<'py> FromPyObject<'py> for Wrap<Schema> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dict = ob.downcast::<PyDict>()?;

        Ok(Wrap(
            dict.iter()
                .map(|(key, val)| {
                    let key = key.extract::<PyBackedStr>()?;
                    let val = val.extract::<Wrap<DataType>>()?;

                    Ok(Field::new((&*key).into(), val.0))
                })
                .collect::<PyResult<Schema>>()?,
        ))
    }
}

impl<'py> FromPyObject<'py> for Wrap<ArrowSchema> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();

        let pyarrow_schema_cls = py
            .import(intern!(py, "pyarrow"))?
            .getattr(intern!(py, "Schema"))?;

        if ob.is_none() {
            return Err(PyValueError::new_err("arrow_schema() returned None").into());
        }

        let schema_cls = ob.getattr(intern!(py, "__class__"))?;

        if !schema_cls.is(&pyarrow_schema_cls) {
            return Err(PyTypeError::new_err(format!(
                "expected pyarrow.Schema, got: {schema_cls}"
            )));
        }

        let mut iter = ob.try_iter()?.map(|x| x.and_then(field_to_rust_arrow));

        let mut last_err = None;

        let schema =
            ArrowSchema::from_iter_check_duplicates(std::iter::from_fn(|| match iter.next() {
                Some(Ok(v)) => Some(v),
                Some(Err(e)) => {
                    last_err = Some(e);
                    None
                },
                None => None,
            }))
            .map_err(to_py_err)?;

        if let Some(last_err) = last_err {
            return Err(last_err.into());
        }

        Ok(Wrap(schema))
    }
}

impl<'py> FromPyObject<'py> for Wrap<ScanSources> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let list = ob.downcast::<PyList>()?.to_owned();

        if list.is_empty() {
            return Ok(Wrap(ScanSources::default()));
        }

        enum MutableSources {
            Paths(Vec<PlPath>),
            Files(Vec<File>),
            Buffers(Vec<MemSlice>),
        }

        let num_items = list.len();
        let mut iter = list
            .into_iter()
            .map(|val| get_python_scan_source_input(val.unbind(), false));

        let Some(first) = iter.next() else {
            return Ok(Wrap(ScanSources::default()));
        };

        let mut sources = match first? {
            PythonScanSourceInput::Path(path) => {
                let mut sources = Vec::with_capacity(num_items);
                sources.push(path);
                MutableSources::Paths(sources)
            },
            PythonScanSourceInput::File(file) => {
                let mut sources = Vec::with_capacity(num_items);
                sources.push(file.into());
                MutableSources::Files(sources)
            },
            PythonScanSourceInput::Buffer(buffer) => {
                let mut sources = Vec::with_capacity(num_items);
                sources.push(buffer);
                MutableSources::Buffers(sources)
            },
        };

        for source in iter {
            match (&mut sources, source?) {
                (MutableSources::Paths(v), PythonScanSourceInput::Path(p)) => v.push(p),
                (MutableSources::Files(v), PythonScanSourceInput::File(f)) => v.push(f.into()),
                (MutableSources::Buffers(v), PythonScanSourceInput::Buffer(f)) => v.push(f),
                _ => {
                    return Err(PyTypeError::new_err(
                        "Cannot combine in-memory bytes, paths and files for scan sources",
                    ));
                },
            }
        }

        Ok(Wrap(match sources {
            MutableSources::Paths(i) => ScanSources::Paths(i.into()),
            MutableSources::Files(i) => ScanSources::Files(i.into()),
            MutableSources::Buffers(i) => ScanSources::Buffers(i.into()),
        }))
    }
}

impl<'py> IntoPyObject<'py> for Wrap<Schema> {
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        self.0
            .iter()
            .try_for_each(|(k, v)| dict.set_item(k.as_str(), &Wrap(v.clone())))?;
        Ok(dict)
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct ObjectValue {
    pub inner: PyObject,
}

impl Clone for ObjectValue {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
        })
    }
}

impl Hash for ObjectValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let h = Python::with_gil(|py| self.inner.bind(py).hash().expect("should be hashable"));
        state.write_isize(h)
    }
}

impl Eq for ObjectValue {}

impl PartialEq for ObjectValue {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| {
            match self
                .inner
                .bind(py)
                .rich_compare(other.inner.bind(py), CompareOp::Eq)
            {
                Ok(result) => result.is_truthy().unwrap(),
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

impl TotalHash for ObjectValue {
    fn tot_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.hash(state);
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

impl<'py> FromPyObject<'py> for ObjectValue {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(ObjectValue {
            inner: ob.to_owned().unbind(),
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

impl<'a, 'py> IntoPyObject<'py> for &'a ObjectValue {
    type Target = PyAny;
    type Output = Borrowed<'a, 'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.inner.bind_borrowed(py))
    }
}

impl Default for ObjectValue {
    fn default() -> Self {
        Python::with_gil(|py| ObjectValue { inner: py.None() })
    }
}

impl<'py, T: NativeType + FromPyObject<'py>> FromPyObject<'py> for Wrap<Vec<T>> {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let seq = obj.downcast::<PySequence>()?;
        let mut v = Vec::with_capacity(seq.len().unwrap_or(0));
        for item in seq.try_iter()? {
            v.push(item?.extract::<T>()?);
        }
        Ok(Wrap(v))
    }
}

#[cfg(feature = "asof_join")]
impl<'py> FromPyObject<'py> for Wrap<AsofStrategy> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*(ob.extract::<PyBackedStr>()?) {
            "backward" => AsofStrategy::Backward,
            "forward" => AsofStrategy::Forward,
            "nearest" => AsofStrategy::Nearest,
            v => {
                return Err(PyValueError::new_err(format!(
                    "asof `strategy` must be one of {{'backward', 'forward', 'nearest'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<InterpolationMethod> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*(ob.extract::<PyBackedStr>()?) {
            "linear" => InterpolationMethod::Linear,
            "nearest" => InterpolationMethod::Nearest,
            v => {
                return Err(PyValueError::new_err(format!(
                    "interpolation `method` must be one of {{'linear', 'nearest'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "avro")]
impl<'py> FromPyObject<'py> for Wrap<Option<AvroCompression>> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "uncompressed" => None,
            "snappy" => Some(AvroCompression::Snappy),
            "deflate" => Some(AvroCompression::Deflate),
            v => {
                return Err(PyValueError::new_err(format!(
                    "avro `compression` must be one of {{'uncompressed', 'snappy', 'deflate'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<CategoricalOrdering> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "lexical" => CategoricalOrdering::Lexical,
            "physical" => {
                polars_warn!(
                    Deprecation,
                    "physical ordering is deprecated, will use lexical ordering instead"
                );
                CategoricalOrdering::Lexical
            },
            v => {
                return Err(PyValueError::new_err(format!(
                    "categorical `ordering` must be one of {{'physical', 'lexical'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<StartBy> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
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
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<ClosedWindow> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "left" => ClosedWindow::Left,
            "right" => ClosedWindow::Right,
            "both" => ClosedWindow::Both,
            "none" => ClosedWindow::None,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`closed` must be one of {{'left', 'right', 'both', 'none'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<RoundMode> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "half_to_even" => RoundMode::HalfToEven,
            "half_away_from_zero" => RoundMode::HalfAwayFromZero,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`mode` must be one of {{'half_to_even', 'half_away_from_zero'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "csv")]
impl<'py> FromPyObject<'py> for Wrap<CsvEncoding> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "utf8" => CsvEncoding::Utf8,
            "utf8-lossy" => CsvEncoding::LossyUtf8,
            v => {
                return Err(PyValueError::new_err(format!(
                    "csv `encoding` must be one of {{'utf8', 'utf8-lossy'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "ipc")]
impl<'py> FromPyObject<'py> for Wrap<Option<IpcCompression>> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "uncompressed" => None,
            "lz4" => Some(IpcCompression::LZ4),
            "zstd" => Some(IpcCompression::ZSTD(Default::default())),
            v => {
                return Err(PyValueError::new_err(format!(
                    "ipc `compression` must be one of {{'uncompressed', 'lz4', 'zstd'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<JoinType> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "inner" => JoinType::Inner,
            "left" => JoinType::Left,
            "right" => JoinType::Right,
            "full" => JoinType::Full,
            "semi" => JoinType::Semi,
            "anti" => JoinType::Anti,
            #[cfg(feature = "cross_join")]
            "cross" => JoinType::Cross,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`how` must be one of {{'inner', 'left', 'full', 'semi', 'anti', 'cross'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<Label> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "left" => Label::Left,
            "right" => Label::Right,
            "datapoint" => Label::DataPoint,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`label` must be one of {{'left', 'right', 'datapoint'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<ListToStructWidthStrategy> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "first_non_null" => ListToStructWidthStrategy::FirstNonNull,
            "max_width" => ListToStructWidthStrategy::MaxWidth,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`n_field_strategy` must be one of {{'first_non_null', 'max_width'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<NonExistent> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "null" => NonExistent::Null,
            "raise" => NonExistent::Raise,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`non_existent` must be one of {{'null', 'raise'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<NullBehavior> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "drop" => NullBehavior::Drop,
            "ignore" => NullBehavior::Ignore,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`null_behavior` must be one of {{'drop', 'ignore'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<NullStrategy> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "ignore" => NullStrategy::Ignore,
            "propagate" => NullStrategy::Propagate,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`null_strategy` must be one of {{'ignore', 'propagate'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "parquet")]
impl<'py> FromPyObject<'py> for Wrap<ParallelStrategy> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "auto" => ParallelStrategy::Auto,
            "columns" => ParallelStrategy::Columns,
            "row_groups" => ParallelStrategy::RowGroups,
            "prefiltered" => ParallelStrategy::Prefiltered,
            "none" => ParallelStrategy::None,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`parallel` must be one of {{'auto', 'columns', 'row_groups', 'prefiltered', 'none'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<IndexOrder> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "fortran" => IndexOrder::Fortran,
            "c" => IndexOrder::C,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`order` must be one of {{'fortran', 'c'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<QuantileMethod> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "lower" => QuantileMethod::Lower,
            "higher" => QuantileMethod::Higher,
            "nearest" => QuantileMethod::Nearest,
            "linear" => QuantileMethod::Linear,
            "midpoint" => QuantileMethod::Midpoint,
            "equiprobable" => QuantileMethod::Equiprobable,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`interpolation` must be one of {{'lower', 'higher', 'nearest', 'linear', 'midpoint', 'equiprobable'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<RankMethod> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "min" => RankMethod::Min,
            "max" => RankMethod::Max,
            "average" => RankMethod::Average,
            "dense" => RankMethod::Dense,
            "ordinal" => RankMethod::Ordinal,
            "random" => RankMethod::Random,
            v => {
                return Err(PyValueError::new_err(format!(
                    "rank `method` must be one of {{'min', 'max', 'average', 'dense', 'ordinal', 'random'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<Roll> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "raise" => Roll::Raise,
            "forward" => Roll::Forward,
            "backward" => Roll::Backward,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`roll` must be one of {{'raise', 'forward', 'backward'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<TimeUnit> {
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
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<UniqueKeepStrategy> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "first" => UniqueKeepStrategy::First,
            "last" => UniqueKeepStrategy::Last,
            "none" => UniqueKeepStrategy::None,
            "any" => UniqueKeepStrategy::Any,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`keep` must be one of {{'first', 'last', 'any', 'none'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "search_sorted")]
impl<'py> FromPyObject<'py> for Wrap<SearchSortedSide> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "any" => SearchSortedSide::Any,
            "left" => SearchSortedSide::Left,
            "right" => SearchSortedSide::Right,
            v => {
                return Err(PyValueError::new_err(format!(
                    "sorted `side` must be one of {{'any', 'left', 'right'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<ClosedInterval> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "both" => ClosedInterval::Both,
            "left" => ClosedInterval::Left,
            "right" => ClosedInterval::Right,
            "none" => ClosedInterval::None,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`closed` must be one of {{'both', 'left', 'right', 'none'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<WindowMapping> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "group_to_rows" => WindowMapping::GroupsToRows,
            "join" => WindowMapping::Join,
            "explode" => WindowMapping::Explode,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`mapping_strategy` must be one of {{'group_to_rows', 'join', 'explode'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<JoinValidation> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "1:1" => JoinValidation::OneToOne,
            "1:m" => JoinValidation::OneToMany,
            "m:m" => JoinValidation::ManyToMany,
            "m:1" => JoinValidation::ManyToOne,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`validate` must be one of {{'m:m', 'm:1', '1:m', '1:1'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<MaintainOrderJoin> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "none" => MaintainOrderJoin::None,
            "left" => MaintainOrderJoin::Left,
            "right" => MaintainOrderJoin::Right,
            "left_right" => MaintainOrderJoin::LeftRight,
            "right_left" => MaintainOrderJoin::RightLeft,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`maintain_order` must be one of {{'none', 'left', 'right', 'left_right', 'right_left'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "csv")]
impl<'py> FromPyObject<'py> for Wrap<QuoteStyle> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "always" => QuoteStyle::Always,
            "necessary" => QuoteStyle::Necessary,
            "non_numeric" => QuoteStyle::NonNumeric,
            "never" => QuoteStyle::Never,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`quote_style` must be one of {{'always', 'necessary', 'non_numeric', 'never'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "cloud")]
pub(crate) fn parse_cloud_options(
    uri: &str,
    kv: impl IntoIterator<Item = (String, String)>,
) -> PyResult<CloudOptions> {
    let iter: &mut dyn Iterator<Item = _> = &mut kv.into_iter();
    let out = CloudOptions::from_untyped_config(CloudScheme::from_uri(uri).as_ref(), iter)
        .map_err(PyPolarsErr::from)?;
    Ok(out)
}

#[cfg(feature = "list_sets")]
impl<'py> FromPyObject<'py> for Wrap<SetOperation> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "union" => SetOperation::Union,
            "difference" => SetOperation::Difference,
            "intersection" => SetOperation::Intersection,
            "symmetric_difference" => SetOperation::SymmetricDifference,
            v => {
                return Err(PyValueError::new_err(format!(
                    "set operation must be one of {{'union', 'difference', 'intersection', 'symmetric_difference'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

// Conversion from ScanCastOptions class from the Python side.
impl<'py> FromPyObject<'py> for Wrap<CastColumnsPolicy> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if ob.is_none() {
            // Initialize the default ScanCastOptions from Python.
            static DEFAULT: GILOnceCell<Wrap<CastColumnsPolicy>> = GILOnceCell::new();

            let out = DEFAULT.get_or_try_init(ob.py(), || {
                let ob = PyModule::import(ob.py(), "polars.io.scan_options.cast_options")
                    .unwrap()
                    .getattr("ScanCastOptions")
                    .unwrap()
                    .call_method0("_default")
                    .unwrap();

                let out = Self::extract_bound(&ob)?;

                // The default policy should match ERROR_ON_MISMATCH (but this can change).
                debug_assert_eq!(&out.0, &CastColumnsPolicy::ERROR_ON_MISMATCH);

                PyResult::Ok(out)
            })?;

            return Ok(out.clone());
        }

        let py = ob.py();

        let integer_upcast = match &*ob
            .getattr(intern!(py, "integer_cast"))?
            .extract::<PyBackedStr>()?
        {
            "upcast" => true,
            "forbid" => false,
            v => {
                return Err(PyValueError::new_err(format!(
                    "unknown option for integer_cast: {v}"
                )));
            },
        };

        let mut float_upcast = false;
        let mut float_downcast = false;

        let float_cast_object = ob.getattr(intern!(py, "float_cast"))?;

        parse_multiple_options("float_cast", float_cast_object, |v| {
            match v {
                "forbid" => {},
                "upcast" => float_upcast = true,
                "downcast" => float_downcast = true,
                v => {
                    return Err(PyValueError::new_err(format!(
                        "unknown option for float_cast: {v}"
                    )));
                },
            }

            Ok(())
        })?;

        let mut datetime_nanoseconds_downcast = false;
        let mut datetime_convert_timezone = false;

        let datetime_cast_object = ob.getattr(intern!(py, "datetime_cast"))?;

        parse_multiple_options("datetime_cast", datetime_cast_object, |v| {
            match v {
                "forbid" => {},
                "nanosecond-downcast" => datetime_nanoseconds_downcast = true,
                "convert-timezone" => datetime_convert_timezone = true,
                v => {
                    return Err(PyValueError::new_err(format!(
                        "unknown option for datetime_cast: {v}"
                    )));
                },
            };

            Ok(())
        })?;

        let missing_struct_fields = match &*ob
            .getattr(intern!(py, "missing_struct_fields"))?
            .extract::<PyBackedStr>()?
        {
            "insert" => MissingColumnsPolicy::Insert,
            "raise" => MissingColumnsPolicy::Raise,
            v => {
                return Err(PyValueError::new_err(format!(
                    "unknown option for missing_struct_fields: {v}"
                )));
            },
        };

        let extra_struct_fields = match &*ob
            .getattr(intern!(py, "extra_struct_fields"))?
            .extract::<PyBackedStr>()?
        {
            "ignore" => ExtraColumnsPolicy::Ignore,
            "raise" => ExtraColumnsPolicy::Raise,
            v => {
                return Err(PyValueError::new_err(format!(
                    "unknown option for extra_struct_fields: {v}"
                )));
            },
        };

        let categorical_to_string = match &*ob
            .getattr(intern!(py, "categorical_to_string"))?
            .extract::<PyBackedStr>()?
        {
            "allow" => true,
            "forbid" => false,
            v => {
                return Err(PyValueError::new_err(format!(
                    "unknown option for categorical_to_string: {v}"
                )));
            },
        };

        return Ok(Wrap(CastColumnsPolicy {
            integer_upcast,
            float_upcast,
            float_downcast,
            datetime_nanoseconds_downcast,
            datetime_microseconds_downcast: false,
            datetime_convert_timezone,
            null_upcast: true,
            categorical_to_string,
            missing_struct_fields,
            extra_struct_fields,
        }));

        fn parse_multiple_options(
            parameter_name: &'static str,
            py_object: Bound<'_, PyAny>,
            mut parser_func: impl FnMut(&str) -> PyResult<()>,
        ) -> PyResult<()> {
            if let Ok(v) = py_object.extract::<PyBackedStr>() {
                parser_func(&v)?;
            } else if let Ok(v) = py_object.try_iter() {
                for v in v {
                    parser_func(&v?.extract::<PyBackedStr>()?)?;
                }
            } else {
                return Err(PyValueError::new_err(format!(
                    "unknown type for {parameter_name}: {py_object}"
                )));
            }

            Ok(())
        }
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
            )));
        },
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
                    ZstdLevel::try_new(lvl).map_err(|e| PyValueError::new_err(format!("{e:?}")))
                })
                .transpose()?,
        ),
        e => {
            return Err(PyValueError::new_err(format!(
                "parquet `compression` must be one of {{'uncompressed', 'snappy', 'gzip', 'lzo', 'brotli', 'lz4', 'zstd'}}, got {e}",
            )));
        },
    };
    Ok(parsed)
}

#[cfg(feature = "csv")]
pub(crate) fn parse_csv_compression(
    compression: &str,
    compression_level: Option<i32>,
) -> PyResult<Option<CsvCompression>> {
    let parsed = match compression {
        "uncompressed" => None,
        "gzip" => Some(CsvCompression::Gzip(
            compression_level
                .map(|lvl| {
                    CsvGzipLevel::try_new(lvl as u8)
                        .map_err(|e| PyValueError::new_err(format!("{e:?}")))
                })
                .transpose()?,
        )),
        "zstd" => Some(CsvCompression::Zstd(
            compression_level
                .map(|lvl| {
                    CsvZstdLevel::try_new(lvl).map_err(|e| PyValueError::new_err(format!("{e:?}")))
                })
                .transpose()?,
        )),
        e => {
            return Err(PyValueError::new_err(format!(
                "csv `compression` must be one of {{'uncompressed', 'gzip', 'zstd'}}, got {e}",
            )));
        },
    };
    Ok(parsed)
}

pub(crate) fn strings_to_pl_smallstr<I, S>(container: I) -> Vec<PlSmallStr>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    container
        .into_iter()
        .map(|s| PlSmallStr::from_str(s.as_ref()))
        .collect()
}

#[derive(Debug, Copy, Clone)]
pub struct PyCompatLevel(pub CompatLevel);

impl<'py> FromPyObject<'py> for PyCompatLevel {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(PyCompatLevel(if let Ok(level) = ob.extract::<u16>() {
            if let Ok(compat_level) = CompatLevel::with_level(level) {
                compat_level
            } else {
                return Err(PyValueError::new_err("invalid compat level"));
            }
        } else if let Ok(future) = ob.extract::<bool>() {
            if future {
                CompatLevel::newest()
            } else {
                CompatLevel::oldest()
            }
        } else {
            return Err(PyTypeError::new_err(
                "'compat_level' argument accepts int or bool",
            ));
        }))
    }
}

#[cfg(feature = "string_normalize")]
impl<'py> FromPyObject<'py> for Wrap<UnicodeForm> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "NFC" => UnicodeForm::NFC,
            "NFKC" => UnicodeForm::NFKC,
            "NFD" => UnicodeForm::NFD,
            "NFKD" => UnicodeForm::NFKD,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`form` must be one of {{'NFC', 'NFKC', 'NFD', 'NFKD'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

#[cfg(feature = "parquet")]
impl<'py> FromPyObject<'py> for Wrap<Option<KeyValueMetadata>> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        #[derive(FromPyObject)]
        enum Metadata {
            Static(Vec<(String, String)>),
            Dynamic(PyObject),
        }

        let metadata = Option::<Metadata>::extract_bound(ob)?;
        let key_value_metadata = metadata.map(|x| match x {
            Metadata::Static(kv) => KeyValueMetadata::from_static(kv),
            Metadata::Dynamic(func) => KeyValueMetadata::from_py_function(func),
        });
        Ok(Wrap(key_value_metadata))
    }
}

impl<'py> FromPyObject<'py> for Wrap<Option<TimeZone>> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let tz = Option::<Wrap<PlSmallStr>>::extract_bound(ob)?;

        let tz = tz.map(|x| x.0);

        Ok(Wrap(TimeZone::opt_try_new(tz).map_err(to_py_err)?))
    }
}

impl<'py> FromPyObject<'py> for Wrap<UpcastOrForbid> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "upcast" => UpcastOrForbid::Upcast,
            "forbid" => UpcastOrForbid::Forbid,
            v => {
                return Err(PyValueError::new_err(format!(
                    "cast parameter must be one of {{'upcast', 'forbid'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<ExtraColumnsPolicy> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "ignore" => ExtraColumnsPolicy::Ignore,
            "raise" => ExtraColumnsPolicy::Raise,
            v => {
                return Err(PyValueError::new_err(format!(
                    "extra column/field parameter must be one of {{'ignore', 'raise'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<MissingColumnsPolicy> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "insert" => MissingColumnsPolicy::Insert,
            "raise" => MissingColumnsPolicy::Raise,
            v => {
                return Err(PyValueError::new_err(format!(
                    "missing column/field parameter must be one of {{'insert', 'raise'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<MissingColumnsPolicyOrExpr> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(pyexpr) = ob.extract::<PyExpr>() {
            return Ok(Wrap(MissingColumnsPolicyOrExpr::InsertWith(pyexpr.inner)));
        }

        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "insert" => MissingColumnsPolicyOrExpr::Insert,
            "raise" => MissingColumnsPolicyOrExpr::Raise,
            v => {
                return Err(PyValueError::new_err(format!(
                    "missing column/field parameter must be one of {{'insert', 'raise', expression}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<ColumnMapping> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let (column_mapping_type, ob): (PyBackedStr, Bound<'_, PyAny>) = ob.extract()?;

        Ok(Wrap(match &*column_mapping_type {
            "iceberg-column-mapping" => {
                let arrow_schema: Wrap<ArrowSchema> = ob.extract()?;
                ColumnMapping::Iceberg(Arc::new(
                    IcebergSchema::from_arrow_schema(&arrow_schema.0).map_err(to_py_err)?,
                ))
            },

            v => {
                return Err(PyValueError::new_err(format!(
                    "unknown column mapping type: {v}"
                )));
            },
        }))
    }
}

impl<'py> FromPyObject<'py> for Wrap<DeletionFilesList> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let (deletion_file_type, ob): (PyBackedStr, Bound<'_, PyAny>) = ob.extract()?;

        Ok(Wrap(match &*deletion_file_type {
            "iceberg-position-delete" => {
                let dict: Bound<'_, PyDict> = ob.extract()?;

                let mut out = PlIndexMap::new();

                for (k, v) in dict
                    .try_iter()?
                    .zip(dict.call_method0("values")?.try_iter()?)
                {
                    let k: usize = k?.extract()?;
                    let v: Bound<'_, PyAny> = v?.extract()?;

                    let files = v
                        .try_iter()?
                        .map(|x| {
                            x.and_then(|x| {
                                let x: String = x.extract()?;
                                Ok(x)
                            })
                        })
                        .collect::<PyResult<Arc<[String]>>>()?;

                    if !files.is_empty() {
                        out.insert(k, files);
                    }
                }

                DeletionFilesList::IcebergPositionDelete(Arc::new(out))
            },

            v => {
                return Err(PyValueError::new_err(format!(
                    "unknown deletion file type: {v}"
                )));
            },
        }))
    }
}

impl<'py> FromPyObject<'py> for Wrap<DefaultFieldValues> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let (default_values_type, ob): (PyBackedStr, Bound<'_, PyAny>) = ob.extract()?;

        Ok(Wrap(match &*default_values_type {
            "iceberg" => {
                let dict: Bound<'_, PyDict> = ob.extract()?;

                let mut out = PlIndexMap::new();

                for (k, v) in dict
                    .try_iter()?
                    .zip(dict.call_method0("values")?.try_iter()?)
                {
                    let k: u32 = k?.extract()?;
                    let v = v?;

                    let v: Result<Column, String> = if let Ok(s) = get_series(&v) {
                        Ok(s.into_column())
                    } else {
                        let err_msg: String = v.extract()?;
                        Err(err_msg)
                    };

                    out.insert(k, v);
                }

                DefaultFieldValues::Iceberg(Arc::new(IcebergIdentityTransformedPartitionFields(
                    out,
                )))
            },

            v => {
                return Err(PyValueError::new_err(format!(
                    "unknown deletion file type: {v}"
                )));
            },
        }))
    }
}

impl<'py> FromPyObject<'py> for Wrap<PlPath> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(path) = ob.extract::<PyBackedStr>() {
            Ok(Wrap(PlPath::new(&path)))
        } else if let Ok(path) = ob.extract::<std::path::PathBuf>() {
            Ok(Wrap(PlPath::Local(path.into())))
        } else {
            Err(
                PyTypeError::new_err(format!("PlPath cannot be formed from '{}'", ob.get_type()))
                    .into(),
            )
        }
    }
}

impl<'py> IntoPyObject<'py> for Wrap<PlPath> {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.0.to_str().into_pyobject(py)
    }
}
