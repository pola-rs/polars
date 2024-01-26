use std::io::Cursor;
use std::sync::Arc;

use arrow::legacy::error::PolarsResult;
use polars_core::datatypes::{DataType, Field};
use polars_core::error::*;
use polars_core::frame::DataFrame;
use polars_core::prelude::Series;
use pyo3::types::{PyBytes, PyModule};
use pyo3::{PyErr, PyObject, Python};
#[cfg(feature = "serde")]
use serde::ser::Error;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::expr_dyn_fn::*;
use crate::constants::MAP_LIST_NAME;
use crate::prelude::*;

// Will be overwritten on python polar start up.
pub static mut CALL_SERIES_UDF_PYTHON: Option<
    fn(s: Series, lambda: &PyObject) -> PolarsResult<Series>,
> = None;
pub static mut CALL_DF_UDF_PYTHON: Option<
    fn(s: DataFrame, lambda: &PyObject) -> PolarsResult<DataFrame>,
> = None;
pub(super) const MAGIC_BYTE_MARK: &[u8] = "POLARS_PYTHON_UDF".as_bytes();

#[derive(Clone, Debug)]
pub struct PythonFunction(pub PyObject);

impl From<PyObject> for PythonFunction {
    fn from(value: PyObject) -> Self {
        Self(value)
    }
}

impl Eq for PythonFunction {}

impl PartialEq for PythonFunction {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| {
            let eq = self.0.getattr(py, "__eq__").unwrap();
            eq.call1(py, (other.0.clone(),))
                .unwrap()
                .extract::<bool>(py)
                // equality can be not implemented, so default to false
                .unwrap_or(false)
        })
    }
}

#[cfg(feature = "serde")]
impl Serialize for PythonFunction {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Python::with_gil(|py| {
            let pickle = PyModule::import(py, "cloudpickle")
                .or(PyModule::import(py, "pickle"))
                .expect("Unable to import 'cloudpickle' or 'pickle'")
                .getattr("dumps")
                .unwrap();

            let python_function = self.0.clone();

            let dumped = pickle
                .call1((python_function,))
                .map_err(|s| S::Error::custom(format!("cannot pickle {s}")))?;
            let dumped = dumped.extract::<&PyBytes>().unwrap();

            serializer.serialize_bytes(dumped.as_bytes())
        })
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for PythonFunction {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        use serde::de::Error;
        let bytes = Vec::<u8>::deserialize(deserializer)?;

        Python::with_gil(|py| {
            let pickle = PyModule::import(py, "cloudpickle")
                .or(PyModule::import(py, "pickle"))
                .expect("Unable to import 'pickle'")
                .getattr("loads")
                .unwrap();
            let arg = (PyBytes::new(py, &bytes),);
            let python_function = pickle
                .call1(arg)
                .map_err(|s| D::Error::custom(format!("cannot pickle {s}")))?;

            Ok(Self(python_function.into()))
        })
    }
}

pub struct PythonUdfExpression {
    python_function: PyObject,
    output_type: Option<DataType>,
    is_elementwise: bool,
}

impl PythonUdfExpression {
    pub fn new(lambda: PyObject, output_type: Option<DataType>, is_elementwise: bool) -> Self {
        Self {
            python_function: lambda,
            output_type,
            is_elementwise,
        }
    }

    #[cfg(feature = "serde")]
    pub(crate) fn try_deserialize(buf: &[u8]) -> PolarsResult<Arc<dyn SeriesUdf>> {
        debug_assert!(buf.starts_with(MAGIC_BYTE_MARK));
        // skip header
        let buf = &buf[MAGIC_BYTE_MARK.len()..];
        let mut reader = Cursor::new(buf);
        let (output_type, is_elementwise): (Option<DataType>, bool) =
            ciborium::de::from_reader(&mut reader).map_err(map_err)?;

        let remainder = &buf[reader.position() as usize..];

        Python::with_gil(|py| {
            let pickle = PyModule::import(py, "cloudpickle")
                .or(PyModule::import(py, "pickle"))
                .expect("Unable to import 'pickle'")
                .getattr("loads")
                .unwrap();
            let arg = (PyBytes::new(py, remainder),);
            let python_function = pickle.call1(arg).map_err(from_pyerr)?;
            Ok(Arc::new(PythonUdfExpression::new(
                python_function.into(),
                output_type,
                is_elementwise,
            )) as Arc<dyn SeriesUdf>)
        })
    }
}

fn from_pyerr(e: PyErr) -> PolarsError {
    PolarsError::ComputeError(format!("error raised in python: {e}").into())
}

impl DataFrameUdf for PythonFunction {
    fn call_udf(&self, df: DataFrame) -> PolarsResult<DataFrame> {
        let func = unsafe { CALL_DF_UDF_PYTHON.unwrap() };
        func(df, &self.0)
    }
}

impl SeriesUdf for PythonUdfExpression {
    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Option<Series>> {
        let func = unsafe { CALL_SERIES_UDF_PYTHON.unwrap() };

        let output_type = self.output_type.clone().unwrap_or(DataType::Unknown);
        let out = func(s[0].clone(), &self.python_function)?;

        polars_ensure!(
            matches!(output_type, DataType::Unknown) || out.dtype() == &output_type,
            SchemaMismatch:
            "expected output type '{:?}', got '{:?}'; set `return_dtype` to the proper datatype",
            output_type, out.dtype(),
        );
        Ok(Some(out))
    }

    #[cfg(feature = "serde")]
    fn try_serialize(&self, buf: &mut Vec<u8>) -> PolarsResult<()> {
        buf.extend_from_slice(MAGIC_BYTE_MARK);
        ciborium::ser::into_writer(&(self.output_type.clone(), self.is_elementwise), &mut *buf)
            .unwrap();

        Python::with_gil(|py| {
            let pickle = PyModule::import(py, "cloudpickle")
                .or(PyModule::import(py, "pickle"))
                .expect("Unable to import 'pickle'")
                .getattr("dumps")
                .unwrap();
            let dumped = pickle
                .call1((self.python_function.clone(),))
                .map_err(from_pyerr)?;
            let dumped = dumped.extract::<&PyBytes>().unwrap();
            buf.extend_from_slice(dumped.as_bytes());
            Ok(())
        })
    }

    fn get_output(&self) -> Option<GetOutput> {
        let output_type = self.output_type.clone();
        Some(GetOutput::map_field(move |fld| match output_type {
            Some(ref dt) => Field::new(fld.name(), dt.clone()),
            None => {
                let mut fld = fld.clone();
                fld.coerce(DataType::Unknown);
                fld
            },
        }))
    }
}

impl Expr {
    pub fn map_python(self, func: PythonUdfExpression, agg_list: bool) -> Expr {
        let (collect_groups, name) = if agg_list {
            (ApplyOptions::ApplyList, MAP_LIST_NAME)
        } else if func.is_elementwise {
            (ApplyOptions::ElementWise, "python_udf")
        } else {
            (ApplyOptions::GroupWise, "python_udf")
        };

        let return_dtype = func.output_type.clone();
        let output_type = GetOutput::map_field(move |fld| match return_dtype {
            Some(ref dt) => Field::new(fld.name(), dt.clone()),
            None => {
                let mut fld = fld.clone();
                fld.coerce(DataType::Unknown);
                fld
            },
        });

        Expr::AnonymousFunction {
            input: vec![self],
            function: SpecialEq::new(Arc::new(func)),
            output_type,
            options: FunctionOptions {
                collect_groups,
                fmt_str: name,
                ..Default::default()
            },
        }
    }
}
