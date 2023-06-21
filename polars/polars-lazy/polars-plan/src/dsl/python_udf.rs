use std::io::Cursor;
use std::sync::Arc;

use polars_arrow::error::PolarsResult;
use polars_core::datatypes::{DataType, Field};
use polars_core::error::*;
use polars_core::prelude::Series;
use pyo3::types::{PyBytes, PyModule};
use pyo3::{PyErr, PyObject, Python};

use super::expr_dyn_fn::*;
use crate::constants::MAP_LIST_NAME;
use crate::prelude::*;

// Will be overwritten on python polar start up.
pub static mut CALL_LAMBDA: Option<fn(s: Series, lambda: &PyObject) -> PolarsResult<Series>> = None;
pub(super) const MAGIC_BYTE_MARK: &[u8] = "POLARS_PYTHON_UDF".as_bytes();

pub struct PythonFunction {
    python_function: PyObject,
    output_type: Option<DataType>,
}

impl PythonFunction {
    pub fn new(lambda: PyObject, output_type: Option<DataType>) -> Self {
        Self {
            python_function: lambda,
            output_type,
        }
    }

    pub(crate) fn try_deserialize(buf: &[u8]) -> PolarsResult<Arc<dyn SeriesUdf>> {
        debug_assert!(buf.starts_with(MAGIC_BYTE_MARK));
        // skip header
        let buf = &buf[MAGIC_BYTE_MARK.len()..];
        let mut reader = Cursor::new(buf);
        let output_type: Option<DataType> =
            ciborium::de::from_reader(&mut reader).map_err(map_err)?;

        let remainder = &buf[reader.position() as usize..];

        Python::with_gil(|py| {
            let pickle = PyModule::import(py, "pickle")
                .expect("Unable to import 'pickle'")
                .getattr("loads")
                .unwrap();
            let arg = (PyBytes::new(py, remainder),);
            let python_function = pickle.call1(arg).map_err(from_pyerr)?;
            Ok(
                Arc::new(PythonFunction::new(python_function.into(), output_type))
                    as Arc<dyn SeriesUdf>,
            )
        })
    }
}

fn from_pyerr(e: PyErr) -> PolarsError {
    PolarsError::ComputeError(format!("error raised in python: {e}").into())
}

impl SeriesUdf for PythonFunction {
    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Option<Series>> {
        let func = unsafe { CALL_LAMBDA.unwrap() };

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

    fn try_serialize(&self, buf: &mut Vec<u8>) -> PolarsResult<()> {
        buf.extend_from_slice(MAGIC_BYTE_MARK);
        ciborium::ser::into_writer(&self.output_type, &mut *buf).unwrap();

        Python::with_gil(|py| {
            let pickle = PyModule::import(py, "pickle")
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
            }
        }))
    }
}

impl Expr {
    pub fn map_python(self, func: PythonFunction, agg_list: bool) -> Expr {
        let (collect_groups, name) = if agg_list {
            (ApplyOptions::ApplyList, MAP_LIST_NAME)
        } else {
            (ApplyOptions::ApplyFlat, "python_udf")
        };

        let return_dtype = func.output_type.clone();
        let output_type = GetOutput::map_field(move |fld| match return_dtype {
            Some(ref dt) => Field::new(fld.name(), dt.clone()),
            None => {
                let mut fld = fld.clone();
                fld.coerce(DataType::Unknown);
                fld
            }
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
