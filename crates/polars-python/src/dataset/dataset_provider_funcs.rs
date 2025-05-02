//! Note: Currently only used for iceberg.
use std::sync::Arc;

use polars::prelude::{DslPlan, PlSmallStr, Schema, SchemaRef};
use polars_core::config;
use polars_error::PolarsResult;
use polars_utils::python_function::PythonObject;
use pyo3::conversion::FromPyObjectBound;
use pyo3::exceptions::PyValueError;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyAnyMethods, PyDict, PyList, PyListMethods};
use pyo3::{PyResult, Python};

use crate::interop::arrow::to_rust::field_to_rust;
use crate::prelude::{Wrap, get_lf};

pub fn reader_name(dataset_object: &PythonObject) -> PlSmallStr {
    Python::with_gil(|py| {
        let name: PyBackedStr = dataset_object
            .getattr(py, "reader_name")?
            .call0(py)?
            .extract(py)?;

        PyResult::Ok(PlSmallStr::from_str(&name))
    })
    .unwrap()
}

pub fn schema(dataset_object: &PythonObject) -> PolarsResult<SchemaRef> {
    Python::with_gil(|py| {
        let pyarrow_schema_cls = py
            .import("pyarrow")
            .ok()
            .and_then(|pa| pa.getattr("Schema").ok());

        let schema_obj = dataset_object.getattr(py, "schema")?.call0(py)?;

        let schema_cls = schema_obj.getattr(py, "__class__")?;

        // PyIceberg returns arrow schemas, we convert them here.
        if let Some(pyarrow_schema_cls) = pyarrow_schema_cls {
            if schema_cls.is(&pyarrow_schema_cls) {
                if config::verbose() {
                    eprintln!("python dataset: convert from arrow schema");
                }

                let mut iter = schema_obj
                    .bind(py)
                    .try_iter()?
                    .map(|x| x.and_then(field_to_rust));

                let mut last_err = None;

                let schema =
                    Schema::from_iter_check_duplicates(std::iter::from_fn(|| match iter.next() {
                        Some(Ok(v)) => Some(v),
                        Some(Err(e)) => {
                            last_err = Some(e);
                            None
                        },
                        None => None,
                    }))?;

                if let Some(last_err) = last_err {
                    return Err(last_err.into());
                }

                return Ok(Arc::new(schema));
            }
        }

        let Wrap(schema) = Wrap::<Schema>::from_py_object_bound(schema_obj.bind_borrowed(py))?;

        Ok(Arc::new(schema))
    })
}

pub fn to_dataset_scan(
    dataset_object: &PythonObject,
    limit: Option<usize>,
    projection: Option<&[PlSmallStr]>,
) -> PolarsResult<DslPlan> {
    Python::with_gil(|py| {
        let kwargs = PyDict::new(py);

        if let Some(limit) = limit {
            kwargs.set_item("limit", limit)?;
        }

        if let Some(projection) = projection {
            let projection_list = PyList::empty(py);

            for name in projection {
                projection_list.append(name.as_str())?;
            }

            kwargs.set_item("projection", projection_list)?;
        }

        let scan = dataset_object
            .getattr(py, "to_dataset_scan")?
            .call(py, (), Some(&kwargs))?;

        let Ok(lf) = get_lf(scan.bind(py)) else {
            return Err(
                PyValueError::new_err(format!("cannot extract LazyFrame from {}", &scan)).into(),
            );
        };

        Ok(lf.logical_plan)
    })
}
