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
use pyo3::{Py, PyAny, PyResult, Python, intern};

use crate::interop::arrow::to_rust::field_to_rust;
use crate::prelude::{Wrap, get_lf};

pub fn name(dataset_object: &PythonObject) -> PlSmallStr {
    Python::attach(|py| {
        PyResult::Ok(PlSmallStr::from_str(
            &dataset_object
                .getattr(py, intern!(py, "__class__"))?
                .getattr(py, intern!(py, "__name__"))?
                .extract::<PyBackedStr>(py)?,
        ))
    })
    .unwrap()
}

pub fn schema(dataset_object: &PythonObject) -> PolarsResult<SchemaRef> {
    Python::attach(|py| {
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
    existing_resolved_version_key: Option<&str>,
    limit: Option<usize>,
    projection: Option<&[PlSmallStr]>,
    filter_columns: Option<&[PlSmallStr]>,
) -> PolarsResult<Option<(DslPlan, PlSmallStr)>> {
    Python::attach(|py| {
        let kwargs = PyDict::new(py);

        kwargs.set_item(
            intern!(py, "existing_resolved_version_key"),
            existing_resolved_version_key,
        )?;

        if let Some(limit) = limit {
            kwargs.set_item(intern!(py, "limit"), limit)?;
        }

        if let Some(projection) = projection {
            let projection_list = PyList::empty(py);

            for name in projection {
                projection_list.append(name.as_str())?;
            }

            kwargs.set_item(intern!(py, "projection"), projection_list)?;
        }

        if let Some(filter_columns) = filter_columns {
            let filter_columns_list = PyList::empty(py);

            for name in filter_columns {
                filter_columns_list.append(name.as_str())?;
            }

            kwargs.set_item(intern!(py, "filter_columns"), filter_columns_list)?;
        }

        let Some((scan, version)): Option<(Py<PyAny>, Wrap<PlSmallStr>)> = dataset_object
            .getattr(py, intern!(py, "to_dataset_scan"))?
            .call(py, (), Some(&kwargs))?
            .extract(py)?
        else {
            return Ok(None);
        };

        let Ok(lf) = get_lf(scan.bind(py)) else {
            return Err(
                PyValueError::new_err(format!("cannot extract LazyFrame from {}", &scan)).into(),
            );
        };

        Ok(Some((lf.logical_plan, version.0)))
    })
}
