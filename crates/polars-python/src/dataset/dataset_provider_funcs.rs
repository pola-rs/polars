//! Note: Currently only used for Iceberg / Delta.
use std::sync::{Arc, LazyLock};

use polars::prelude::{DslPlan, PlSmallStr, Schema, SchemaRef};
use polars_core::config;
use polars_error::PolarsResult;
use polars_plan::plans::PyScanResolveThreadPool;
use polars_utils::python_function::PythonObject;
use pyo3::call::PyCallArgs;
use pyo3::conversion::FromPyObject;
use pyo3::exceptions::PyValueError;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyAnyMethods, PyBytes, PyDict, PyList, PyListMethods};
use pyo3::{Py, PyAny, PyResult, Python, intern};

use crate::interned;
use crate::interop::arrow::to_rust::field_to_rust;
use crate::prelude::{Wrap, get_lf};

pub fn name(dataset_object: &PythonObject) -> PlSmallStr {
    Python::attach(|py| {
        PyResult::Ok(PlSmallStr::from_str(
            &dataset_object
                .getattr(py, interned::DUNDER_CLASS.get(py))?
                .getattr(py, interned::DUNDER_NAME.get(py))?
                .extract::<PyBackedStr>(py)?,
        ))
    })
    .unwrap()
}

pub fn schema(
    dataset_object: &PythonObject,
    py_scan_resolve_threadpool: &PyScanResolveThreadPool,
) -> PolarsResult<SchemaRef> {
    Python::attach(|py| {
        let pyarrow_schema_cls = py
            .import("pyarrow")
            .ok()
            .and_then(|pa| pa.getattr("Schema").ok());

        let schema_obj = py_spawn_call(
            py,
            &dataset_object.getattr(py, "schema")?,
            (),
            None,
            py_scan_resolve_threadpool,
        )?;
        let schema_cls = schema_obj.getattr(py, interned::DUNDER_CLASS.get(py))?;

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

        let Wrap(schema) = Wrap::<Schema>::extract(schema_obj.bind_borrowed(py))?;

        Ok(Arc::new(schema))
    })
}

#[allow(clippy::too_many_arguments)]
pub fn to_dataset_scan(
    dataset_object: &PythonObject,
    existing_resolved_version_key: Option<&str>,
    limit: Option<usize>,
    projection: Option<&[PlSmallStr]>,
    filter_columns: Option<&[PlSmallStr]>,
    pyarrow_predicate: Option<&str>,
    serialized_predicate: Option<&[u8]>,
    py_scan_resolve_threadpool: &PyScanResolveThreadPool,
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

        if let Some(pyarrow_predicate) = pyarrow_predicate {
            kwargs.set_item(intern!(py, "pyarrow_predicate"), pyarrow_predicate)?;
        }

        let function = dataset_object.getattr(py, intern!(py, "to_dataset_scan"))?;

        // For a dataset that lowers filters into its own language rather than
        // the PyArrow subset. Only passed to a provider that asks for it: an
        // unexpected keyword would break providers written against an older
        // Polars.
        if let Some(serialized_predicate) = serialized_predicate
            && accepts_keyword(py, &function, "serialized_predicate")
        {
            kwargs.set_item(
                intern!(py, "serialized_predicate"),
                PyBytes::new(py, serialized_predicate),
            )?;
        }

        let Some((scan, version)): Option<(Py<PyAny>, Wrap<PlSmallStr>)> =
            py_spawn_call(py, &function, (), Some(&kwargs), py_scan_resolve_threadpool)?
                .extract(py)?
        else {
            return Ok(None);
        };

        let Ok(lf) = get_lf(scan.bind(py)) else {
            return Err(
                PyValueError::new_err(format!("cannot extract LazyFrame from {}", scan)).into(),
            );
        };

        Ok(Some((lf.logical_plan, version.0)))
    })
}

/// Whether `function` would accept `keyword`, either by name or via `**kwargs`.
///
/// A callable `inspect` cannot describe is treated as not accepting it.
fn accepts_keyword(py: Python<'_>, function: &Py<PyAny>, keyword: &str) -> bool {
    (|| {
        let inspect = py.import(intern!(py, "inspect"))?;
        let parameters = inspect
            .call_method1(intern!(py, "signature"), (function,))?
            .getattr(intern!(py, "parameters"))?;

        if parameters.contains(keyword)? {
            return PyResult::Ok(true);
        }

        let var_keyword = inspect
            .getattr(intern!(py, "Parameter"))?
            .getattr(intern!(py, "VAR_KEYWORD"))?;

        for parameter in parameters.call_method0(intern!(py, "values"))?.try_iter()? {
            if parameter?.getattr(intern!(py, "kind"))?.eq(&var_keyword)? {
                return PyResult::Ok(true);
            }
        }

        PyResult::Ok(false)
    })()
    .unwrap_or(false)
}

fn py_spawn_call<'a>(
    py: Python<'a>,
    function: &Py<PyAny>,
    args: impl PyCallArgs<'a>,
    kwargs: Option<&pyo3::Bound<'a, PyDict>>,
    py_scan_resolve_threadpool: &PyScanResolveThreadPool,
) -> PyResult<Py<PyAny>> {
    if LazyLock::get(&FN_POOL_WRAP_CLS).is_none() {
        // Initialization needs GIL, so we must release it to avoid deadlock.
        py.detach(|| {
            LazyLock::force(&FN_POOL_WRAP_CLS);
        })
    }

    return FN_POOL_WRAP_CLS
        .call1(py, (function, py_scan_resolve_threadpool))?
        .call(py, args, kwargs);

    static FN_POOL_WRAP_CLS: LazyLock<Py<PyAny>> = LazyLock::new(|| {
        Python::attach(|py| {
            (|| {
                PyResult::Ok(
                    py.import("polars._utils.threading")?
                        .getattr("FnPoolWrap")?
                        .unbind(),
                )
            })()
            .unwrap()
        })
    });
}
