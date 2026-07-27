use std::any::Any;
use std::string::ToString;
use std::sync::Arc;

use polars_error::PolarsError;
use polars_observer::{
    NoopQueryMetrics, PlannedQuery, QueryMetrics, QueryObserver, QueryObserverFactory,
    set_query_observer_factory,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use uuid::Uuid;

#[pyclass(name = "CloudStreamingMetricsHandle")]
pub struct CloudStreamingMetricsHandle {
    metrics: Box<dyn QueryMetrics>,
}

const POLARS_CLOUD_PACKAGE_NAME: &str = "polars_cloud";

#[pymethods]
impl CloudStreamingMetricsHandle {
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let rows = self.metrics.snapshot();
        let bytes =
            rmp_serde::to_vec_named(&rows).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes))
    }
}

struct CloudObserverFactory {
    observer: Arc<Py<PyAny>>,
}

impl CloudObserverFactory {
    fn new(observer: Py<PyAny>) -> Self {
        Self {
            observer: Arc::new(observer),
        }
    }
}

impl QueryObserverFactory for CloudObserverFactory {
    fn new_observer(&self) -> Box<dyn QueryObserver> {
        Box::new(CloudObserver::new(self.observer.clone()))
    }
}

pub type QueryId = Uuid;

struct CloudObserver {
    observer: Arc<Py<PyAny>>,
    query_id: QueryId,
}

impl CloudObserver {
    fn new(observer: Arc<Py<PyAny>>) -> Self {
        Self {
            observer,
            query_id: Uuid::new_v4(),
        }
    }
}

struct CloudObserverGuard {
    py_guard: Option<Py<PyAny>>,
}

impl Drop for CloudObserverGuard {
    fn drop(&mut self) {
        let Some(guard) = self.py_guard.take() else {
            return;
        };
        Python::attach(|py| {
            if let Err(err) = guard.call_method0(py, "close") {
                err.print(py);
            }
        });
    }
}

impl QueryObserver for CloudObserver {
    fn on_query_started(&self) {
        Python::attach(|py| {
            if let Err(err) = self
                .observer
                .call_method1(py, "on_query_started", (self.query_id,))
            {
                err.print(py);
            }
        });
    }

    fn on_query_planned(&self, query: PlannedQuery) -> Box<dyn Any + Send> {
        let ir_bytes = rmp_serde::to_vec_named(&query.ir).unwrap_or_default();
        let phys_bytes = rmp_serde::to_vec_named(&query.physical).unwrap_or_default();

        let py_guard: Option<Py<PyAny>> = Python::attach(|py| -> PyResult<Option<Py<PyAny>>> {
            let handle = CloudStreamingMetricsHandle {
                metrics: query.metrics.unwrap_or_else(|| Box::new(NoopQueryMetrics)),
            };
            let ir_py = PyBytes::new(py, &ir_bytes);
            let phys_py = PyBytes::new(py, &phys_bytes);
            let ret = self.observer.call_method1(
                py,
                "on_query_planned",
                (self.query_id, handle, ir_py, phys_py),
            )?;

            Ok((!ret.is_none(py)).then_some(ret))
        })
        .unwrap_or_else(|err| {
            Python::attach(|py| err.print(py));
            None
        });

        Box::new(CloudObserverGuard { py_guard }) as Box<dyn Any + Send>
    }

    fn on_query_failed(&self, err: &PolarsError) {
        Python::attach(|py| {
            if let Err(err) =
                self.observer
                    .call_method1(py, "on_query_failed", (self.query_id, err.to_string()))
            {
                err.print(py);
            }
        });
    }
}

#[pyfunction]
pub fn set_query_monitoring(py: Python<'_>, enable: bool) -> PyResult<()> {
    if !enable {
        set_query_observer_factory(None);
        return Ok(());
    }

    let module = py.import(POLARS_CLOUD_PACKAGE_NAME).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "query monitoring requires the `polars_cloud>=0.11.0` package, which could not be imported. \
             Install it into this environment (e.g. `pip install 'polars-cloud>=0.11.0'`). ({e})",
        ))
    })?;
    let cls = module.getattr("QueryCloudObserver").map_err(|_| {
        PyRuntimeError::new_err(
            "the installed `polars_cloud` is incompatible with this Polars build.\
             Ensure the polars_cloud and polars versions match.",
        )
    })?;
    let observer = cls
        .call0()
        .map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to construct the Polars Cloud observer: {e}"
            ))
        })?
        .unbind();

    set_query_observer_factory(Some(Arc::new(CloudObserverFactory::new(observer))));
    Ok(())
}
