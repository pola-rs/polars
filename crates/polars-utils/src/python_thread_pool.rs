use pyo3::call::PyCallArgs;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAnyMethods as _, PyDict};
use pyo3::{Bound, IntoPyObject, Py, PyAny, PyResult, Python};

#[derive(IntoPyObject)]
pub struct PyThreadPool(
    /// polars._utils.threading.PyThreadPool
    Py<PyAny>,
);

impl<'py> IntoPyObject<'py> for &'py PyThreadPool {
    type Output = <&'py Py<PyAny> as IntoPyObject<'py>>::Output;
    type Target = <&'py Py<PyAny> as IntoPyObject<'py>>::Target;
    type Error = <&'py Py<PyAny> as IntoPyObject<'py>>::Error;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        IntoPyObject::into_pyobject(&self.0, py)
    }
}

impl PyThreadPool {
    pub fn new() -> Self {
        use std::num::NonZeroUsize;

        Python::attach(|py| {
            let num_threads =
                std::env::var("POLARS_PYTHON_SCAN_RESOLVE_THREADS").map_or(128, |x| {
                    x.parse::<NonZeroUsize>()
                        .unwrap_or_else(|_| {
                            panic!("invalid value for POLARS_PYTHON_SCAN_RESOLVE_THREADS: {x}")
                        })
                        .get()
                });

            if polars_config::config().verbose() {
                eprintln!("python scan_resolve_threadpool threads: {num_threads}")
            }

            return Self(
                py_scan_resolve_threadpool_cls(py)
                    .bind(py)
                    .call1((num_threads,))
                    .map(|x| x.unbind())
                    .unwrap(),
            );

            fn py_scan_resolve_threadpool_cls(py: Python<'_>) -> &'static Py<PyAny> {
                static CLS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

                CLS.get_or_init(py, || {
                    py.import("polars._utils.threading")
                        .unwrap()
                        .getattr("PyThreadPool")
                        .unwrap()
                        .unbind()
                })
            }
        })
    }

    pub fn spawn_call<'a>(
        &self,
        py: Python<'a>,
        function: &Py<PyAny>,
        args: impl PyCallArgs<'a>,
        kwargs: Option<&Bound<'a, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        static CLS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

        CLS.get_or_init(py, || {
            py.import("polars._utils.threading")
                .unwrap()
                .getattr("FnPoolWrap")
                .unwrap()
                .unbind()
        })
        .call1(py, (function, self))?
        .call(py, args, kwargs)
    }
}

impl Default for PyThreadPool {
    fn default() -> Self {
        Self::new()
    }
}
