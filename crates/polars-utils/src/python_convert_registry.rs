use std::any::Any;
use std::ops::Deref;
use std::sync::{Arc, LazyLock, RwLock};

use pyo3::types::PyAnyMethods;
use pyo3::{Py, PyAny, PyResult, Python};

pub type FromPython = Arc<dyn Fn(Py<PyAny>) -> PyResult<Box<dyn Any>> + Send + Sync>;
pub type ToPython = Arc<dyn for<'a> Fn(&'a dyn Any) -> PyResult<Py<PyAny>> + Send + Sync>;

#[derive(Clone)]
pub struct FromPythonConvertRegistry {
    pub file_provider_result: FromPython,
    pub series: FromPython,
    pub df: FromPython,
    pub dsl_plan: FromPython,
    pub schema: FromPython,
}

#[derive(Clone)]
pub struct ToPythonConvertRegistry {
    pub df: ToPython,
    pub series: ToPython,
    pub dsl_plan: ToPython,
    pub schema: ToPython,
}

impl ToPythonConvertRegistry {
    /// Convert a Rust `DataFrame` to a Python `pl.DataFrame` object.
    pub fn df_to_wrapped_pydf(&self, df: &dyn Any) -> PyResult<Py<PyAny>> {
        static WRAP_DF: LazyLock<Py<PyAny>> = LazyLock::new(|| {
            Python::attach(|py| {
                py.import("polars._utils.wrap")
                    .unwrap()
                    .getattr("wrap_df")
                    .unwrap()
                    .unbind()
            })
        });

        let pydf = (self.df)(df)?;

        Python::attach(|py| WRAP_DF.call1(py, (pydf,)))
    }
}

#[derive(Clone)]
pub struct PythonConvertRegistry {
    pub from_py: FromPythonConvertRegistry,
    pub to_py: ToPythonConvertRegistry,
}

impl PythonConvertRegistry {
    pub fn py_file_provider_args_dataclass(&self) -> &'static Py<PyAny> {
        static CLS: LazyLock<Py<PyAny>> = LazyLock::new(|| {
            Python::attach(|py| {
                py.import("polars.io.partition")
                    .unwrap()
                    .getattr("FileProviderArgs")
                    .unwrap()
                    .unbind()
            })
        });

        &CLS
    }
}

static PYTHON_CONVERT_REGISTRY: LazyLock<RwLock<Option<PythonConvertRegistry>>> =
    LazyLock::new(Default::default);

pub fn get_python_convert_registry() -> PythonConvertRegistry {
    PYTHON_CONVERT_REGISTRY
        .deref()
        .read()
        .unwrap()
        .as_ref()
        .unwrap()
        .clone()
}

pub fn register_converters(registry: PythonConvertRegistry) {
    *PYTHON_CONVERT_REGISTRY.deref().write().unwrap() = Some(registry);
}
