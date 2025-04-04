use std::any::Any;
use std::ops::Deref;
use std::sync::{Arc, LazyLock, RwLock};

use pyo3::{Py, PyAny, PyResult};

pub type PythonToSinkTarget = Arc<dyn Fn(Py<PyAny>) -> PyResult<Box<dyn Any>> + Send + Sync>;

#[derive(Clone)]
pub struct FromPythonConvertRegistry {
    pub sink_target: PythonToSinkTarget,
}

#[derive(Clone)]
pub struct PythonConvertRegistry {
    pub from_py: FromPythonConvertRegistry,
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
