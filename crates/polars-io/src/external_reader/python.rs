use std::sync::Arc;

use polars_error::PolarsResult;
use polars_utils::aliases::PlIndexMap;
use polars_utils::python_function::PythonObject;
use pyo3::types::{PyAnyMethods, PyDict};
use pyo3::{
    Borrowed, Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, Py, PyAny, PyErr, PyResult,
    Python, intern,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct PythonFileReaderBuilder {
    builder: Arc<PythonObject>,
}

impl PythonFileReaderBuilder {
    pub fn new(builder: Py<PyAny>) -> Self {
        Self {
            builder: Arc::new(PythonObject(builder)),
        }
    }

    pub fn builder(&self) -> &Arc<PythonObject> {
        &self.builder
    }

    pub fn explain_properties(&self) -> PolarsResult<PlIndexMap<String, String>> {
        Python::attach(|py| {
            let properties: Py<PyDict> = self
                .builder
                .call_method0(py, intern!(py, "explain_properties"))?
                .extract(py)?;

            let properties = properties.bind(py);

            let mut ret = PlIndexMap::default();

            for (k, v) in properties
                .try_iter()?
                .zip(properties.call_method0(intern!(py, "values"))?.try_iter()?)
            {
                let k: String = k?.extract()?;
                let v: String = v?.extract()?;

                ret.insert(k, v);
            }

            Ok(ret)
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PythonFileReaderBuilder {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self {
            builder: Arc::new(PythonObject(ob.into_py_any(ob.py())?)),
        })
    }
}

impl<'py> IntoPyObject<'py> for PythonFileReaderBuilder {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: pyo3::prelude::Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.builder.0.as_ref().clone_ref(py).into_bound(py))
    }
}

impl<'py> IntoPyObject<'py> for &PythonFileReaderBuilder {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: pyo3::prelude::Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.builder.0.clone_ref(py).into_bound(py))
    }
}
