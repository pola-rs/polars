use polars::prelude::CloudScheme;
use polars_io::cloud::CloudOptions;
use polars_io::cloud::credential_provider::PlCredentialProvider;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;

use crate::utils::to_py_err;

/// Interface to `class ScanOptions` on the Python side
pub struct PyStorageOptions<'py>(Bound<'py, PyAny>);

impl<'a, 'py> FromPyObject<'a, 'py> for PyStorageOptions<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(ob.to_owned()))
    }
}

impl PyStorageOptions<'_> {
    pub fn extract_cloud_options(
        &self,
        cloud_scheme: Option<CloudScheme>,
        credential_provider: Option<Py<PyAny>>,
    ) -> PyResult<CloudOptions> {
        if self.0.is_none() {
            return Ok(CloudOptions::default());
        }

        let py = self.0.py();

        let storage_options_dict: Bound<'_, PyDict> = self.0.extract()?;
        let mut storage_options: Vec<(PyBackedStr, String)> = Vec::with_capacity(
            storage_options_dict
                .call_method0(intern!(py, "__len__"))?
                .extract()?,
        );

        let mut retries: usize = 2;

        for v in storage_options_dict
            .call_method0(intern!(py, "items"))?
            .try_iter()?
        {
            let (key, value): (PyBackedStr, Bound<'_, PyAny>) = v?.extract()?;

            macro_rules! expected_type {
                ($key_name:expr, $type_name:expr) => {{
                    |_| {
                        let key_name = $key_name;
                        let type_name = $type_name;
                        PyValueError::new_err(format!(
                            "invalid value for '{key_name}': '{value}': (expected {type_name})"
                        ))
                    }
                }};
            }

            match &*key {
                "retries" => {
                    retries = value.extract().map_err(expected_type!("retries", "int"))?;
                },
                _ => {
                    let value: String = value.extract().map_err(expected_type!(&key, "str"))?;
                    storage_options.push((key, value))
                },
            }
        }

        let cloud_options = CloudOptions::from_untyped_config(cloud_scheme, storage_options)
            .map_err(to_py_err)?
            .with_max_retries(retries)
            .with_credential_provider(
                credential_provider.map(PlCredentialProvider::from_python_builder),
            );

        Ok(cloud_options)
    }
}
