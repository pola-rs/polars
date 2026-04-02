use std::time::Duration;

use polars::prelude::CloudScheme;
use polars_core::config::verbose_print_sensitive;
use polars_io::cloud::{CloudOptions, CloudRetryConfig};
use polars_utils::total_ord::TotalOrdWrap;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;

use crate::utils::to_py_err;

/// Interface to `StorageOptionsDict | None` on the Python side
pub struct OptPyCloudOptions<'py>(Bound<'py, PyAny>);

impl<'a, 'py> FromPyObject<'a, 'py> for OptPyCloudOptions<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(ob.to_owned()))
    }
}

impl OptPyCloudOptions<'_> {
    pub fn extract_opt_cloud_options(
        &self,
        cloud_scheme: Option<CloudScheme>,
        credential_provider: Option<Py<PyAny>>,
    ) -> PyResult<Option<CloudOptions>> {
        if self.0.is_none() && credential_provider.is_none() {
            return Ok(None);
        }

        let py = self.0.py();

        let mut storage_options: Vec<(PyBackedStr, String)> = vec![];
        let mut file_cache_ttl: u64 = 2;
        let mut retry_config = CloudRetryConfig::default();

        let storage_options_dict: Option<Bound<'_, PyDict>> = self.0.extract()?;

        if let Some(storage_options_dict) = storage_options_dict {
            storage_options.reserve(
                storage_options_dict
                    .call_method0(intern!(py, "__len__"))?
                    .extract()?,
            );

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
                                "invalid value for '{key_name}': '{value}' (expected {type_name})"
                            ))
                        }
                    }};
                }

                match &*key {
                    "file_cache_ttl" => {
                        file_cache_ttl = value
                            .extract()
                            .map_err(expected_type!("file_cache_ttl", "int"))?;
                    },
                    "max_retries" => {
                        retry_config.max_retries = value
                            .extract()
                            .map_err(expected_type!("max_retries", "int"))?;
                    },
                    "retry_timeout_ms" => {
                        retry_config.retry_timeout = Some(Duration::from_millis(
                            value
                                .extract()
                                .map_err(expected_type!("retry_timeout", "int"))?,
                        ));
                    },
                    "retry_init_backoff_ms" => {
                        retry_config.retry_init_backoff = Some(Duration::from_millis(
                            value
                                .extract()
                                .map_err(expected_type!("retry_init_backoff", "int"))?,
                        ));
                    },
                    "retry_max_backoff_ms" => {
                        retry_config.retry_max_backoff = Some(Duration::from_millis(
                            value
                                .extract()
                                .map_err(expected_type!("retry_max_backoff", "int"))?,
                        ));
                    },
                    "retry_base_multiplier" => {
                        retry_config.retry_base_multiplier = Some(TotalOrdWrap(
                            value
                                .extract()
                                .map_err(expected_type!("retry_base_multiplier", "float"))?,
                        ));
                    },
                    _ => {
                        let value: String = value.extract().map_err(expected_type!(&key, "str"))?;
                        storage_options.push((key, value))
                    },
                }
            }
        }

        let cloud_options = CloudOptions::from_untyped_config(cloud_scheme, storage_options)
            .map_err(to_py_err)?
            .with_retry_config(retry_config);

        #[cfg(feature = "cloud")]
        let mut cloud_options =
            cloud_options.with_credential_provider(credential_provider.map(
                polars_io::cloud::credential_provider::PlCredentialProvider::from_python_builder,
            ));

        #[cfg(feature = "cloud")]
        if file_cache_ttl > 0 {
            cloud_options.file_cache_ttl = file_cache_ttl;
        }
        verbose_print_sensitive(|| format!("extracted cloud_options: {:?}", &cloud_options));

        Ok(Some(cloud_options))
    }
}
