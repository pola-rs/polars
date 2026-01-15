use std::io::BufReader;
use std::time::Duration;

#[cfg(any(feature = "ipc", feature = "parquet"))]
use polars::prelude::ArrowSchema;
use polars::prelude::CloudScheme;
use polars_io::cloud::CloudOptions;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::file::{EitherRustPythonFile, get_either_file};

#[cfg(feature = "ipc")]
#[pyfunction]
pub fn read_ipc_schema(py: Python<'_>, py_f: Py<PyAny>) -> PyResult<Bound<'_, PyDict>> {
    use arrow::io::ipc::read::read_file_metadata;
    let metadata = match get_either_file(py_f, false)? {
        EitherRustPythonFile::Rust(r) => {
            read_file_metadata(&mut BufReader::new(r)).map_err(PyPolarsErr::from)?
        },
        EitherRustPythonFile::Py(mut r) => read_file_metadata(&mut r).map_err(PyPolarsErr::from)?,
    };

    let dict = PyDict::new(py);
    fields_to_pydict(&metadata.schema, &dict)?;
    Ok(dict)
}

#[cfg(feature = "parquet")]
#[pyfunction]
pub fn read_parquet_metadata(
    py: Python,
    py_f: Py<PyAny>,
    storage_options: Option<Vec<(String, String)>>,
    credential_provider: Option<Py<PyAny>>,
    retries: usize,
    retry_config: Option<Py<PyAny>>,
) -> PyResult<Bound<PyDict>> {
    use std::io::Cursor;

    use polars_error::feature_gated;
    use polars_io::pl_async::get_runtime;
    use polars_parquet::read::read_metadata;
    use polars_parquet::read::schema::read_custom_key_value_metadata;

    use crate::file::{PythonScanSourceInput, get_python_scan_source_input};

    let metadata = match get_python_scan_source_input(py_f, false)? {
        PythonScanSourceInput::Buffer(buf) => {
            read_metadata(&mut Cursor::new(buf)).map_err(PyPolarsErr::from)?
        },
        PythonScanSourceInput::Path(p) => {
            let cloud_options = parse_cloud_options(
                CloudScheme::from_path(p.as_str()),
                storage_options,
                credential_provider,
                retries,
                retry_config,
            )?;

            if p.has_scheme() {
                feature_gated!("cloud", {
                    use polars::prelude::ParquetObjectStore;
                    use polars_error::PolarsResult;

                    py.detach(|| {
                        get_runtime().block_on(async {
                            let mut reader =
                                ParquetObjectStore::from_uri(p, cloud_options.as_ref(), None)
                                    .await?;
                            let result = reader.get_metadata().await?;
                            PolarsResult::Ok((**result).clone())
                        })
                    })
                })
                .map_err(PyPolarsErr::from)?
            } else {
                let file = polars_utils::open_file(p.as_std_path()).map_err(PyPolarsErr::from)?;
                read_metadata(&mut BufReader::new(file)).map_err(PyPolarsErr::from)?
            }
        },
        PythonScanSourceInput::File(f) => {
            read_metadata(&mut BufReader::new(f)).map_err(PyPolarsErr::from)?
        },
    };

    let key_value_metadata = read_custom_key_value_metadata(metadata.key_value_metadata());
    let dict = PyDict::new(py);
    for (key, value) in key_value_metadata.into_iter() {
        dict.set_item(key.as_str(), value.as_str())?;
    }
    Ok(dict)
}

#[cfg(any(feature = "ipc", feature = "parquet"))]
fn fields_to_pydict(schema: &ArrowSchema, dict: &Bound<'_, PyDict>) -> PyResult<()> {
    for field in schema.iter_values() {
        let dt = Wrap(polars::prelude::DataType::from_arrow_field(field));
        dict.set_item(field.name.as_str(), &dt)?;
    }
    Ok(())
}

#[cfg(feature = "clipboard")]
#[pyfunction]
pub fn read_clipboard_string() -> PyResult<String> {
    use arboard;
    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
    let result = clipboard
        .get_text()
        .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
    Ok(result)
}

#[cfg(feature = "clipboard")]
#[pyfunction]
pub fn write_clipboard_string(s: &str) -> PyResult<()> {
    use arboard;
    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
    clipboard
        .set_text(s)
        .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
    Ok(())
}

pub fn parse_cloud_options(
    cloud_scheme: Option<CloudScheme>,
    storage_options: Option<Vec<(String, String)>>,
    credential_provider: Option<Py<PyAny>>,
    retries: usize,
    retry_config: Option<Py<PyAny>>,
) -> PyResult<Option<CloudOptions>> {
    let result: Option<CloudOptions> = {
        #[cfg(feature = "cloud")]
        {
            use polars_io::cloud::credential_provider::PlCredentialProvider;

            use crate::prelude::parse_cloud_options;

            let cloud_options =
                parse_cloud_options(cloud_scheme, storage_options.unwrap_or_default())?;
            let retry_config = parse_retry_config(retry_config)?;

            let cloud_options = if let Some(retry_config) = retry_config {
                cloud_options.with_retry_config(retry_config)
            } else {
                cloud_options.with_max_retries(retries)
            };

            Some(cloud_options.with_credential_provider(
                credential_provider.map(PlCredentialProvider::from_python_builder),
            ))
        }

        #[cfg(not(feature = "cloud"))]
        {
            None
        }
    };
    Ok(result)
}

#[cfg(feature = "cloud")]
fn parse_retry_config(
    retry_config: Option<Py<PyAny>>,
) -> PyResult<Option<polars_io::cloud::RetryConfig>> {
    #[derive(FromPyObject)]
    struct PyBackoffConfig {
        init_backoff: f64,
        max_backoff: f64,
        base: f64,
    }

    #[derive(FromPyObject)]
    struct PyRetryConfig {
        backoff: PyBackoffConfig,
        max_retries: usize,
        retry_timeout: f64,
    }

    fn to_duration(value: f64, name: &str) -> PyResult<Duration> {
        if !value.is_finite() || value < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} must be a non-negative finite number of seconds"
            )));
        }
        Ok(Duration::from_secs_f64(value))
    }

    let Some(retry_config) = retry_config else {
        return Ok(None);
    };

    Python::attach(|py| {
        let config = retry_config.bind(py).extract::<PyRetryConfig>()?;
        let backoff = polars_io::cloud::RetryBackoffConfig {
            init_backoff: to_duration(config.backoff.init_backoff, "backoff.init_backoff")?,
            max_backoff: to_duration(config.backoff.max_backoff, "backoff.max_backoff")?,
            base: config.backoff.base,
        };

        Ok(Some(polars_io::cloud::RetryConfig {
            backoff,
            max_retries: config.max_retries,
            retry_timeout: to_duration(config.retry_timeout, "retry_timeout")?,
        }))
    })
}
