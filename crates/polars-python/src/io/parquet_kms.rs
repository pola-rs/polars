//! Parquet modular encryption with keys managed by a KMS, using a KMS client implemented in Python.

use std::sync::Arc;
use std::time::Duration;

use parquet_key_management::crypto_factory::{
    CryptoFactory, DecryptionConfiguration, EncryptionConfiguration,
};
use parquet_key_management::errors::Error as KmsError;
use parquet_key_management::key_unwrapper::KeyUnwrapper;
use parquet_key_management::kms::{
    KmsClient, KmsClientFactory, KmsClientRef, KmsConnectionConfig, KmsConnectionInfo,
};
use polars::prelude::{PolarsError, PolarsResult, polars_err};
use polars_io::parquet::encryption::{
    DecryptionPropertiesFactory, EncryptionPropertiesFactory, ParquetDecryption, ParquetEncryption,
};
use polars_parquet::parquet::encryption::decrypt::{FileDecryptionProperties, KeyRetriever};
use polars_parquet::parquet::encryption::encrypt::FileEncryptionProperties;
use polars_parquet::parquet::error::{ParquetError, ParquetResult};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyString};

use crate::io::parquet_encryption::{PyParquetDecryption, PyParquetEncryption};

fn kms_error_to_polars(err: KmsError) -> PolarsError {
    polars_err!(ComputeError: "{}", err)
}

/// A KMS client implemented in Python, with `wrap_key` and `unwrap_key` methods.
struct PyKmsClient(Py<PyAny>);

impl KmsClient for PyKmsClient {
    fn wrap_key(
        &self,
        key_bytes: &[u8],
        master_key_identifier: &str,
    ) -> parquet_key_management::errors::Result<String> {
        Python::attach(|py| {
            let wrapped = self.0.call_method1(
                py,
                "wrap_key",
                (PyBytes::new(py, key_bytes), master_key_identifier),
            )?;
            // Allow returning bytes, as PyArrow does.
            let wrapped = wrapped.bind(py);
            if let Ok(wrapped) = wrapped.cast::<PyBytes>() {
                String::from_utf8(wrapped.as_bytes().to_vec())
                    .map_err(|_| PyValueError::new_err("wrapped key bytes must be valid UTF-8"))
            } else {
                wrapped.extract::<String>().map_err(PyErr::from)
            }
        })
        .map_err(|err: PyErr| KmsError::General(err.to_string()))
    }

    fn unwrap_key(
        &self,
        wrapped_key: &str,
        master_key_identifier: &str,
    ) -> parquet_key_management::errors::Result<Vec<u8>> {
        Python::attach(|py| {
            let key =
                self.0
                    .call_method1(py, "unwrap_key", (wrapped_key, master_key_identifier))?;
            // Accept any bytes-like object, such as a bytearray or memoryview.
            let key = key.bind(py);
            let buffer = PyBuffer::<u8>::get(key).map_err(|_| {
                let type_name = key
                    .get_type()
                    .qualname()
                    .map_or_else(|_| "unknown".to_string(), |name| name.to_string());
                PyTypeError::new_err(format!(
                    "`unwrap_key` must return a bytes-like object, got '{type_name}'"
                ))
            })?;
            buffer.to_vec(py)
        })
        .map_err(|err: PyErr| KmsError::General(err.to_string()))
    }
}

/// Creates KMS clients by calling a Python function with a `KmsConnectionConfig`.
struct PyKmsClientFactory(Py<PyAny>);

impl KmsClientFactory for PyKmsClientFactory {
    fn create_client(
        &self,
        kms_connection_info: &KmsConnectionInfo,
    ) -> parquet_key_management::errors::Result<KmsClientRef> {
        let config = PyKmsConnectionConfig {
            inner: ConnectionConfig::Info(kms_connection_info.clone()),
        };
        Python::attach(|py| {
            let client = self.0.call1(py, (config,))?;
            Ok(Arc::new(PyKmsClient(client)) as KmsClientRef)
        })
        .map_err(|err: PyErr| KmsError::General(err.to_string()))
    }
}

enum ConnectionConfig {
    /// A configuration created by the user, used when reading or writing files.
    Config(Arc<KmsConnectionConfig>),
    /// A read-only snapshot of a configuration, passed to the KMS client factory.
    Info(KmsConnectionInfo),
}

#[pyclass(frozen)]
pub struct PyKmsConnectionConfig {
    inner: ConnectionConfig,
}

impl PyKmsConnectionConfig {
    fn config(&self) -> PyResult<&Arc<KmsConnectionConfig>> {
        match &self.inner {
            ConnectionConfig::Config(config) => Ok(config),
            ConnectionConfig::Info(_) => Err(PyValueError::new_err(
                "the KMS connection configuration passed to the KMS client factory \
                can't be used to read or write files",
            )),
        }
    }
}

#[pymethods]
impl PyKmsConnectionConfig {
    #[new]
    #[pyo3(signature = (kms_instance_id, kms_instance_url, key_access_token, custom_kms_conf))]
    fn new(
        kms_instance_id: String,
        kms_instance_url: String,
        key_access_token: String,
        custom_kms_conf: Vec<(String, String)>,
    ) -> Self {
        let mut builder = KmsConnectionConfig::builder()
            .set_kms_instance_id(kms_instance_id)
            .set_kms_instance_url(kms_instance_url)
            .set_key_access_token(key_access_token);
        for (key, value) in custom_kms_conf {
            builder = builder.set_custom_kms_conf_option(key, value);
        }
        Self {
            inner: ConnectionConfig::Config(Arc::new(builder.build())),
        }
    }

    #[getter]
    fn kms_instance_id(&self) -> &str {
        match &self.inner {
            ConnectionConfig::Config(config) => config.kms_instance_id(),
            ConnectionConfig::Info(info) => info.kms_instance_id(),
        }
    }

    #[getter]
    fn kms_instance_url(&self) -> &str {
        match &self.inner {
            ConnectionConfig::Config(config) => config.kms_instance_url(),
            ConnectionConfig::Info(info) => info.kms_instance_url(),
        }
    }

    #[getter]
    fn key_access_token(&self) -> String {
        match &self.inner {
            ConnectionConfig::Config(config) => config.key_access_token(),
            ConnectionConfig::Info(info) => info.key_access_token().to_owned(),
        }
    }

    #[getter]
    fn custom_kms_conf<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let custom_kms_conf = match &self.inner {
            ConnectionConfig::Config(config) => config.custom_kms_conf(),
            ConnectionConfig::Info(info) => info.custom_kms_conf(),
        };
        let dict = PyDict::new(py);
        for (key, value) in custom_kms_conf {
            dict.set_item(PyString::new(py, key), PyString::new(py, value))?;
        }
        Ok(dict)
    }

    fn refresh_key_access_token(&self, value: String) -> PyResult<()> {
        match &self.inner {
            ConnectionConfig::Config(config) => {
                config.refresh_key_access_token(value);
                Ok(())
            },
            ConnectionConfig::Info(_) => Err(PyValueError::new_err(
                "The key access token cannot be refreshed using the read-only KMS connection \
                configuration passed to the KMS client factory; refresh the token using \
                the original configuration",
            )),
        }
    }
}

#[pyclass(frozen)]
pub struct PyCryptoFactory {
    inner: Arc<CryptoFactory>,
}

#[pymethods]
impl PyCryptoFactory {
    #[new]
    fn new(kms_client_factory: Py<PyAny>) -> Self {
        Self {
            inner: Arc::new(CryptoFactory::new(PyKmsClientFactory(kms_client_factory))),
        }
    }
}

#[pyclass(frozen)]
pub struct PyEncryptionConfiguration {
    inner: EncryptionConfiguration,
}

#[pymethods]
impl PyEncryptionConfiguration {
    /// With no column keys, all columns are encrypted with the footer key.
    #[new]
    #[pyo3(signature = (footer_key, column_keys, plaintext_footer, double_wrapping, cache_lifetime))]
    fn new(
        footer_key: String,
        column_keys: Vec<(String, Vec<String>)>,
        plaintext_footer: bool,
        double_wrapping: bool,
        cache_lifetime: Option<Duration>,
    ) -> PyResult<Self> {
        let mut builder = EncryptionConfiguration::builder(footer_key)
            .set_plaintext_footer(plaintext_footer)
            .set_double_wrapping(double_wrapping)
            .set_cache_lifetime(cache_lifetime);
        for (master_key_id, columns) in column_keys {
            builder = builder.add_column_key(master_key_id, columns);
        }
        let inner = builder
            .build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }
}

#[pyclass(frozen)]
pub struct PyDecryptionConfiguration {
    inner: DecryptionConfiguration,
}

#[pymethods]
impl PyDecryptionConfiguration {
    #[new]
    #[pyo3(signature = (cache_lifetime, read_kms_url))]
    fn new(cache_lifetime: Option<Duration>, read_kms_url: bool) -> Self {
        Self {
            inner: DecryptionConfiguration::builder()
                .set_cache_lifetime(cache_lifetime)
                .set_read_kms_url(read_kms_url)
                .build(),
        }
    }
}

/// Generates new data encryption keys for each file, wrapped using the KMS.
struct KmsEncryptionFactory {
    crypto_factory: Arc<CryptoFactory>,
    kms_connection_config: Arc<KmsConnectionConfig>,
    encryption_configuration: EncryptionConfiguration,
}

impl EncryptionPropertiesFactory for KmsEncryptionFactory {
    fn file_encryption_properties(&self) -> PolarsResult<Arc<FileEncryptionProperties>> {
        let keys = self
            .crypto_factory
            .file_encryption_keys(
                Arc::clone(&self.kms_connection_config),
                &self.encryption_configuration,
            )
            .map_err(kms_error_to_polars)?;
        let footer_key = keys.footer_key();
        let mut builder = FileEncryptionProperties::builder(footer_key.key().to_vec())
            .with_footer_key_metadata(footer_key.metadata().to_vec())
            .with_plaintext_footer(keys.plaintext_footer());
        for (column_path, column_key) in keys.column_keys() {
            builder = builder.with_column_key_and_metadata(
                column_path,
                column_key.key().to_vec(),
                column_key.metadata().to_vec(),
            );
        }
        Ok(builder.build()?)
    }

    fn encrypted_column_names(&self) -> Vec<String> {
        self.encryption_configuration
            .column_key_ids()
            .values()
            .flatten()
            .cloned()
            .collect()
    }
}

/// Retrieves keys by unwrapping them with the KMS.
struct KeyUnwrapperRetriever(KeyUnwrapper);

impl KeyRetriever for KeyUnwrapperRetriever {
    fn retrieve_key(&self, key_metadata: &[u8]) -> ParquetResult<Vec<u8>> {
        self.0
            .unwrap_key(key_metadata)
            .map_err(|err| ParquetError::Encryption(err.to_string()))
    }
}

/// Creates a new key unwrapper for each file.
struct KmsDecryptionFactory {
    crypto_factory: Arc<CryptoFactory>,
    kms_connection_config: Arc<KmsConnectionConfig>,
    decryption_configuration: DecryptionConfiguration,
}

impl DecryptionPropertiesFactory for KmsDecryptionFactory {
    fn file_decryption_properties(&self) -> PolarsResult<Arc<FileDecryptionProperties>> {
        let key_unwrapper = self
            .crypto_factory
            .key_unwrapper(
                Arc::clone(&self.kms_connection_config),
                self.decryption_configuration.clone(),
            )
            .map_err(kms_error_to_polars)?;
        Ok(
            FileDecryptionProperties::with_key_retriever(Arc::new(KeyUnwrapperRetriever(
                key_unwrapper,
            )))
            .build()?,
        )
    }
}

#[pymethods]
impl PyParquetEncryption {
    /// Encrypt files using keys managed by a KMS.
    #[staticmethod]
    fn from_kms(
        crypto_factory: &PyCryptoFactory,
        kms_connection_config: &PyKmsConnectionConfig,
        encryption_config: &PyEncryptionConfiguration,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ParquetEncryption::new(KmsEncryptionFactory {
                crypto_factory: Arc::clone(&crypto_factory.inner),
                kms_connection_config: Arc::clone(kms_connection_config.config()?),
                encryption_configuration: encryption_config.inner.clone(),
            }),
        })
    }
}

#[pymethods]
impl PyParquetDecryption {
    /// Decrypt files using keys managed by a KMS.
    #[staticmethod]
    fn from_kms(
        crypto_factory: &PyCryptoFactory,
        kms_connection_config: &PyKmsConnectionConfig,
        decryption_config: &PyDecryptionConfiguration,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ParquetDecryption::new(KmsDecryptionFactory {
                crypto_factory: Arc::clone(&crypto_factory.inner),
                kms_connection_config: Arc::clone(kms_connection_config.config()?),
                decryption_configuration: decryption_config.inner.clone(),
            }),
        })
    }
}
