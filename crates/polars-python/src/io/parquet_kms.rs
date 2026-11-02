//! Parquet modular encryption with keys managed by a KMS, using a KMS client implemented in Python.
//!
//! The KMS integration itself lives in `polars_io::parquet::encryption::kms`, so that it can be
//! serialized as part of a query plan.

use std::time::Duration;

use parquet_key_management::crypto_factory::{DecryptionConfiguration, EncryptionConfiguration};
pub use polars_io::parquet::encryption::kms::PyKmsConnectionConfig;
use polars_io::parquet::encryption::kms::{KmsDecryption, KmsEncryption, PythonCryptoFactory};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::io::parquet_encryption::{PyParquetDecryption, PyParquetEncryption};

#[pyclass(frozen)]
pub struct PyCryptoFactory {
    inner: PythonCryptoFactory,
}

#[pymethods]
impl PyCryptoFactory {
    #[new]
    fn new(kms_client_factory: Py<PyAny>) -> Self {
        Self {
            inner: PythonCryptoFactory::new(kms_client_factory),
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
            inner: KmsEncryption::new(
                crypto_factory.inner.clone(),
                kms_connection_config.config()?.clone(),
                encryption_config.inner.clone(),
            )
            .into(),
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
            inner: KmsDecryption::new(
                crypto_factory.inner.clone(),
                kms_connection_config.config()?.clone(),
                decryption_config.inner.clone(),
            )
            .into(),
        })
    }
}
