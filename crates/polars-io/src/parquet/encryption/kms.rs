//! Parquet modular encryption with keys managed by a KMS, using a KMS client implemented in Python.
//!
//! The KMS configuration can be serialized, so that queries can be run elsewhere, such as on
//! Polars Cloud. The Python KMS client factory is pickled, and the KMS connection configuration
//! is serialized as is, including the key access token.

use std::sync::Arc;

use parquet_key_management::crypto_factory::{
    CryptoFactory, DecryptionConfiguration, EncryptionConfiguration,
};
use parquet_key_management::errors::Error as KmsError;
use parquet_key_management::key_unwrapper::KeyUnwrapper;
use parquet_key_management::kms::{
    KmsClient, KmsClientFactory, KmsClientRef, KmsConnectionConfig, KmsConnectionInfo,
};
use polars_core::runtime::ASYNC;
use polars_error::{PolarsError, PolarsResult, polars_err};
use polars_parquet::parquet::encryption::decrypt::{FileDecryptionProperties, KeyRetriever};
use polars_parquet::parquet::encryption::encrypt::FileEncryptionProperties;
use polars_parquet::parquet::error::{ParquetError, ParquetResult};
use polars_utils::aliases::PlHashMap;
use polars_utils::python_function::PythonObject;
use polars_utils::unique_id::UniqueId;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyMemoryView, PyString};

use super::{DecryptionPropertiesFactory, EncryptionPropertiesFactory};

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
                wrapped.extract::<String>()
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
            let key = key.bind(py);
            if let Ok(bytes) = key.cast::<PyBytes>() {
                return Ok(bytes.as_bytes().to_vec());
            }
            // Accept any other bytes-like object, such as a bytearray or memoryview.
            // This goes through a memoryview because `PyBuffer` isn't available with
            // the limited API before Python 3.11.
            let invalid_type = || {
                let type_name = key
                    .get_type()
                    .qualname()
                    .map_or_else(|_| "unknown".to_string(), |name| name.to_string());
                PyTypeError::new_err(format!(
                    "`unwrap_key` must return a bytes-like object, got '{type_name}'"
                ))
            };
            let view = PyMemoryView::from(key).map_err(|_| invalid_type())?;
            if view.getattr("itemsize")?.extract::<usize>()? != 1 {
                return Err(invalid_type());
            }
            let bytes = view.call_method0("tobytes")?;
            Ok(bytes.cast::<PyBytes>()?.as_bytes().to_vec())
        })
        .map_err(|err: PyErr| KmsError::General(err.to_string()))
    }
}

/// Creates KMS clients by calling a Python function with a `PyKmsConnectionConfig`.
struct PyKmsClientFactory(Arc<PythonObject>);

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
    Config(Arc<KmsConnectionConfig>),
    Info(KmsConnectionInfo),
}

/// Configuration for connecting to a KMS.
///
/// This is either a configuration created in Python, which can be used to read or write files,
/// or a read-only snapshot of the configuration passed to the KMS client factory.
#[pyclass(frozen)]
pub struct PyKmsConnectionConfig {
    inner: ConnectionConfig,
}

impl PyKmsConnectionConfig {
    /// The configuration used to read or write files.
    pub fn config(&self) -> PyResult<&Arc<KmsConnectionConfig>> {
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

/// A crypto factory that creates KMS clients by calling a Python function.
///
/// The function is called with a `PyKmsConnectionConfig`, and returns an object with
/// `wrap_key` and `unwrap_key` methods.
#[derive(Clone)]
pub struct PythonCryptoFactory {
    /// ID used for serialization, so that when the same factory is reused between
    /// different plan nodes, it can be shared again after deserialization (along with its caches).
    id: UniqueId,
    kms_client_factory: Arc<PythonObject>,
    crypto_factory: Arc<CryptoFactory>,
}

impl PythonCryptoFactory {
    pub fn new(kms_client_factory: Py<PyAny>) -> Self {
        Self::from_python_object(UniqueId::new(), Arc::new(PythonObject(kms_client_factory)))
    }

    fn from_python_object(id: UniqueId, kms_client_factory: Arc<PythonObject>) -> Self {
        let crypto_factory = Arc::new(CryptoFactory::new(PyKmsClientFactory(Arc::clone(
            &kms_client_factory,
        ))));
        Self {
            id,
            kms_client_factory,
            crypto_factory,
        }
    }
}

/// Generates new data encryption keys for each file, wrapped using the KMS.
pub struct KmsEncryption {
    crypto_factory: PythonCryptoFactory,
    kms_connection_config: Arc<KmsConnectionConfig>,
    encryption_configuration: EncryptionConfiguration,
}

impl KmsEncryption {
    pub fn new(
        crypto_factory: PythonCryptoFactory,
        kms_connection_config: Arc<KmsConnectionConfig>,
        encryption_configuration: EncryptionConfiguration,
    ) -> Self {
        Self {
            crypto_factory,
            kms_connection_config,
            encryption_configuration,
        }
    }
}

impl EncryptionPropertiesFactory for KmsEncryption {
    fn file_encryption_properties(&self) -> PolarsResult<Arc<FileEncryptionProperties>> {
        // Wrapping keys may block on KMS calls, or on another thread making the same
        // call, so let other tasks use this executor thread in the meantime.
        let keys = ASYNC
            .block_in_place(|| {
                self.crypto_factory.crypto_factory.file_encryption_keys(
                    Arc::clone(&self.kms_connection_config),
                    &self.encryption_configuration,
                )
            })
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
        // Unwrapping keys may block on KMS calls, or on another thread making the
        // same call, so let other tasks use this executor thread in the meantime.
        ASYNC
            .block_in_place(|| self.0.unwrap_key(key_metadata))
            .map_err(|err| ParquetError::Encryption(err.to_string()))
    }
}

/// Creates a new key unwrapper for each file.
pub struct KmsDecryption {
    crypto_factory: PythonCryptoFactory,
    kms_connection_config: Arc<KmsConnectionConfig>,
    decryption_configuration: DecryptionConfiguration,
}

impl KmsDecryption {
    pub fn new(
        crypto_factory: PythonCryptoFactory,
        kms_connection_config: Arc<KmsConnectionConfig>,
        decryption_configuration: DecryptionConfiguration,
    ) -> Self {
        Self {
            crypto_factory,
            kms_connection_config,
            decryption_configuration,
        }
    }
}

impl DecryptionPropertiesFactory for KmsDecryption {
    fn file_decryption_properties(&self) -> PolarsResult<Arc<FileDecryptionProperties>> {
        let key_unwrapper = self
            .crypto_factory
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

/// The first factory seen with each ID.
pub(super) type SeenCryptoFactories = PlHashMap<UniqueId, PythonCryptoFactory>;

impl PythonCryptoFactory {
    /// Returns the factory with the same ID that was seen first, if it isn't this one.
    fn shared(&self, seen: &mut SeenCryptoFactories) -> Option<Self> {
        let first = seen.entry(self.id).or_insert_with(|| self.clone());
        (!Arc::ptr_eq(&first.crypto_factory, &self.crypto_factory)).then(|| first.clone())
    }
}

impl KmsEncryption {
    /// Returns a copy using the crypto factory with the same ID that was seen first,
    /// if it doesn't already use it.
    pub(super) fn with_shared_crypto_factory(
        &self,
        seen: &mut SeenCryptoFactories,
    ) -> Option<Self> {
        Some(Self {
            crypto_factory: self.crypto_factory.shared(seen)?,
            kms_connection_config: Arc::clone(&self.kms_connection_config),
            encryption_configuration: self.encryption_configuration.clone(),
        })
    }
}

impl KmsDecryption {
    /// Returns a copy using the crypto factory with the same ID that was seen first,
    /// if it doesn't already use it.
    pub(super) fn with_shared_crypto_factory(
        &self,
        seen: &mut SeenCryptoFactories,
    ) -> Option<Self> {
        Some(Self {
            crypto_factory: self.crypto_factory.shared(seen)?,
            kms_connection_config: Arc::clone(&self.kms_connection_config),
            decryption_configuration: self.decryption_configuration.clone(),
        })
    }
}

/// Serialization of the KMS configuration.
///
/// The connection configuration is a snapshot, so refreshing the key access token
/// after serialization doesn't affect the deserialized configuration.
#[cfg(any(feature = "serde", feature = "dsl-schema"))]
#[cfg_attr(not(feature = "serde"), allow(dead_code))]
mod serialization {
    use std::collections::BTreeMap;
    use std::time::Duration;

    use super::*;

    // Maps are sorted so that serialization is deterministic.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "dsl-schema", schemars(rename = "KmsConnectionConfig"))]
    struct SerializableKmsConnectionConfig {
        kms_instance_id: String,
        kms_instance_url: String,
        key_access_token: String,
        custom_kms_conf: BTreeMap<String, String>,
    }

    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    #[cfg_attr(
        feature = "dsl-schema",
        schemars(rename = "KmsEncryptionConfiguration")
    )]
    struct SerializableEncryptionConfiguration {
        footer_key_id: String,
        column_key_ids: BTreeMap<String, Vec<String>>,
        plaintext_footer: bool,
        double_wrapping: bool,
        cache_lifetime: Option<Duration>,
        internal_key_material: bool,
        data_key_length_bits: u32,
    }

    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    #[cfg_attr(
        feature = "dsl-schema",
        schemars(rename = "KmsDecryptionConfiguration")
    )]
    struct SerializableDecryptionConfiguration {
        cache_lifetime: Option<Duration>,
        read_kms_url: bool,
    }

    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "dsl-schema", schemars(rename = "PythonCryptoFactory"))]
    struct SerializablePythonCryptoFactory {
        id: UniqueId,
        #[cfg_attr(
            feature = "serde",
            serde(
                serialize_with = "PythonObject::serialize_with_pyversion",
                deserialize_with = "PythonObject::deserialize_with_pyversion"
            )
        )]
        kms_client_factory: Arc<PythonObject>,
    }

    impl PythonCryptoFactory {
        fn to_serializable(&self) -> SerializablePythonCryptoFactory {
            SerializablePythonCryptoFactory {
                id: self.id,
                kms_client_factory: Arc::clone(&self.kms_client_factory),
            }
        }

        #[cfg(feature = "serde")]
        fn from_serializable(serializable: SerializablePythonCryptoFactory) -> Self {
            Self::from_python_object(serializable.id, serializable.kms_client_factory)
        }
    }

    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "dsl-schema", schemars(rename = "KmsEncryption"))]
    struct SerializableKmsEncryption {
        crypto_factory: SerializablePythonCryptoFactory,
        kms_connection_config: SerializableKmsConnectionConfig,
        encryption_configuration: SerializableEncryptionConfiguration,
    }

    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "dsl-schema", schemars(rename = "KmsDecryption"))]
    struct SerializableKmsDecryption {
        crypto_factory: SerializablePythonCryptoFactory,
        kms_connection_config: SerializableKmsConnectionConfig,
        decryption_configuration: SerializableDecryptionConfiguration,
    }

    impl SerializableKmsConnectionConfig {
        fn new(config: &KmsConnectionConfig) -> Self {
            Self {
                kms_instance_id: config.kms_instance_id().to_string(),
                kms_instance_url: config.kms_instance_url().to_string(),
                key_access_token: config.key_access_token(),
                custom_kms_conf: config
                    .custom_kms_conf()
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            }
        }

        fn into_config(self) -> Arc<KmsConnectionConfig> {
            Arc::new(
                KmsConnectionConfig::builder()
                    .set_kms_instance_id(self.kms_instance_id)
                    .set_kms_instance_url(self.kms_instance_url)
                    .set_key_access_token(self.key_access_token)
                    .set_custom_kms_conf(self.custom_kms_conf.into_iter().collect())
                    .build(),
            )
        }
    }

    impl SerializableEncryptionConfiguration {
        fn new(config: &EncryptionConfiguration) -> Self {
            Self {
                footer_key_id: config.footer_key_id().to_string(),
                column_key_ids: config
                    .column_key_ids()
                    .iter()
                    .map(|(key_id, columns)| (key_id.clone(), columns.clone()))
                    .collect(),
                plaintext_footer: config.plaintext_footer(),
                double_wrapping: config.double_wrapping(),
                cache_lifetime: config.cache_lifetime(),
                internal_key_material: config.internal_key_material(),
                data_key_length_bits: config.data_key_length_bits(),
            }
        }

        fn into_config(self) -> parquet_key_management::errors::Result<EncryptionConfiguration> {
            let mut builder = EncryptionConfiguration::builder(self.footer_key_id)
                .set_plaintext_footer(self.plaintext_footer)
                .set_double_wrapping(self.double_wrapping)
                .set_cache_lifetime(self.cache_lifetime);
            for (key_id, columns) in self.column_key_ids {
                builder = builder.add_column_key(key_id, columns);
            }
            let config = builder.build()?;
            // The builder can't set these yet, so check that they have the values that
            // were serialized, rather than silently changing them.
            if config.internal_key_material() != self.internal_key_material
                || config.data_key_length_bits() != self.data_key_length_bits
            {
                return Err(KmsError::General(format!(
                    "cannot deserialize a KMS encryption configuration with \
                    internal_key_material={} and data_key_length_bits={}",
                    self.internal_key_material, self.data_key_length_bits
                )));
            }
            Ok(config)
        }
    }

    impl SerializableDecryptionConfiguration {
        fn new(config: &DecryptionConfiguration) -> Self {
            Self {
                cache_lifetime: config.cache_lifetime(),
                read_kms_url: config.read_kms_url(),
            }
        }

        fn into_config(self) -> DecryptionConfiguration {
            DecryptionConfiguration::builder()
                .set_cache_lifetime(self.cache_lifetime)
                .set_read_kms_url(self.read_kms_url)
                .build()
        }
    }

    #[cfg(feature = "serde")]
    impl serde::Serialize for KmsEncryption {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            SerializableKmsEncryption {
                crypto_factory: self.crypto_factory.to_serializable(),
                kms_connection_config: SerializableKmsConnectionConfig::new(
                    &self.kms_connection_config,
                ),
                encryption_configuration: SerializableEncryptionConfiguration::new(
                    &self.encryption_configuration,
                ),
            }
            .serialize(serializer)
        }
    }

    #[cfg(feature = "serde")]
    impl<'de> serde::Deserialize<'de> for KmsEncryption {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            use serde::de::Error;

            let encryption = SerializableKmsEncryption::deserialize(deserializer)?;
            Ok(Self::new(
                PythonCryptoFactory::from_serializable(encryption.crypto_factory),
                encryption.kms_connection_config.into_config(),
                encryption
                    .encryption_configuration
                    .into_config()
                    .map_err(D::Error::custom)?,
            ))
        }
    }

    #[cfg(feature = "serde")]
    impl serde::Serialize for KmsDecryption {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            SerializableKmsDecryption {
                crypto_factory: self.crypto_factory.to_serializable(),
                kms_connection_config: SerializableKmsConnectionConfig::new(
                    &self.kms_connection_config,
                ),
                decryption_configuration: SerializableDecryptionConfiguration::new(
                    &self.decryption_configuration,
                ),
            }
            .serialize(serializer)
        }
    }

    #[cfg(feature = "serde")]
    impl<'de> serde::Deserialize<'de> for KmsDecryption {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let decryption = SerializableKmsDecryption::deserialize(deserializer)?;
            Ok(Self::new(
                PythonCryptoFactory::from_serializable(decryption.crypto_factory),
                decryption.kms_connection_config.into_config(),
                decryption.decryption_configuration.into_config(),
            ))
        }
    }

    #[cfg(feature = "dsl-schema")]
    impl schemars::JsonSchema for KmsEncryption {
        fn schema_name() -> std::borrow::Cow<'static, str> {
            SerializableKmsEncryption::schema_name()
        }

        fn schema_id() -> std::borrow::Cow<'static, str> {
            SerializableKmsEncryption::schema_id()
        }

        fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
            SerializableKmsEncryption::json_schema(generator)
        }
    }

    #[cfg(feature = "dsl-schema")]
    impl schemars::JsonSchema for KmsDecryption {
        fn schema_name() -> std::borrow::Cow<'static, str> {
            SerializableKmsDecryption::schema_name()
        }

        fn schema_id() -> std::borrow::Cow<'static, str> {
            SerializableKmsDecryption::schema_id()
        }

        fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
            SerializableKmsDecryption::json_schema(generator)
        }
    }
}
