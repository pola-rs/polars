use std::collections::HashMap;

use aws_config::BehaviorVersion;
use aws_sdk_kms::Client;
use aws_sdk_kms::config::{Credentials, Region};
use aws_sdk_kms::primitives::Blob;
use aws_sdk_kms::types::EncryptionAlgorithmSpec;
use polars_parquet::parquet::encryption::AesGcmDecryptor;
use polars_parquet::parquet::encryption::iceberg::decrypt_stream;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::exceptions::ComputeError;

#[pyclass]
pub struct PyIcebergKmsClient {
    client: Client,
    algorithm: EncryptionAlgorithmSpec,
}

#[pymethods]
impl PyIcebergKmsClient {
    #[new]
    fn new(py: Python<'_>, properties: HashMap<String, String>) -> PyResult<Self> {
        let algorithm = EncryptionAlgorithmSpec::try_parse(
            properties
                .get("kms.encryption-algorithm-spec")
                .map(String::as_str)
                .unwrap_or("SYMMETRIC_DEFAULT"),
        )
        .map_err(|_| ComputeError::new_err("Unsupported AWS KMS encryption algorithm"))?;
        py.detach(|| polars_core::runtime::ASYNC.block_on(async {
            let mut loader = aws_config::defaults(BehaviorVersion::latest());
            match (properties.get("aws_access_key_id"), properties.get("aws_secret_access_key")) {
                (Some(access), Some(secret)) => {
                    loader = loader.credentials_provider(Credentials::new(
                        access, secret, properties.get("aws_session_token").cloned(), None, "iceberg-catalog",
                    ));
                },
                (None, None) if !properties.contains_key("aws_session_token") => {},
                _ => return Err(ComputeError::new_err("AWS static credentials require both 'aws_access_key_id' and 'aws_secret_access_key'")),
            }
            if let Some(profile) = properties.get("profile_name") {
                loader = loader.profile_name(profile);
            }
            if let Some(region) = properties.get("region_name") {
                loader = loader.region(Region::new(region.clone()));
            }
            let config = loader.load().await;
            let mut builder = aws_sdk_kms::config::Builder::from(&config);
            if let Some(endpoint) = properties.get("kms.endpoint") {
                builder = builder.endpoint_url(endpoint);
            }
            Ok(Self { client: Client::from_conf(builder.build()), algorithm })
        }))
    }

    fn unwrap_key<'py>(
        &self,
        py: Python<'py>,
        wrapped_key: Vec<u8>,
        wrapping_key_id: String,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let key = py.detach(|| {
            polars_core::runtime::ASYNC.block_on(async {
                let response = self
                    .client
                    .decrypt()
                    .key_id(wrapping_key_id)
                    .encryption_algorithm(self.algorithm.clone())
                    .ciphertext_blob(Blob::new(wrapped_key))
                    .send()
                    .await
                    .map_err(|_| ComputeError::new_err("AWS KMS Decrypt failed"))?;
                response.plaintext.map(Blob::into_inner).ok_or_else(|| {
                    ComputeError::new_err("AWS KMS Decrypt response missing plaintext")
                })
            })
        })?;
        Ok(PyBytes::new(py, &key))
    }
}

#[pyfunction]
pub fn _iceberg_decrypt<'py>(
    py: Python<'py>,
    ciphertext: &[u8],
    key: &[u8],
    aad: &[u8],
    stream: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let plaintext = py
        .detach(|| {
            if stream {
                decrypt_stream(ciphertext, key, aad)
            } else {
                AesGcmDecryptor::new(key)?.decrypt_bytes(ciphertext, aad)
            }
        })
        .map_err(|_| {
            ComputeError::new_err(
                "Iceberg decryption failed: invalid key, authentication data, or ciphertext",
            )
        })?;
    Ok(PyBytes::new(py, &plaintext))
}
