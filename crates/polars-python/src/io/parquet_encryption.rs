use std::sync::Arc;

use polars_parquet::parquet::encryption::ParquetEncryptionAlgorithm;
use polars_parquet::parquet::encryption::decrypt::{FileDecryptionProperties, KeyRetriever};
use polars_parquet::parquet::encryption::encrypt::FileEncryptionProperties;
use polars_parquet::parquet::error::{ParquetError, ParquetResult};
use polars_utils::aliases::{InitHashMaps, PlHashMap, PlHashSet};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBytes, PyDict, PyDictMethods};

struct PythonKeyRetriever {
    retriever: Py<PyAny>,
}

impl KeyRetriever for PythonKeyRetriever {
    fn retrieve_key(&self, key_metadata: &[u8]) -> ParquetResult<Vec<u8>> {
        Python::attach(|py| {
            self.retriever
                .call1(py, (PyBytes::new(py, key_metadata),))
                .and_then(|value| value.extract::<Vec<u8>>(py))
        })
        .map_err(|err| {
            ParquetError::InvalidParameter(format!("Python key retriever failed: {err}"))
        })
    }
}

fn validate_keys(dict: &Bound<'_, PyDict>, allowed: &[&str], parameter_name: &str) -> PyResult<()> {
    let allowed = allowed.iter().copied().collect::<PlHashSet<_>>();
    for key in dict.keys() {
        let key = key.extract::<PyBackedStr>()?;
        if !allowed.contains(&*key) {
            return Err(PyValueError::new_err(format!(
                "unknown key for `{parameter_name}`: {key}"
            )));
        }
    }
    Ok(())
}

fn required_item<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    parameter_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?.ok_or_else(|| {
        PyValueError::new_err(format!("`{parameter_name}` requires a `{key}` entry"))
    })
}

fn extract_key_bytes(value: &Bound<'_, PyAny>, key_name: &str) -> PyResult<Vec<u8>> {
    value
        .extract::<Vec<u8>>()
        .map_err(|_| PyTypeError::new_err(format!("`{key_name}` must be a bytes-like object")))
}

fn extract_optional_bytes(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<u8>>> {
    dict.get_item(key)?
        .map(|value| extract_key_bytes(&value, key))
        .transpose()
}

fn extract_column_bytes_map(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<PlHashMap<String, Vec<u8>>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(PlHashMap::new());
    };

    let value = value.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err(format!(
            "`{key}` must be a dict mapping column names to bytes"
        ))
    })?;
    let mut out = PlHashMap::with_capacity(value.len());
    for (column_name, key_bytes) in value.iter() {
        let column_name = column_name.extract::<PyBackedStr>()?.to_string();
        let key_bytes = extract_key_bytes(&key_bytes, key)?;
        out.insert(column_name, key_bytes);
    }
    Ok(out)
}

pub(crate) fn parse_file_encryption_properties(
    value: Option<Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<FileEncryptionProperties>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }

    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("`encryption_properties` must be a dict or None"))?;
    validate_keys(
        dict,
        &[
            "footer_key",
            "footer_key_metadata",
            "column_keys",
            "column_key_metadata",
            "plaintext_footer",
            "aad_prefix",
            "store_aad_prefix",
            "encryption_algorithm",
        ],
        "encryption_properties",
    )?;

    let footer_key = extract_key_bytes(
        &required_item(dict, "footer_key", "encryption_properties")?,
        "footer_key",
    )?;
    let mut builder = FileEncryptionProperties::builder(footer_key);

    if let Some(algorithm) = dict.get_item("encryption_algorithm")? {
        let algorithm = match &*algorithm.extract::<PyBackedStr>()? {
            "AES_GCM_V1" => ParquetEncryptionAlgorithm::AesGcmV1,
            "AES_GCM_CTR_V1" => ParquetEncryptionAlgorithm::AesGcmCtrV1,
            algorithm => {
                return Err(PyValueError::new_err(format!(
                    "unsupported Parquet encryption algorithm: {algorithm}"
                )));
            },
        };
        builder = builder.with_algorithm(algorithm);
    }

    if let Some(plaintext_footer) = dict.get_item("plaintext_footer")? {
        builder = builder.with_plaintext_footer(plaintext_footer.extract::<bool>()?);
    }
    if let Some(footer_key_metadata) = extract_optional_bytes(dict, "footer_key_metadata")? {
        builder = builder.with_footer_key_metadata(footer_key_metadata);
    }
    if let Some(aad_prefix) = extract_optional_bytes(dict, "aad_prefix")? {
        builder = builder.with_aad_prefix(aad_prefix);
    }
    if let Some(store_aad_prefix) = dict.get_item("store_aad_prefix")? {
        builder = builder.with_aad_prefix_storage(store_aad_prefix.extract::<bool>()?);
    }

    let column_keys = extract_column_bytes_map(dict, "column_keys")?;
    let mut column_key_metadata = extract_column_bytes_map(dict, "column_key_metadata")?;
    for (column_name, column_key) in column_keys {
        if let Some(metadata) = column_key_metadata.remove(&column_name) {
            builder = builder.with_column_key_and_metadata(&column_name, column_key, metadata);
        } else {
            builder = builder.with_column_key(&column_name, column_key);
        }
    }

    if !column_key_metadata.is_empty() {
        let mut missing = column_key_metadata.into_keys().collect::<Vec<_>>();
        missing.sort();
        return Err(PyValueError::new_err(format!(
            "`column_key_metadata` contains columns without `column_keys`: {}",
            missing.join(", ")
        )));
    }

    builder
        .build()
        .map(Some)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

pub(crate) fn parse_file_decryption_properties(
    value: Option<Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<FileDecryptionProperties>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }

    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("`decryption_properties` must be a dict or None"))?;
    validate_keys(
        dict,
        &[
            "footer_key",
            "key_retriever",
            "column_keys",
            "aad_prefix",
            "check_plaintext_footer_integrity",
        ],
        "decryption_properties",
    )?;

    let aad_prefix = extract_optional_bytes(dict, "aad_prefix")?;
    let check_integrity = dict
        .get_item("check_plaintext_footer_integrity")?
        .map(|value| value.extract::<bool>())
        .transpose()?
        .unwrap_or(true);

    if let Some(retriever) = dict.get_item("key_retriever")? {
        if dict.contains("footer_key")? || dict.contains("column_keys")? {
            return Err(PyValueError::new_err(
                "`key_retriever` cannot be combined with explicit decryption keys",
            ));
        }
        let mut builder =
            FileDecryptionProperties::with_key_retriever(Arc::new(PythonKeyRetriever {
                retriever: retriever.unbind(),
            }));
        if let Some(aad_prefix) = aad_prefix {
            builder = builder.with_aad_prefix(aad_prefix);
        }
        if !check_integrity {
            builder = builder.disable_footer_signature_verification();
        }
        return builder
            .build()
            .map(Some)
            .map_err(|err| PyValueError::new_err(err.to_string()));
    }

    let footer_key = extract_key_bytes(
        &required_item(dict, "footer_key", "decryption_properties")?,
        "footer_key",
    )?;
    let mut builder = FileDecryptionProperties::builder(footer_key);
    for (column_name, column_key) in extract_column_bytes_map(dict, "column_keys")? {
        builder = builder.with_column_key(&column_name, column_key);
    }
    if let Some(aad_prefix) = aad_prefix {
        builder = builder.with_aad_prefix(aad_prefix);
    }
    if !check_integrity {
        builder = builder.disable_footer_signature_verification();
    }
    builder
        .build()
        .map(Some)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}
