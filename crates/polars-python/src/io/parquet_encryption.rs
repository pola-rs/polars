use std::sync::Arc;

use polars::prelude::PolarsError;
use polars_parquet::parquet::encryption::decrypt::FileDecryptionProperties;
use pyo3::prelude::*;

use crate::error::PyPolarsErr;

#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
pub struct PyFileDecryptionProperties {
    pub inner: Arc<FileDecryptionProperties>,
}

#[pymethods]
impl PyFileDecryptionProperties {
    #[new]
    #[pyo3(signature = (footer_key, column_keys, aad_prefix, verify_footer_signature))]
    fn new(
        footer_key: Vec<u8>,
        column_keys: Vec<(String, Vec<u8>)>,
        aad_prefix: Option<Vec<u8>>,
        verify_footer_signature: bool,
    ) -> PyResult<Self> {
        let mut builder = FileDecryptionProperties::builder(footer_key);
        for (column_name, key) in column_keys {
            builder = builder.with_column_key(&column_name, key);
        }
        if let Some(aad_prefix) = aad_prefix {
            builder = builder.with_aad_prefix(aad_prefix);
        }
        if !verify_footer_signature {
            builder = builder.disable_footer_signature_verification();
        }
        let inner = builder
            .build()
            .map_err(|e| PyPolarsErr::from(PolarsError::from(e)))?;
        Ok(Self { inner })
    }
}
