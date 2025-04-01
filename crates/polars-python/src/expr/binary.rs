use polars::prelude::DataType;
use pyo3::prelude::*;

use crate::PyExpr;
use crate::prelude::Wrap;

#[pymethods]
impl PyExpr {
    fn bin_contains(&self, lit: PyExpr) -> Self {
        self.inner
            .clone()
            .binary()
            .contains_literal(lit.inner)
            .into()
    }

    fn bin_ends_with(&self, sub: PyExpr) -> Self {
        self.inner.clone().binary().ends_with(sub.inner).into()
    }

    fn bin_starts_with(&self, sub: PyExpr) -> Self {
        self.inner.clone().binary().starts_with(sub.inner).into()
    }

    #[cfg(feature = "binary_encoding")]
    fn bin_hex_decode(&self, strict: bool) -> Self {
        self.inner.clone().binary().hex_decode(strict).into()
    }

    #[cfg(feature = "binary_encoding")]
    fn bin_base64_decode(&self, strict: bool) -> Self {
        self.inner.clone().binary().base64_decode(strict).into()
    }

    #[cfg(feature = "binary_encoding")]
    fn bin_hex_encode(&self) -> Self {
        self.inner.clone().binary().hex_encode().into()
    }

    #[cfg(feature = "binary_encoding")]
    fn bin_base64_encode(&self) -> Self {
        self.inner.clone().binary().base64_encode().into()
    }

    #[cfg(feature = "binary_encoding")]
    #[allow(clippy::wrong_self_convention)]
    fn from_buffer(&self, dtype: Wrap<DataType>, kind: &str) -> PyResult<Self> {
        use pyo3::exceptions::PyValueError;

        let is_little_endian = match kind.to_lowercase().as_str() {
            "little" => true,
            "big" => false,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid endianness: {kind}. Valid values are \"little\" or \"big\"."
                )));
            },
        };
        Ok(self
            .inner
            .clone()
            .binary()
            .from_buffer(dtype.0, is_little_endian)
            .into())
    }

    fn bin_size_bytes(&self) -> Self {
        self.inner.clone().binary().size_bytes().into()
    }
}
