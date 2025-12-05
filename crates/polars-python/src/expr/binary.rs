use pyo3::prelude::*;

use super::datatype::PyDataTypeExpr;
use crate::PyExpr;

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
    fn bin_reinterpret(&self, dtype: PyDataTypeExpr, kind: &str) -> PyResult<Self> {
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
            .reinterpret(dtype.inner, is_little_endian)
            .into())
    }

    fn bin_size_bytes(&self) -> Self {
        self.inner.clone().binary().size_bytes().into()
    }

    fn bin_slice(&self, offset: PyExpr, length: PyExpr) -> Self {
        self.inner
            .clone()
            .binary()
            .slice(offset.inner, length.inner)
            .into()
    }

    fn bin_head(&self, n: PyExpr) -> Self {
        self.inner.clone().binary().head(n.inner).into()
    }

    fn bin_tail(&self, n: PyExpr) -> Self {
        self.inner.clone().binary().tail(n.inner).into()
    }
}
