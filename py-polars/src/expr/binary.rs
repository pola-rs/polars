use pyo3::prelude::*;

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

    fn bin_size_bytes(&self) -> Self {
        self.inner.clone().binary().size_bytes().into()
    }
}
