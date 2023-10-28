use polars::prelude::*;
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
        self.inner
            .clone()
            .map(
                move |s| {
                    s.binary()?
                        .hex_decode(strict)
                        .map(|s| Some(s.into_series()))
                },
                GetOutput::same_type(),
            )
            .with_fmt("bin.hex_decode")
            .into()
    }

    #[cfg(feature = "binary_encoding")]
    fn bin_base64_decode(&self, strict: bool) -> Self {
        self.inner
            .clone()
            .map(
                move |s| {
                    s.binary()?
                        .base64_decode(strict)
                        .map(|s| Some(s.into_series()))
                },
                GetOutput::same_type(),
            )
            .with_fmt("bin.base64_decode")
            .into()
    }

    #[cfg(feature = "binary_encoding")]
    fn bin_hex_encode(&self) -> Self {
        self.inner
            .clone()
            .map(
                move |s| s.binary().map(|s| Some(s.hex_encode().into_series())),
                GetOutput::from_type(DataType::Utf8),
            )
            .with_fmt("bin.hex_encode")
            .into()
    }

    #[cfg(feature = "binary_encoding")]
    fn bin_base64_encode(&self) -> Self {
        self.inner
            .clone()
            .map(
                move |s| s.binary().map(|s| Some(s.base64_encode().into_series())),
                GetOutput::from_type(DataType::Utf8),
            )
            .with_fmt("bin.base64_encode")
            .into()
    }
}
