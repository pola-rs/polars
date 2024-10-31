use pyo3::prelude::*;

use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn bitwise_count_ones(&self) -> Self {
        self.inner.clone().bitwise_count_ones().into()
    }

    fn bitwise_count_zeros(&self) -> Self {
        self.inner.clone().bitwise_count_zeros().into()
    }

    fn bitwise_leading_ones(&self) -> Self {
        self.inner.clone().bitwise_leading_ones().into()
    }

    fn bitwise_leading_zeros(&self) -> Self {
        self.inner.clone().bitwise_leading_zeros().into()
    }

    fn bitwise_trailing_ones(&self) -> Self {
        self.inner.clone().bitwise_trailing_ones().into()
    }

    fn bitwise_trailing_zeros(&self) -> Self {
        self.inner.clone().bitwise_trailing_zeros().into()
    }

    fn bitwise_and(&self) -> Self {
        self.inner.clone().bitwise_and().into()
    }

    fn bitwise_or(&self) -> Self {
        self.inner.clone().bitwise_or().into()
    }

    fn bitwise_xor(&self) -> Self {
        self.inner.clone().bitwise_xor().into()
    }
}
