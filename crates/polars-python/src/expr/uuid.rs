use pyo3::prelude::*;

use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn uuid_generate_v4(&self) -> Self {
        self.inner.clone().uuid().generate_v4().into()
    }

    fn uuid_generate_v7(&self) -> Self {
        self.inner.clone().uuid().generate_v7().into()
    }

    fn uuid_version(&self) -> Self {
        self.inner.clone().uuid().version().into()
    }

    fn uuid_timestamp(&self, strict: bool) -> Self {
        self.inner.clone().uuid().timestamp(strict).into()
    }
}
