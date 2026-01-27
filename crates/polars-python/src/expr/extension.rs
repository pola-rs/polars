use pyo3::prelude::*;

use crate::PyExpr;
use crate::expr::datatype::PyDataTypeExpr;

#[pymethods]
impl PyExpr {
    fn ext_storage(&self) -> Self {
        self.inner.clone().ext().storage().into()
    }

    fn ext_to(&self, dtype: PyDataTypeExpr) -> Self {
        self.inner.clone().ext().to(dtype.inner).into()
    }
}
