use pyo3::prelude::*;

use crate::PyExpr;
use crate::expr::datatype::PyDataTypeExpr;

#[pymethods]
impl PyExpr {
    fn union_tag(&self) -> Self {
        self.inner.clone().union().tag().into()
    }

    fn union_field(&self, name: &str) -> Self {
        self.inner.clone().union().field(name.into()).into()
    }

    fn union_value(&self, dtype: PyDataTypeExpr, variant: &str) -> Self {
        self.inner
            .clone()
            .union()
            .value(dtype.inner, variant.into())
            .into()
    }
}
