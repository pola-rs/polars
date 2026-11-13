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

    #[pyo3(signature = (arms, default))]
    fn union_match(&self, arms: Vec<(String, PyExpr)>, default: Option<PyExpr>) -> Self {
        let arms = arms
            .into_iter()
            .map(|(name, arm)| (name.into(), arm.inner))
            .collect();
        self.inner
            .clone()
            .union()
            .match_(arms, default.map(|d| d.inner))
            .into()
    }
}
