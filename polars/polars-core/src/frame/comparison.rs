use crate::prelude::*;

macro_rules! impl_compare {
    ($self:expr, $rhf:expr, $method:ident) => {{
        DataFrame::new(
            $self
                .iter()
                .zip($rhf.iter())
                .map(|(lhs, rhs)| lhs.$method(rhs).unwrap())
                .collect(),
        )
    }};
}

impl DataFrame {
    pub fn equal(&self, rhf: &DataFrame) -> PolarsResult<Self> {
        impl_compare!(self, rhf, equal)
    }
    pub fn not_equal(&self, rhf: &DataFrame) -> PolarsResult<Self> {
        impl_compare!(self, rhf, not_equal)
    }
    pub fn gt(&self, rhf: &DataFrame) -> PolarsResult<Self> {
        impl_compare!(self, rhf, gt)
    }
    pub fn lt(&self, rhf: &DataFrame) -> PolarsResult<Self> {
        impl_compare!(self, rhf, lt)
    }
    pub fn gt_eq(&self, rhf: &DataFrame) -> PolarsResult<Self> {
        impl_compare!(self, rhf, gt_eq)
    }
    pub fn lt_eq(&self, rhf: &DataFrame) -> PolarsResult<Self> {
        impl_compare!(self, rhf, lt_eq)
    }
}
