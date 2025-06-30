use polars::prelude::{DataType, Selector};
use polars_plan::dsl;

use crate::prelude::Wrap;

#[pyo3::pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PySelector {
    pub inner: Selector,
}

impl From<Selector> for PySelector {
    fn from(inner: Selector) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "pymethods")]
#[pyo3::pymethods]
impl PySelector {
    fn union(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() | other.inner.clone(),
        }
    }

    fn difference(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() - other.inner.clone(),
        }
    }

    fn exclusive_or(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() ^ other.inner.clone(),
        }
    }

    fn intersect(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() & other.inner.clone(),
        }
    }

    fn exclude_columns(&self, names: Vec<String>) -> Self {
        self.inner.clone().exclude_cols(names).into()
    }

    fn exclude_dtype(&self, dtypes: Vec<Wrap<DataType>>) -> Self {
        let dtypes = dtypes.into_iter().map(|x| x.0).collect::<Vec<_>>();
        self.inner.clone().exclude_dtype(dtypes).into()
    }

    #[staticmethod]
    fn with_datatype(dtypes: Vec<Wrap<DataType>>) -> Self {
        let dtypes = dtypes.into_iter().map(|x| x.0).collect::<Vec<_>>();
        dsl::dtype_cols(dtypes).into()
    }

    #[staticmethod]
    fn with_name(names: Vec<String>) -> Self {
        dsl::cols(names).into()
    }

    #[staticmethod]
    fn at_index(indices: Vec<i64>) -> Self {
        dsl::index_cols(indices).into()
    }

    #[staticmethod]
    fn nth(n: i64) -> Self {
        dsl::nth(n).into()
    }

    #[staticmethod]
    fn first() -> Self {
        dsl::first().into()
    }

    #[staticmethod]
    fn last() -> Self {
        dsl::last().into()
    }

    #[staticmethod]
    fn regex(regex: String) -> Self {
        Self {
            inner: Selector::Regex(regex.into()),
        }
    }

    #[staticmethod]
    fn all() -> Self {
        dsl::all().into()
    }
}
