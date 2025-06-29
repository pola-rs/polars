use polars::prelude::{DataType, PlSmallStr, Selector};

#[pyo3::pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PySelector {
    inner: Selector,
}

#[cfg(feature = "pymethods")]
#[pyo3::pymethods]
impl PySelector {
    fn union(&self, other: &Self) -> Self {
        Self { inner: self.inner.clone() | other.inner.clone() }
    }

    fn difference(&self, other: &Self) -> Self {
        Self { inner: self.inner.clone() - other.inner.clone() }
    }

    fn exclusive_or(&self, other: &Self) -> Self {
        Self { inner: self.inner.clone() ^ other.inner.clone() }
    }

    fn intersect(&self, other: &Self) -> Self {
        Self { inner: self.inner.clone() & other.inner.clone() }
    }

    fn exclude_columns(&self, names: Vec<String>) -> Self {
        Self { inner: self.inner.clone().exclude_columns(names) }
    }

    #[staticmethod]
    fn with_datatype(dtypes: Vec<DataType>) -> Self {
        Self { inner: Selector::WithDataType(dtypes.into()) }
    }

    #[staticmethod]
    fn with_name(names: Vec<String>) -> Self {
        Self { inner: Selector::WithName(names.into_iter().map(PlSmallStr::from).collect()) }
    }

    #[staticmethod]
    fn at_index(indices: Vec<i64>) -> Self {
        Self { inner: Selector::AtIndex(indices.into()) }
    }

    #[staticmethod]
    fn regex(regex: String) -> Self {
        Self { inner: Selector::Regex(regex.into()) }
    }

    #[staticmethod]
    fn wildcard() -> Self {
        Self { inner: Selector::Wildcard }
    }
}
