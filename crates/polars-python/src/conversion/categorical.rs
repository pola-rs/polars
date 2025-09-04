use std::sync::Arc;

use polars_dtype::categorical::{CatSize, Categories};
use pyo3::{pyclass, pymethods};

#[pyclass(frozen)]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyCategories {
    categories: Arc<Categories>,
}

impl PyCategories {
    pub fn categories(&self) -> &Arc<Categories> {
        &self.categories
    }
}

#[pymethods]
impl PyCategories {
    #[new]
    pub fn __init__(name: String, namespace: String, physical: String) -> Self {
        Self {
            categories: Categories::new(name.into(), namespace.into(), physical.parse().unwrap()),
        }
    }

    fn __getnewargs__(&self) -> (String, String, String) {
        (
            self.categories.name().to_string(),
            self.categories.namespace().to_string(),
            self.categories.physical().as_str().to_owned(),
        )
    }

    #[staticmethod]
    pub fn global_categories() -> Self {
        Self {
            categories: Categories::global(),
        }
    }

    #[staticmethod]
    pub fn random(namespace: String, physical: String) -> Self {
        Self {
            categories: Categories::random(namespace.into(), physical.parse().unwrap()),
        }
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.categories, &other.categories)
    }

    pub fn __hash__(&self) -> u64 {
        self.categories.hash()
    }

    pub fn name(&self) -> &str {
        self.categories.name()
    }

    pub fn namespace(&self) -> &str {
        self.categories.namespace()
    }

    pub fn physical(&self) -> &str {
        self.categories.physical().as_str()
    }

    pub fn get_cat(&self, s: &str) -> Option<CatSize> {
        self.categories.mapping().get_cat(s)
    }

    pub fn cat_to_str(&self, cat: CatSize) -> Option<String> {
        Some(self.categories.mapping().cat_to_str(cat)?.to_owned())
    }

    pub fn is_global(&self) -> bool {
        self.categories.is_global()
    }
}

impl From<Arc<Categories>> for PyCategories {
    fn from(categories: Arc<Categories>) -> Self {
        Self { categories }
    }
}
