use std::path::PathBuf;
use std::sync::Arc;

use polars::prelude::PartitionVariant;
use polars_utils::IdxSize;
use pyo3::{pyclass, pymethods};

#[pyclass]
#[derive(Clone)]
pub struct PyPartitioning {
    pub path: Arc<PathBuf>,
    pub variant: PartitionVariant,
}

#[pymethods]
impl PyPartitioning {
    #[staticmethod]
    pub fn new_max_size(path: PathBuf, max_size: IdxSize) -> PyPartitioning {
        PyPartitioning {
            path: Arc::new(path),
            variant: PartitionVariant::MaxSize(max_size),
        }
    }
}
