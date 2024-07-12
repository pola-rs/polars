use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

#[pyclass(frozen)]
pub struct PartitionedWriteOptions {
    pub path: PyBackedStr,
    pub partition_by: Vec<PyBackedStr>,
    pub chunk_size_bytes: usize,
}

#[pymethods]
impl PartitionedWriteOptions {
    #[new]
    pub fn __init__(
        path: PyBackedStr,
        partition_by: Vec<PyBackedStr>,
        chunk_size_bytes: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            path,
            partition_by,
            chunk_size_bytes,
        })
    }
}
