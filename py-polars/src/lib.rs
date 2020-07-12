use pyo3::prelude::*;

pub mod dataframe;
pub mod error;
pub mod series;

use crate::{dataframe::PyDataFrame, series::PySeries};

#[pymodule]
fn polars(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySeries>()?;
    m.add_class::<PyDataFrame>()?;
    Ok(())
}
