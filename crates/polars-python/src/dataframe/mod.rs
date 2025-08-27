#[cfg(feature = "pymethods")]
mod construction;
#[cfg(feature = "pymethods")]
mod export;
#[cfg(feature = "pymethods")]
mod general;
#[cfg(feature = "pymethods")]
mod io;
#[cfg(feature = "pymethods")]
mod serde;

use parking_lot::RwLock;
use polars::prelude::DataFrame;
use pyo3::pyclass;

#[pyclass(frozen)]
#[repr(transparent)]
pub struct PyDataFrame {
    pub df: RwLock<DataFrame>,
}

impl Clone for PyDataFrame {
    fn clone(&self) -> Self {
        PyDataFrame {
            df: RwLock::new(self.df.read().clone()),
        }
    }
}

impl From<DataFrame> for PyDataFrame {
    fn from(df: DataFrame) -> Self {
        Self::new(df)
    }
}

impl From<PyDataFrame> for DataFrame {
    fn from(pdf: PyDataFrame) -> Self {
        pdf.df.into_inner()
    }
}

impl PyDataFrame {
    pub(crate) fn new(df: DataFrame) -> Self {
        PyDataFrame {
            df: RwLock::new(df),
        }
    }
}
