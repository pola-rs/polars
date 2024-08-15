#[cfg(feature = "pymethods")]
mod aggregation;
#[cfg(feature = "pymethods")]
mod arithmetic;
#[cfg(feature = "pymethods")]
mod buffers;
#[cfg(feature = "pymethods")]
mod c_interface;
#[cfg(feature = "pymethods")]
mod comparison;
#[cfg(feature = "pymethods")]
mod construction;
#[cfg(feature = "pymethods")]
mod export;
#[cfg(feature = "pymethods")]
mod general;
#[cfg(feature = "pymethods")]
mod import;
#[cfg(feature = "pymethods")]
mod numpy_ufunc;
#[cfg(feature = "pymethods")]
mod scatter;

use polars::prelude::Series;
use pyo3::pyclass;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PySeries {
    pub series: Series,
}

impl From<Series> for PySeries {
    fn from(series: Series) -> Self {
        PySeries { series }
    }
}

impl PySeries {
    pub(crate) fn new(series: Series) -> Self {
        PySeries { series }
    }
}

pub(crate) trait ToSeries {
    fn to_series(self) -> Vec<Series>;
}

impl ToSeries for Vec<PySeries> {
    fn to_series(self) -> Vec<Series> {
        // SAFETY: repr is transparent.
        unsafe { std::mem::transmute(self) }
    }
}

pub(crate) trait ToPySeries {
    fn to_pyseries(self) -> Vec<PySeries>;
}

impl ToPySeries for Vec<Series> {
    fn to_pyseries(self) -> Vec<PySeries> {
        // SAFETY: repr is transparent.
        unsafe { std::mem::transmute(self) }
    }
}
