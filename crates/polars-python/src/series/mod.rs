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
mod map;
#[cfg(feature = "pymethods")]
mod numpy_ufunc;
#[cfg(feature = "pymethods")]
mod scatter;
pub(crate) use import::import_schema_pycapsule;
use parking_lot::RwLock;
use polars::prelude::{Column, Series};
use pyo3::pyclass;

#[pyclass(frozen)]
#[repr(transparent)]
pub struct PySeries {
    pub series: RwLock<Series>,
}

impl Clone for PySeries {
    fn clone(&self) -> Self {
        Self {
            series: RwLock::new(self.series.read().clone()),
        }
    }
}

impl From<Series> for PySeries {
    fn from(series: Series) -> Self {
        Self::new(series)
    }
}

impl PySeries {
    pub(crate) fn new(series: Series) -> Self {
        PySeries {
            series: RwLock::new(series),
        }
    }
}

pub(crate) trait ToSeries {
    fn to_series(self) -> Vec<Series>;
}

impl ToSeries for Vec<PySeries> {
    fn to_series(self) -> Vec<Series> {
        self.into_iter().map(|s| s.series.into_inner()).collect()
    }
}

pub(crate) trait ToPySeries {
    fn to_pyseries(self) -> Vec<PySeries>;
}

impl ToPySeries for Vec<Series> {
    fn to_pyseries(self) -> Vec<PySeries> {
        self.into_iter().map(PySeries::from).collect()
    }
}

impl ToPySeries for Vec<Column> {
    fn to_pyseries(self) -> Vec<PySeries> {
        // @scalar-opt
        let series: Vec<Series> = self
            .into_iter()
            .map(|c| c.take_materialized_series())
            .collect();
        series.to_pyseries()
    }
}
