use polars::prelude::*;
use pyo3::{exceptions::RuntimeError, prelude::*};

use crate::{
    error::PyPolarsEr,
    series::{to_pyseries_collection, to_series_collection, PySeries},
};

#[pyclass]
#[repr(transparent)]
pub struct PyDataFrame {
    pub df: DataFrame,
}

impl PyDataFrame {
    fn new(df: DataFrame) -> Self {
        PyDataFrame { df }
    }
}

#[pymethods]
impl PyDataFrame {
    #[new]
    pub fn __init__(columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = to_series_collection(columns);
        let df = DataFrame::new(columns).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn as_str(&self) -> String {
        format!("{:?}", self.df)
    }

    pub fn inner_join(&self, other: &PyDataFrame, left_on: &str, right_on: &str) -> PyResult<Self> {
        let df = self
            .df
            .inner_join(&other.df, left_on, right_on)
            .map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn left_join(&self, other: &PyDataFrame, left_on: &str, right_on: &str) -> PyResult<Self> {
        let df = self
            .df
            .left_join(&other.df, left_on, right_on)
            .map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn outer_join(&self, other: &PyDataFrame, left_on: &str, right_on: &str) -> PyResult<Self> {
        let df = self
            .df
            .outer_join(&other.df, left_on, right_on)
            .map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn get_columns(&self) -> Vec<PySeries> {
        let cols = self.df.get_columns().clone();
        to_pyseries_collection(cols)
    }

    pub fn columns(&self) -> Vec<&str> {
        self.df.columns()
    }

    pub fn n_chunks(&self) -> PyResult<usize> {
        let n = self.df.n_chunks().map_err(PyPolarsEr::from)?;
        Ok(n)
    }

    pub fn shape(&self) -> (usize, usize) {
        self.df.shape()
    }

    pub fn height(&self) -> usize {
        self.df.height()
    }

    pub fn width(&self) -> usize {
        self.df.width()
    }

    pub fn hstack(&mut self, columns: Vec<PySeries>) -> PyResult<()> {
        let columns = to_series_collection(columns);
        self.df.hstack(&columns).map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn drop(&mut self, name: &str) -> PyResult<PySeries> {
        let s = self.df.drop(name).map_err(PyPolarsEr::from)?;
        Ok(PySeries { series: s })
    }

    pub fn drop_pure(&self, name: &str) -> PyResult<Self> {
        let df = self.df.drop_pure(name).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn select_idx(&self, idx: usize) -> Option<PySeries> {
        self.df.select_idx(idx).map(|s| PySeries::new(s.clone()))
    }

    pub fn find_idx_by_bane(&self, name: &str) -> Option<usize> {
        self.df.find_idx_by_name(name)
    }

    pub fn column(&self, name: &str) -> Option<PySeries> {
        self.df.column(name).map(|s| PySeries::new(s.clone()))
    }

    pub fn select(&self, selection: Vec<String>) -> PyResult<Self> {
        let df = self.df.select(&selection).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn filter(&self, mask: &PySeries) -> PyResult<Self> {
        let filter_series = &mask.series;
        if let Series::Bool(ca) = filter_series {
            let df = self.df.filter(ca).map_err(PyPolarsEr::from)?;
            Ok(PyDataFrame::new(df))
        } else {
            Err(RuntimeError::py_err("Expected a boolean mask"))
        }
    }

    pub fn take(&self, indices: Vec<usize>) -> PyResult<Self> {
        let df = self.df.take(&indices).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn take_with_series(&self, indices: &PySeries) -> PyResult<Self> {
        let idx = indices.series.u32().map_err(PyPolarsEr::from)?;
        let df = self.df.take(&idx).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn sort(&self, by_column: &str) -> PyResult<Self> {
        let df = self.df.sort(by_column).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn sort_in_place(&mut self, by_column: &str) -> PyResult<()> {
        self.df.sort_in_place(by_column).map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn replace(&mut self, column: &str, new_col: PySeries) -> PyResult<()> {
        self.df
            .replace(column, new_col.series)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn slice(&self, offset: usize, length: usize) -> PyResult<Self> {
        let df = self.df.slice(offset, length).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn head(&self, length: Option<usize>) -> Self {
        let df = self.df.head(length);
        PyDataFrame::new(df)
    }

    pub fn tail(&self, length: Option<usize>) -> Self {
        let df = self.df.tail(length);
        PyDataFrame::new(df)
    }

    pub fn frame_equal(&self, other: &PyDataFrame) -> bool {
        self.df.frame_equal(&other.df)
    }

    pub fn groupby(&self, by: &str, select: &str, agg: &str) -> PyResult<Self> {
        let gb = self.df.groupby(by).map_err(PyPolarsEr::from)?;
        let selection = gb.select(select);
        let df = match agg {
            "min" => selection.min(),
            "max" => selection.max(),
            "mean" => selection.mean(),
            "sum" => selection.sum(),
            "count" => selection.count(),
            a => Err(PolarsError::Other(format!("agg fn {} does not exists", a))),
        };
        let df = df.map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }
}
