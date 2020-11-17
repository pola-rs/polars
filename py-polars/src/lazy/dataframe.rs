use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsEr;
use crate::lazy::{dsl::PyExpr, utils::py_exprs_to_exprs};
use polars::lazy::frame::{LazyFrame, LazyGroupBy};
use pyo3::prelude::*;

#[pyclass]
#[repr(transparent)]
pub struct PyLazyGroupBy {
    // option because we cannot get a self by value in pyo3
    pub lgb: Option<LazyGroupBy>,
}

#[pymethods]
impl PyLazyGroupBy {
    pub fn agg(&mut self, aggs: Vec<PyExpr>) -> PyLazyFrame {
        let lgb = self.lgb.take().unwrap();
        let aggs = py_exprs_to_exprs(aggs);
        lgb.agg(aggs).into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyLazyFrame {
    // option because we cannot get a self by value in pyo3
    pub ldf: LazyFrame,
}

impl From<LazyFrame> for PyLazyFrame {
    fn from(ldf: LazyFrame) -> Self {
        PyLazyFrame { ldf }
    }
}

#[pymethods]
impl PyLazyFrame {
    pub fn describe_plan(&self) -> String {
        self.ldf.describe_plan()
    }

    pub fn describe_optimized_plan(&self) -> PyResult<String> {
        let result = self
            .ldf
            .describe_optimized_plan()
            .map_err(PyPolarsEr::from)?;
        Ok(result)
    }

    pub fn optimization_toggle(
        &self,
        type_coercion: bool,
        predicate_pushdown: bool,
        projection_pushdown: bool,
    ) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let ldf = ldf
            .with_type_coercion_optimization(type_coercion)
            .with_predicate_pushdown_optimization(predicate_pushdown)
            .with_projection_pushdown_optimization(projection_pushdown);
        ldf.into()
    }

    pub fn sort(&self, by_column: &str, reverse: bool) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.sort(by_column, reverse).into()
    }

    pub fn collect(&self) -> PyResult<PyDataFrame> {
        let ldf = self.ldf.clone();
        let df = ldf.collect().map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn filter(&mut self, predicate: PyExpr) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.filter(predicate.inner).into()
    }

    pub fn select(&mut self, exprs: Vec<PyExpr>) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let exprs = py_exprs_to_exprs(exprs);
        ldf.select(exprs).into()
    }

    pub fn groupby(&mut self, by: Vec<&str>) -> PyLazyGroupBy {
        let ldf = self.ldf.clone();
        let lazy_gb = ldf.groupby(by);

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    pub fn inner_join(
        &mut self,
        other: PyLazyFrame,
        left_on: PyExpr,
        right_on: PyExpr,
    ) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let other = other.ldf;
        ldf.inner_join(other, left_on.inner, right_on.inner).into()
    }

    pub fn outer_join(
        &mut self,
        other: PyLazyFrame,
        left_on: PyExpr,
        right_on: PyExpr,
    ) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let other = other.ldf;
        ldf.outer_join(other, left_on.inner, right_on.inner).into()
    }

    pub fn left_join(
        &mut self,
        other: PyLazyFrame,
        left_on: PyExpr,
        right_on: PyExpr,
    ) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let other = other.ldf;
        ldf.left_join(other, left_on.inner, right_on.inner).into()
    }

    pub fn with_column(&mut self, expr: PyExpr) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.with_column(expr.inner).into()
    }

    pub fn with_columns(&mut self, exprs: Vec<PyExpr>) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.with_columns(py_exprs_to_exprs(exprs)).into()
    }

    pub fn reverse(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.reverse().into()
    }

    pub fn shift(&self, periods: i32) -> Self {
        let ldf = self.ldf.clone();
        ldf.shift(periods).into()
    }

    pub fn min(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.min().into()
    }

    pub fn max(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.max().into()
    }

    pub fn sum(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.sum().into()
    }

    pub fn mean(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.mean().into()
    }

    pub fn median(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.median().into()
    }

    pub fn quantile(&self, quantile: f64) -> Self {
        let ldf = self.ldf.clone();
        ldf.quantile(quantile).into()
    }

    pub fn explode(&self, column: &str) -> Self {
        let ldf = self.ldf.clone();
        ldf.explode(column).into()
    }

    pub fn clone(&self) -> PyLazyFrame {
        self.ldf.clone().into()
    }
}
