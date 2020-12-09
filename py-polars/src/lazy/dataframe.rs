use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsEr;
use crate::lazy::{dsl::PyExpr, utils::py_exprs_to_exprs};
use polars::lazy::frame::{JoinOptions, LazyFrame, LazyGroupBy};
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
    #[staticmethod]
    pub fn new_from_csv(
        path: String,
        sep: &str,
        has_header: bool,
        ignore_errors: bool,
        skip_rows: usize,
        stop_after_n_rows: Option<usize>,
        cache: bool,
    ) -> Self {
        let delimiter = sep.as_bytes()[0];
        LazyFrame::new_from_csv(
            path,
            delimiter,
            has_header,
            ignore_errors,
            skip_rows,
            stop_after_n_rows,
            cache,
        )
        .into()
    }

    #[staticmethod]
    pub fn new_from_parquet(path: String, stop_after_n_rows: Option<usize>, cache: bool) -> Self {
        LazyFrame::new_from_parquet(path, stop_after_n_rows, cache).into()
    }

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
        simplify_expr: bool,
    ) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let ldf = ldf
            .with_type_coercion(type_coercion)
            .with_predicate_pushdown(predicate_pushdown)
            .with_simplify_expr(simplify_expr)
            .with_projection_pushdown(projection_pushdown);
        ldf.into()
    }

    pub fn sort(&self, by_column: &str, reverse: bool) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.sort(by_column, reverse).into()
    }
    pub fn cache(&self) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.cache().into()
    }

    pub fn collect(&self) -> PyResult<PyDataFrame> {
        let ldf = self.ldf.clone();
        let gil = Python::acquire_gil();
        let py = gil.python();
        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        let df = py.allow_threads(|| ldf.collect().map_err(PyPolarsEr::from))?;
        Ok(df.into())
    }

    pub fn fetch(&self, n_rows: usize) -> PyResult<PyDataFrame> {
        let ldf = self.ldf.clone();
        let df = ldf.fetch(n_rows).map_err(PyPolarsEr::from)?;
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
        allow_parallel: bool,
        force_parallel: bool,
    ) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let other = other.ldf;
        let options = JoinOptions {
            allow_parallel,
            force_parallel,
        };
        ldf.inner_join(other, left_on.inner, right_on.inner, Some(options))
            .into()
    }

    pub fn outer_join(
        &mut self,
        other: PyLazyFrame,
        left_on: PyExpr,
        right_on: PyExpr,
        allow_parallel: bool,
        force_parallel: bool,
    ) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let other = other.ldf;
        let options = JoinOptions {
            allow_parallel,
            force_parallel,
        };
        ldf.outer_join(other, left_on.inner, right_on.inner, Some(options))
            .into()
    }

    pub fn left_join(
        &mut self,
        other: PyLazyFrame,
        left_on: PyExpr,
        right_on: PyExpr,
        allow_parallel: bool,
        force_parallel: bool,
    ) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let other = other.ldf;
        let options = JoinOptions {
            allow_parallel,
            force_parallel,
        };
        ldf.left_join(other, left_on.inner, right_on.inner, Some(options))
            .into()
    }

    pub fn with_column(&mut self, expr: PyExpr) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.with_column(expr.inner).into()
    }

    pub fn with_columns(&mut self, exprs: Vec<PyExpr>) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.with_columns(py_exprs_to_exprs(exprs)).into()
    }

    pub fn with_column_renamed(&mut self, existing: &str, new: &str) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.with_column_renamed(existing, new).into()
    }

    pub fn reverse(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.reverse().into()
    }

    pub fn shift(&self, periods: i32) -> Self {
        let ldf = self.ldf.clone();
        ldf.shift(periods).into()
    }

    pub fn fill_none(&self, fill_value: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.fill_none(fill_value.inner).into()
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

    pub fn std(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.std().into()
    }

    pub fn var(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.var().into()
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

    pub fn drop_duplicates(&self, maintain_order: bool, subset: Option<Vec<String>>) -> Self {
        let ldf = self.ldf.clone();
        ldf.drop_duplicates(maintain_order, subset).into()
    }

    pub fn drop_nulls(&self, subset: Option<Vec<String>>) -> Self {
        let ldf = self.ldf.clone();
        ldf.drop_nulls(subset.as_ref().map(|v| v.as_ref())).into()
    }

    pub fn slice(&self, offset: usize, len: usize) -> Self {
        let ldf = self.ldf.clone();
        ldf.slice(offset, len).into()
    }

    pub fn clone(&self) -> PyLazyFrame {
        self.ldf.clone().into()
    }
}
