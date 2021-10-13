use crate::conversion::Wrap;
use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsEr;
use crate::lazy::{dsl::PyExpr, utils::py_exprs_to_exprs};
use crate::prelude::NullValues;
use crate::utils::str_to_polarstype;
use polars::lazy::frame::{AllowedOptimizations, LazyCsvReader, LazyFrame, LazyGroupBy};
use polars::lazy::prelude::col;
use polars::prelude::{DataFrame, Field, JoinType, Schema};
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

    pub fn head(&mut self, n: usize) -> PyLazyFrame {
        let lgb = self.lgb.take().unwrap();
        lgb.head(Some(n)).into()
    }

    pub fn tail(&mut self, n: usize) -> PyLazyFrame {
        let lgb = self.lgb.take().unwrap();
        lgb.tail(Some(n)).into()
    }

    pub fn apply(&mut self, lambda: PyObject) -> PyLazyFrame {
        let lgb = self.lgb.take().unwrap();

        let function = move |df: DataFrame| {
            let gil = Python::acquire_gil();
            let py = gil.python();
            // get the pypolars module
            let pypolars = PyModule::import(py, "polars").unwrap();

            // create a PyDataFrame struct/object for Python
            let pydf = PyDataFrame::new(df);

            // Wrap this PySeries object in the python side DataFrame wrapper
            let python_df_wrapper = pypolars.getattr("wrap_df").unwrap().call1((pydf,)).unwrap();

            // call the lambda and get a python side DataFrame wrapper
            let result_df_wrapper = match lambda.call1(py, (python_df_wrapper,)) {
                Ok(pyobj) => pyobj,
                Err(e) => panic!("UDF failed: {}", e.pvalue(py).to_string()),
            };
            // unpack the wrapper in a PyDataFrame
            let py_pydf = result_df_wrapper.getattr(py, "_df").expect(
                "Could net get DataFrame attribute '_df'. Make sure that you return a DataFrame object.",
            );
            // Downcast to Rust
            let pydf = py_pydf.extract::<PyDataFrame>(py).unwrap();
            // Finally get the actual DataFrame
            Ok(pydf.df)
        };
        lgb.apply(function).into()
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
#[allow(clippy::should_implement_trait)]
impl PyLazyFrame {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn new_from_csv(
        path: String,
        sep: &str,
        has_header: bool,
        ignore_errors: bool,
        skip_rows: usize,
        stop_after_n_rows: Option<usize>,
        cache: bool,
        overwrite_dtype: Option<Vec<(&str, &PyAny)>>,
        low_memory: bool,
        comment_char: Option<&str>,
        quote_char: Option<&str>,
        null_values: Option<Wrap<NullValues>>,
    ) -> Self {
        let null_values = null_values.map(|w| w.0);
        let comment_char = comment_char.map(|s| s.as_bytes()[0]);
        let quote_char = quote_char.map(|s| s.as_bytes()[0]);
        let delimiter = sep.as_bytes()[0];

        let overwrite_dtype = overwrite_dtype.map(|overwrite_dtype| {
            let fields = overwrite_dtype
                .iter()
                .map(|(name, dtype)| {
                    let str_repr = dtype.str().unwrap().to_str().unwrap();
                    let dtype = str_to_polarstype(str_repr);
                    Field::new(name, dtype)
                })
                .collect();
            Schema::new(fields)
        });

        LazyCsvReader::new(path)
            .with_delimiter(delimiter)
            .has_header(has_header)
            .with_ignore_parser_errors(ignore_errors)
            .with_skip_rows(skip_rows)
            .with_stop_after_n_rows(stop_after_n_rows)
            .with_cache(cache)
            .with_dtype_overwrite(overwrite_dtype.as_ref())
            .low_memory(low_memory)
            .with_comment_char(comment_char)
            .with_quote_char(quote_char)
            .with_null_values(null_values)
            .finish()
            .into()
    }

    #[staticmethod]
    #[cfg(feature = "parquet")]
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
    pub fn to_dot(&self, optimized: bool) -> PyResult<String> {
        let result = self.ldf.to_dot(optimized).map_err(PyPolarsEr::from)?;
        Ok(result)
    }

    pub fn optimization_toggle(
        &self,
        type_coercion: bool,
        predicate_pushdown: bool,
        projection_pushdown: bool,
        simplify_expr: bool,
        string_cache: bool,
    ) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let ldf = ldf
            .with_type_coercion(type_coercion)
            .with_predicate_pushdown(predicate_pushdown)
            .with_simplify_expr(simplify_expr)
            .with_string_cache(string_cache)
            .with_projection_pushdown(projection_pushdown);
        ldf.into()
    }

    pub fn sort(&self, by_column: &str, reverse: bool) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.sort(by_column, reverse).into()
    }

    pub fn sort_by_exprs(&self, by_column: Vec<PyExpr>, reverse: Vec<bool>) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let exprs = py_exprs_to_exprs(by_column);
        ldf.sort_by_exprs(exprs, reverse).into()
    }
    pub fn cache(&self) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.cache().into()
    }

    pub fn collect(&self, py: Python) -> PyResult<PyDataFrame> {
        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        let df = py.allow_threads(|| {
            let ldf = self.ldf.clone();
            ldf.collect().map_err(PyPolarsEr::from)
        })?;
        Ok(df.into())
    }

    pub fn fetch(&self, n_rows: usize) -> PyResult<PyDataFrame> {
        let ldf = self.ldf.clone();
        let gil = Python::acquire_gil();
        let py = gil.python();
        let df = py.allow_threads(|| ldf.fetch(n_rows).map_err(PyPolarsEr::from))?;
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

    pub fn groupby(&mut self, by: Vec<PyExpr>, maintain_order: bool) -> PyLazyGroupBy {
        let ldf = self.ldf.clone();
        let by = py_exprs_to_exprs(by);
        let lazy_gb = if maintain_order {
            ldf.stable_groupby(by)
        } else {
            ldf.groupby(by)
        };

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    pub fn join(
        &mut self,
        other: PyLazyFrame,
        left_on: Vec<PyExpr>,
        right_on: Vec<PyExpr>,
        allow_parallel: bool,
        force_parallel: bool,
        how: &str,
        suffix: String,
    ) -> PyLazyFrame {
        let how = match how {
            "left" => JoinType::Left,
            "inner" => JoinType::Inner,
            "outer" => JoinType::Outer,
            "asof" => JoinType::AsOf,
            "cross" => JoinType::Cross,
            _ => panic!("not supported"),
        };

        let ldf = self.ldf.clone();
        let other = other.ldf;
        let left_on = left_on.into_iter().map(|pyexpr| pyexpr.inner).collect();
        let right_on = right_on.into_iter().map(|pyexpr| pyexpr.inner).collect();

        ldf.join_builder()
            .with(other)
            .left_on(left_on)
            .right_on(right_on)
            .allow_parallel(allow_parallel)
            .force_parallel(force_parallel)
            .how(how)
            .suffix(suffix)
            .finish()
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

    pub fn rename(&mut self, existing: Vec<String>, new: Vec<String>) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.rename(existing, new).into()
    }

    pub fn with_column_renamed(&mut self, existing: &str, new: &str) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.with_column_renamed(existing, new).into()
    }

    pub fn reverse(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.reverse().into()
    }

    pub fn shift(&self, periods: i64) -> Self {
        let ldf = self.ldf.clone();
        ldf.shift(periods).into()
    }

    pub fn shift_and_fill(&self, periods: i64, fill_value: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.shift_and_fill(periods, fill_value.inner).into()
    }

    pub fn fill_null(&self, fill_value: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.fill_null(fill_value.inner).into()
    }

    pub fn fill_nan(&self, fill_value: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.fill_nan(fill_value.inner).into()
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

    pub fn explode(&self, column: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        let column = py_exprs_to_exprs(column);
        ldf.explode(column).into()
    }

    pub fn drop_duplicates(&self, maintain_order: bool, subset: Option<Vec<String>>) -> Self {
        let ldf = self.ldf.clone();
        ldf.drop_duplicates(maintain_order, subset).into()
    }

    pub fn drop_nulls(&self, subset: Option<Vec<String>>) -> Self {
        let ldf = self.ldf.clone();
        ldf.drop_nulls(subset.map(|v| v.into_iter().map(|s| col(&s)).collect()))
            .into()
    }

    pub fn slice(&self, offset: i64, len: usize) -> Self {
        let ldf = self.ldf.clone();
        ldf.slice(offset, len).into()
    }

    pub fn tail(&self, n: usize) -> Self {
        let ldf = self.ldf.clone();
        ldf.tail(n).into()
    }

    pub fn melt(&self, id_vars: Vec<String>, value_vars: Vec<String>) -> Self {
        let ldf = self.ldf.clone();
        ldf.melt(id_vars, value_vars).into()
    }

    pub fn map(&self, lambda: PyObject, predicate_pd: bool, projection_pd: bool) -> Self {
        let opt = AllowedOptimizations {
            predicate_pushdown: predicate_pd,
            projection_pushdown: projection_pd,
            ..Default::default()
        };

        let function = move |s: DataFrame| {
            let gil = Python::acquire_gil();
            let py = gil.python();
            // get the pypolars module
            let pypolars = PyModule::import(py, "polars").unwrap();
            // create a PyDataFrame struct/object for Python
            let pydf = PyDataFrame::new(s);
            // Wrap this PyDataFrame object in the python side DataFrame wrapper
            let python_df_wrapper = pypolars.getattr("wrap_df").unwrap().call1((pydf,)).unwrap();
            // call the lambda and get a python side Series wrapper
            let result_df_wrapper = match lambda.call1(py, (python_df_wrapper,)) {
                Ok(pyobj) => pyobj,
                Err(e) => panic!("UDF failed: {}", e.pvalue(py).to_string()),
            };
            // unpack the wrapper in a PyDataFrame
            let py_pydf = result_df_wrapper.getattr(py, "_df").expect(
                "Could net get DataFrame attribute '_s'. Make sure that you return a DataFrame object.",
            );
            // Downcast to Rust
            let pydf = py_pydf.extract::<PyDataFrame>(py).unwrap();
            // Finally get the actual Series
            Ok(pydf.df)
        };

        let ldf = self.ldf.clone();
        ldf.map(function, Some(opt), None).into()
    }

    pub fn drop_columns(&self, cols: Vec<String>) -> Self {
        let ldf = self.ldf.clone();
        ldf.drop_columns(cols).into()
    }

    pub fn clone(&self) -> PyLazyFrame {
        self.ldf.clone().into()
    }

    pub fn columns(&self) -> Vec<String> {
        self.ldf
            .schema()
            .fields()
            .iter()
            .map(|fld| fld.name().to_string())
            .collect()
    }
}
