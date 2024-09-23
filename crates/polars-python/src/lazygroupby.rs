use std::sync::Arc;

use polars::lazy::frame::{LazyFrame, LazyGroupBy};
use polars::prelude::{DataFrame, PolarsError, Schema};
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::expr::ToExprs;
use crate::{PyDataFrame, PyExpr, PyLazyFrame};

#[pyclass]
#[repr(transparent)]
pub struct PyLazyGroupBy {
    // option because we cannot get a self by value in pyo3
    pub lgb: Option<LazyGroupBy>,
}

#[pymethods]
impl PyLazyGroupBy {
    fn agg(&mut self, aggs: Vec<PyExpr>) -> PyLazyFrame {
        let lgb = self.lgb.clone().unwrap();
        let aggs = aggs.to_exprs();
        lgb.agg(aggs).into()
    }

    fn head(&mut self, n: usize) -> PyLazyFrame {
        let lgb = self.lgb.clone().unwrap();
        lgb.head(Some(n)).into()
    }

    fn tail(&mut self, n: usize) -> PyLazyFrame {
        let lgb = self.lgb.clone().unwrap();
        lgb.tail(Some(n)).into()
    }

    fn map_groups(
        &mut self,
        lambda: PyObject,
        schema: Option<Wrap<Schema>>,
    ) -> PyResult<PyLazyFrame> {
        let lgb = self.lgb.clone().unwrap();
        let schema = match schema {
            Some(schema) => Arc::new(schema.0),
            None => LazyFrame::from(lgb.logical_plan.clone())
                .collect_schema()
                .map_err(PyPolarsErr::from)?,
        };

        let function = move |df: DataFrame| {
            Python::with_gil(|py| {
                // get the pypolars module
                let pypolars = PyModule::import_bound(py, "polars").unwrap();

                // create a PyDataFrame struct/object for Python
                let pydf = PyDataFrame::new(df);

                // Wrap this PySeries object in the python side DataFrame wrapper
                let python_df_wrapper =
                    pypolars.getattr("wrap_df").unwrap().call1((pydf,)).unwrap();

                // call the lambda and get a python side DataFrame wrapper
                let result_df_wrapper = lambda.call1(py, (python_df_wrapper,)).map_err(|e| {
                    PolarsError::ComputeError(
                        format!("User provided python function failed: {e}").into(),
                    )
                })?;
                // unpack the wrapper in a PyDataFrame
                let py_pydf = result_df_wrapper.getattr(py, "_df").expect(
                "Could not get DataFrame attribute '_df'. Make sure that you return a DataFrame object.",
            );
                // Downcast to Rust
                let pydf = py_pydf.extract::<PyDataFrame>(py).unwrap();
                // Finally get the actual DataFrame
                Ok(pydf.df)
            })
        };
        Ok(lgb.apply(function, schema).into())
    }
}
