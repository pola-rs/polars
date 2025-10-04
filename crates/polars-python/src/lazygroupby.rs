use std::sync::Arc;

use polars::lazy::frame::{LazyFrame, LazyGroupBy};
use polars::prelude::{PlanCallback, Schema};
use polars_utils::python_function::PythonObject;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::expr::ToExprs;
use crate::{PyExpr, PyLazyFrame};

#[pyclass(frozen)]
#[repr(transparent)]
pub struct PyLazyGroupBy {
    // option because we cannot get a self by value in pyo3
    pub lgb: Option<LazyGroupBy>,
}

#[pymethods]
impl PyLazyGroupBy {
    fn agg(&self, aggs: Vec<PyExpr>) -> PyLazyFrame {
        let lgb = self.lgb.clone().unwrap();
        let aggs = aggs.to_exprs();
        lgb.agg(aggs).into()
    }

    fn head(&self, n: usize) -> PyLazyFrame {
        let lgb = self.lgb.clone().unwrap();
        lgb.head(Some(n)).into()
    }

    fn tail(&self, n: usize) -> PyLazyFrame {
        let lgb = self.lgb.clone().unwrap();
        lgb.tail(Some(n)).into()
    }

    #[pyo3(signature = (lambda, schema))]
    fn map_groups(&self, lambda: Py<PyAny>, schema: Option<Wrap<Schema>>) -> PyResult<PyLazyFrame> {
        let lgb = self.lgb.clone().unwrap();
        let schema = match schema {
            Some(schema) => Arc::new(schema.0),
            None => LazyFrame::from(lgb.logical_plan.clone())
                .collect_schema()
                .map_err(PyPolarsErr::from)?,
        };

        let function = PythonObject(lambda);

        Ok(lgb.apply(PlanCallback::new_python(function), schema).into())
    }
}
