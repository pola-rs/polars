use polars::lazy::dsl::lit;
use polars::prelude::*;
use polars::series::ops::NullBehavior;
use pyo3::prelude::*;
use smartstring::alias::String as SmartString;

use crate::conversion::Wrap;
use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn list_arg_max(&self) -> Self {
        self.inner.clone().list().arg_max().into()
    }

    fn list_arg_min(&self) -> Self {
        self.inner.clone().list().arg_min().into()
    }

    #[cfg(feature = "is_in")]
    fn list_contains(&self, other: PyExpr) -> Self {
        self.inner.clone().list().contains(other.inner).into()
    }

    #[cfg(feature = "list_count")]
    fn list_count_match(&self, expr: PyExpr) -> Self {
        self.inner.clone().list().count_match(expr.inner).into()
    }

    fn list_diff(&self, n: i64, null_behavior: Wrap<NullBehavior>) -> PyResult<Self> {
        Ok(self.inner.clone().list().diff(n, null_behavior.0).into())
    }

    fn list_eval(&self, expr: PyExpr, parallel: bool) -> Self {
        self.inner.clone().list().eval(expr.inner, parallel).into()
    }

    fn list_get(&self, index: PyExpr) -> Self {
        self.inner.clone().list().get(index.inner).into()
    }

    fn list_join(&self, separator: &str) -> Self {
        self.inner.clone().list().join(separator).into()
    }

    fn list_lengths(&self) -> Self {
        self.inner.clone().list().lengths().into()
    }

    fn list_max(&self) -> Self {
        self.inner.clone().list().max().into()
    }

    fn list_mean(&self) -> Self {
        self.inner
            .clone()
            .list()
            .mean()
            .with_fmt("list.mean")
            .into()
    }

    fn list_min(&self) -> Self {
        self.inner.clone().list().min().into()
    }

    fn list_reverse(&self) -> Self {
        self.inner.clone().list().reverse().into()
    }

    fn list_shift(&self, periods: i64) -> Self {
        self.inner.clone().list().shift(periods).into()
    }

    fn list_slice(&self, offset: PyExpr, length: Option<PyExpr>) -> Self {
        let length = match length {
            Some(i) => i.inner,
            None => lit(i64::MAX),
        };
        self.inner.clone().list().slice(offset.inner, length).into()
    }

    fn list_tail(&self, n: PyExpr) -> Self {
        self.inner.clone().list().tail(n.inner).into()
    }

    fn list_sort(&self, descending: bool) -> Self {
        self.inner
            .clone()
            .list()
            .sort(SortOptions {
                descending,
                ..Default::default()
            })
            .with_fmt("list.sort")
            .into()
    }

    fn list_sum(&self) -> Self {
        self.inner.clone().list().sum().with_fmt("list.sum").into()
    }

    #[cfg(feature = "list_take")]
    fn list_take(&self, index: PyExpr, null_on_oob: bool) -> Self {
        self.inner
            .clone()
            .list()
            .take(index.inner, null_on_oob)
            .into()
    }

    #[pyo3(signature = (width_strat, name_gen, upper_bound))]
    fn list_to_struct(
        &self,
        width_strat: Wrap<ListToStructWidthStrategy>,
        name_gen: Option<PyObject>,
        upper_bound: usize,
    ) -> PyResult<Self> {
        let name_gen = name_gen.map(|lambda| {
            Arc::new(move |idx: usize| {
                Python::with_gil(|py| {
                    let out = lambda.call1(py, (idx,)).unwrap();
                    let out: SmartString = out.extract::<&str>(py).unwrap().into();
                    out
                })
            }) as NameGenerator
        });

        Ok(self
            .inner
            .clone()
            .list()
            .to_struct(width_strat.0, name_gen, upper_bound)
            .into())
    }

    fn list_unique(&self, maintain_order: bool) -> Self {
        let e = self.inner.clone();

        if maintain_order {
            e.list().unique_stable().into()
        } else {
            e.list().unique().into()
        }
    }
}
