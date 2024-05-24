use std::borrow::Cow;

use polars::prelude::*;
use polars::series::ops::NullBehavior;
use pyo3::prelude::*;
use smartstring::alias::String as SmartString;

use crate::conversion::Wrap;
use crate::PyExpr;

#[pymethods]
impl PyExpr {
    #[cfg(feature = "list_any_all")]
    fn list_all(&self) -> Self {
        self.inner.clone().list().all().into()
    }

    #[cfg(feature = "list_any_all")]
    fn list_any(&self) -> Self {
        self.inner.clone().list().any().into()
    }

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
    fn list_count_matches(&self, expr: PyExpr) -> Self {
        self.inner.clone().list().count_matches(expr.inner).into()
    }

    fn list_diff(&self, n: i64, null_behavior: Wrap<NullBehavior>) -> PyResult<Self> {
        Ok(self.inner.clone().list().diff(n, null_behavior.0).into())
    }

    fn list_eval(&self, expr: PyExpr, parallel: bool) -> Self {
        self.inner.clone().list().eval(expr.inner, parallel).into()
    }

    fn list_get(&self, index: PyExpr, null_on_oob: bool) -> Self {
        self.inner
            .clone()
            .list()
            .get(index.inner, null_on_oob)
            .into()
    }

    fn list_join(&self, separator: PyExpr, ignore_nulls: bool) -> Self {
        self.inner
            .clone()
            .list()
            .join(separator.inner, ignore_nulls)
            .into()
    }

    fn list_len(&self) -> Self {
        self.inner.clone().list().len().into()
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

    fn list_median(&self) -> Self {
        self.inner
            .clone()
            .list()
            .median()
            .with_fmt("list.median")
            .into()
    }

    fn list_std(&self, ddof: u8) -> Self {
        self.inner
            .clone()
            .list()
            .std(ddof)
            .with_fmt("list.std")
            .into()
    }

    fn list_var(&self, ddof: u8) -> Self {
        self.inner
            .clone()
            .list()
            .var(ddof)
            .with_fmt("list.var")
            .into()
    }

    fn list_min(&self) -> Self {
        self.inner.clone().list().min().into()
    }

    fn list_reverse(&self) -> Self {
        self.inner.clone().list().reverse().into()
    }

    fn list_shift(&self, periods: PyExpr) -> Self {
        self.inner.clone().list().shift(periods.inner).into()
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

    fn list_sort(&self, descending: bool, nulls_last: bool) -> Self {
        self.inner
            .clone()
            .list()
            .sort(
                SortOptions::default()
                    .with_order_descending(descending)
                    .with_nulls_last(nulls_last),
            )
            .into()
    }

    fn list_sum(&self) -> Self {
        self.inner.clone().list().sum().with_fmt("list.sum").into()
    }

    #[cfg(feature = "list_drop_nulls")]
    fn list_drop_nulls(&self) -> Self {
        self.inner.clone().list().drop_nulls().into()
    }

    #[cfg(feature = "list_sample")]
    fn list_sample_n(
        &self,
        n: PyExpr,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        self.inner
            .clone()
            .list()
            .sample_n(n.inner, with_replacement, shuffle, seed)
            .into()
    }

    #[cfg(feature = "list_sample")]
    fn list_sample_fraction(
        &self,
        fraction: PyExpr,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        self.inner
            .clone()
            .list()
            .sample_fraction(fraction.inner, with_replacement, shuffle, seed)
            .into()
    }

    #[cfg(feature = "list_gather")]
    fn list_gather(&self, index: PyExpr, null_on_oob: bool) -> Self {
        self.inner
            .clone()
            .list()
            .gather(index.inner, null_on_oob)
            .into()
    }

    #[cfg(feature = "list_gather")]
    fn list_gather_every(&self, n: PyExpr, offset: PyExpr) -> Self {
        self.inner
            .clone()
            .list()
            .gather_every(n.inner, offset.inner)
            .into()
    }

    fn list_to_array(&self, width: usize) -> Self {
        self.inner.clone().list().to_array(width).into()
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
                    let out: SmartString = out.extract::<Cow<str>>(py).unwrap().into();
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

    fn list_n_unique(&self) -> Self {
        self.inner.clone().list().n_unique().into()
    }

    fn list_unique(&self, maintain_order: bool) -> Self {
        let e = self.inner.clone();

        if maintain_order {
            e.list().unique_stable().into()
        } else {
            e.list().unique().into()
        }
    }

    #[cfg(feature = "list_sets")]
    fn list_set_operation(&self, other: PyExpr, operation: Wrap<SetOperation>) -> Self {
        let e = self.inner.clone().list();
        match operation.0 {
            SetOperation::Intersection => e.set_intersection(other.inner),
            SetOperation::Difference => e.set_difference(other.inner),
            SetOperation::Union => e.union(other.inner),
            SetOperation::SymmetricDifference => e.set_symmetric_difference(other.inner),
        }
        .into()
    }
}
