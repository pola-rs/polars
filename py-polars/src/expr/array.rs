use polars::prelude::*;
use pyo3::pymethods;

use crate::expr::PyExpr;

#[pymethods]
impl PyExpr {
    fn arr_max(&self) -> Self {
        self.inner.clone().arr().max().into()
    }

    fn arr_min(&self) -> Self {
        self.inner.clone().arr().min().into()
    }

    fn arr_sum(&self) -> Self {
        self.inner.clone().arr().sum().into()
    }

    fn arr_std(&self, ddof: u8) -> Self {
        self.inner.clone().arr().std(ddof).into()
    }

    fn arr_var(&self, ddof: u8) -> Self {
        self.inner.clone().arr().var(ddof).into()
    }

    fn arr_median(&self) -> Self {
        self.inner.clone().arr().median().into()
    }

    fn arr_unique(&self, maintain_order: bool) -> Self {
        if maintain_order {
            self.inner.clone().arr().unique_stable().into()
        } else {
            self.inner.clone().arr().unique().into()
        }
    }

    fn arr_to_list(&self) -> Self {
        self.inner.clone().arr().to_list().into()
    }

    fn arr_all(&self) -> Self {
        self.inner.clone().arr().all().into()
    }

    fn arr_any(&self) -> Self {
        self.inner.clone().arr().any().into()
    }

    fn arr_sort(&self, descending: bool, nulls_last: bool) -> Self {
        self.inner
            .clone()
            .arr()
            .sort(SortOptions {
                descending,
                nulls_last,
                ..Default::default()
            })
            .into()
    }

    fn arr_reverse(&self) -> Self {
        self.inner.clone().arr().reverse().into()
    }

    fn arr_arg_min(&self) -> Self {
        self.inner.clone().arr().arg_min().into()
    }

    fn arr_arg_max(&self) -> Self {
        self.inner.clone().arr().arg_max().into()
    }

    fn arr_get(&self, index: PyExpr) -> Self {
        self.inner.clone().arr().get(index.inner).into()
    }

    fn arr_join(&self, separator: PyExpr, ignore_nulls: bool) -> Self {
        self.inner
            .clone()
            .arr()
            .join(separator.inner, ignore_nulls)
            .into()
    }

    #[cfg(feature = "is_in")]
    fn arr_contains(&self, other: PyExpr) -> Self {
        self.inner.clone().arr().contains(other.inner).into()
    }

    #[cfg(feature = "array_count")]
    fn arr_count_matches(&self, expr: PyExpr) -> Self {
        self.inner.clone().arr().count_matches(expr.inner).into()
    }
}
