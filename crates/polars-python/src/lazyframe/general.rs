use std::collections::HashMap;

use polars::time::*;
use polars_core::prelude::*;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;

use super::PyLazyFrame;
use crate::error::PyPolarsErr;
use crate::expr::ToExprs;
use crate::lazyframe::visit::NodeTraverser;
use crate::prelude::*;
use crate::{PyDataFrame, PyExpr, PyLazyGroupBy};

#[pymethods]
#[allow(clippy::should_implement_trait)]
impl PyLazyFrame {
    fn describe_plan(&self) -> PyResult<String> {
        self.ldf
            .describe_plan()
            .map_err(PyPolarsErr::from)
            .map_err(Into::into)
    }

    fn describe_optimized_plan(&self) -> PyResult<String> {
        self.ldf
            .describe_optimized_plan()
            .map_err(PyPolarsErr::from)
            .map_err(Into::into)
    }

    fn describe_plan_tree(&self) -> PyResult<String> {
        self.ldf
            .describe_plan_tree()
            .map_err(PyPolarsErr::from)
            .map_err(Into::into)
    }

    fn describe_optimized_plan_tree(&self) -> PyResult<String> {
        self.ldf
            .describe_optimized_plan_tree()
            .map_err(PyPolarsErr::from)
            .map_err(Into::into)
    }

    fn to_dot(&self, optimized: bool) -> PyResult<String> {
        let result = self.ldf.to_dot(optimized).map_err(PyPolarsErr::from)?;
        Ok(result)
    }

    fn optimization_toggle(
        &self,
        type_coercion: bool,
        predicate_pushdown: bool,
        projection_pushdown: bool,
        simplify_expression: bool,
        slice_pushdown: bool,
        comm_subplan_elim: bool,
        comm_subexpr_elim: bool,
        cluster_with_columns: bool,
        collapse_joins: bool,
        streaming: bool,
        _eager: bool,
        #[allow(unused_variables)] new_streaming: bool,
    ) -> Self {
        let ldf = self.ldf.clone();
        let mut ldf = ldf
            .with_type_coercion(type_coercion)
            .with_predicate_pushdown(predicate_pushdown)
            .with_simplify_expr(simplify_expression)
            .with_slice_pushdown(slice_pushdown)
            .with_cluster_with_columns(cluster_with_columns)
            .with_collapse_joins(collapse_joins)
            ._with_eager(_eager)
            .with_projection_pushdown(projection_pushdown);

        #[cfg(feature = "streaming")]
        {
            ldf = ldf.with_streaming(streaming);
        }

        #[cfg(feature = "new_streaming")]
        {
            ldf = ldf.with_new_streaming(new_streaming);
        }

        #[cfg(feature = "cse")]
        {
            ldf = ldf.with_comm_subplan_elim(comm_subplan_elim);
            ldf = ldf.with_comm_subexpr_elim(comm_subexpr_elim);
        }

        ldf.into()
    }

    fn sort(
        &self,
        by_column: &str,
        descending: bool,
        nulls_last: bool,
        maintain_order: bool,
        multithreaded: bool,
    ) -> Self {
        let ldf = self.ldf.clone();
        ldf.sort(
            [by_column],
            SortMultipleOptions {
                descending: vec![descending],
                nulls_last: vec![nulls_last],
                multithreaded,
                maintain_order,
                limit: None,
            },
        )
        .into()
    }

    fn sort_by_exprs(
        &self,
        by: Vec<PyExpr>,
        descending: Vec<bool>,
        nulls_last: Vec<bool>,
        maintain_order: bool,
        multithreaded: bool,
    ) -> Self {
        let ldf = self.ldf.clone();
        let exprs = by.to_exprs();
        ldf.sort_by_exprs(
            exprs,
            SortMultipleOptions {
                descending,
                nulls_last,
                maintain_order,
                multithreaded,
                limit: None,
            },
        )
        .into()
    }

    fn top_k(&self, k: IdxSize, by: Vec<PyExpr>, reverse: Vec<bool>) -> Self {
        let ldf = self.ldf.clone();
        let exprs = by.to_exprs();
        ldf.top_k(
            k,
            exprs,
            SortMultipleOptions::new().with_order_descending_multi(reverse),
        )
        .into()
    }

    fn bottom_k(&self, k: IdxSize, by: Vec<PyExpr>, reverse: Vec<bool>) -> Self {
        let ldf = self.ldf.clone();
        let exprs = by.to_exprs();
        ldf.bottom_k(
            k,
            exprs,
            SortMultipleOptions::new().with_order_descending_multi(reverse),
        )
        .into()
    }

    fn cache(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.cache().into()
    }

    fn profile(&self, py: Python) -> PyResult<(PyDataFrame, PyDataFrame)> {
        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        let (df, time_df) = py.allow_threads(|| {
            let ldf = self.ldf.clone();
            ldf.profile().map_err(PyPolarsErr::from)
        })?;
        Ok((df.into(), time_df.into()))
    }

    #[pyo3(signature = (lambda_post_opt=None))]
    fn collect(&self, py: Python, lambda_post_opt: Option<PyObject>) -> PyResult<PyDataFrame> {
        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        let df = py.allow_threads(|| {
            let ldf = self.ldf.clone();
            if let Some(lambda) = lambda_post_opt {
                ldf._collect_post_opt(|root, lp_arena, expr_arena| {
                    Python::with_gil(|py| {
                        let nt = NodeTraverser::new(
                            root,
                            std::mem::take(lp_arena),
                            std::mem::take(expr_arena),
                        );

                        // Get a copy of the arena's.
                        let arenas = nt.get_arenas();

                        // Pass the node visitor which allows the python callback to replace parts of the query plan.
                        // Remove "cuda" or specify better once we have multiple post-opt callbacks.
                        lambda.call1(py, (nt,)).map_err(
                            |e| polars_err!(ComputeError: "'cuda' conversion failed: {}", e),
                        )?;

                        // Unpack the arena's.
                        // At this point the `nt` is useless.

                        std::mem::swap(lp_arena, &mut *arenas.0.lock().unwrap());
                        std::mem::swap(expr_arena, &mut *arenas.1.lock().unwrap());

                        Ok(())
                    })
                })
            } else {
                ldf.collect()
            }
            .map_err(PyPolarsErr::from)
        })?;
        Ok(df.into())
    }

    #[pyo3(signature = (lambda,))]
    fn collect_with_callback(&self, lambda: PyObject) {
        let ldf = self.ldf.clone();

        polars_core::POOL.spawn(move || {
            let result = ldf
                .collect()
                .map(PyDataFrame::new)
                .map_err(PyPolarsErr::from);

            Python::with_gil(|py| match result {
                Ok(df) => {
                    lambda.call1(py, (df,)).map_err(|err| err.restore(py)).ok();
                },
                Err(err) => {
                    lambda
                        .call1(py, (PyErr::from(err).to_object(py),))
                        .map_err(|err| err.restore(py))
                        .ok();
                },
            });
        });
    }

    fn fetch(&self, py: Python, n_rows: usize) -> PyResult<PyDataFrame> {
        let ldf = self.ldf.clone();
        let df = py.allow_threads(|| ldf.fetch(n_rows).map_err(PyPolarsErr::from))?;
        Ok(df.into())
    }

    fn filter(&mut self, predicate: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.filter(predicate.inner).into()
    }

    fn select(&mut self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        let exprs = exprs.to_exprs();
        ldf.select(exprs).into()
    }

    fn select_seq(&mut self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        let exprs = exprs.to_exprs();
        ldf.select_seq(exprs).into()
    }

    fn group_by(&mut self, by: Vec<PyExpr>, maintain_order: bool) -> PyLazyGroupBy {
        let ldf = self.ldf.clone();
        let by = by.to_exprs();
        let lazy_gb = if maintain_order {
            ldf.group_by_stable(by)
        } else {
            ldf.group_by(by)
        };

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    fn rolling(
        &mut self,
        index_column: PyExpr,
        period: &str,
        offset: &str,
        closed: Wrap<ClosedWindow>,
        by: Vec<PyExpr>,
    ) -> PyResult<PyLazyGroupBy> {
        let closed_window = closed.0;
        let ldf = self.ldf.clone();
        let by = by
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();
        let lazy_gb = ldf.rolling(
            index_column.inner,
            by,
            RollingGroupOptions {
                index_column: "".into(),
                period: Duration::try_parse(period).map_err(PyPolarsErr::from)?,
                offset: Duration::try_parse(offset).map_err(PyPolarsErr::from)?,
                closed_window,
            },
        );

        Ok(PyLazyGroupBy { lgb: Some(lazy_gb) })
    }

    fn group_by_dynamic(
        &mut self,
        index_column: PyExpr,
        every: &str,
        period: &str,
        offset: &str,
        label: Wrap<Label>,
        include_boundaries: bool,
        closed: Wrap<ClosedWindow>,
        group_by: Vec<PyExpr>,
        start_by: Wrap<StartBy>,
    ) -> PyResult<PyLazyGroupBy> {
        let closed_window = closed.0;
        let group_by = group_by
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();
        let ldf = self.ldf.clone();
        let lazy_gb = ldf.group_by_dynamic(
            index_column.inner,
            group_by,
            DynamicGroupOptions {
                every: Duration::try_parse(every).map_err(PyPolarsErr::from)?,
                period: Duration::try_parse(period).map_err(PyPolarsErr::from)?,
                offset: Duration::try_parse(offset).map_err(PyPolarsErr::from)?,
                label: label.0,
                include_boundaries,
                closed_window,
                start_by: start_by.0,
                ..Default::default()
            },
        );

        Ok(PyLazyGroupBy { lgb: Some(lazy_gb) })
    }

    fn with_context(&self, contexts: Vec<Self>) -> Self {
        let contexts = contexts.into_iter().map(|ldf| ldf.ldf).collect::<Vec<_>>();
        self.ldf.clone().with_context(contexts).into()
    }

    #[cfg(feature = "asof_join")]
    #[pyo3(signature = (other, left_on, right_on, left_by, right_by, allow_parallel, force_parallel, suffix, strategy, tolerance, tolerance_str, coalesce))]
    fn join_asof(
        &self,
        other: Self,
        left_on: PyExpr,
        right_on: PyExpr,
        left_by: Option<Vec<PyBackedStr>>,
        right_by: Option<Vec<PyBackedStr>>,
        allow_parallel: bool,
        force_parallel: bool,
        suffix: String,
        strategy: Wrap<AsofStrategy>,
        tolerance: Option<Wrap<AnyValue<'_>>>,
        tolerance_str: Option<String>,
        coalesce: bool,
    ) -> PyResult<Self> {
        let coalesce = if coalesce {
            JoinCoalesce::CoalesceColumns
        } else {
            JoinCoalesce::KeepColumns
        };
        let ldf = self.ldf.clone();
        let other = other.ldf;
        let left_on = left_on.inner;
        let right_on = right_on.inner;
        Ok(ldf
            .join_builder()
            .with(other)
            .left_on([left_on])
            .right_on([right_on])
            .allow_parallel(allow_parallel)
            .force_parallel(force_parallel)
            .coalesce(coalesce)
            .how(JoinType::AsOf(AsOfOptions {
                strategy: strategy.0,
                left_by: left_by.map(strings_to_pl_smallstr),
                right_by: right_by.map(strings_to_pl_smallstr),
                tolerance: tolerance.map(|t| t.0.into_static()),
                tolerance_str: tolerance_str.map(|s| s.into()),
            }))
            .suffix(suffix)
            .finish()
            .into())
    }

    #[pyo3(signature = (other, left_on, right_on, allow_parallel, force_parallel, join_nulls, how, suffix, validate, maintain_order, coalesce=None))]
    fn join(
        &self,
        other: Self,
        left_on: Vec<PyExpr>,
        right_on: Vec<PyExpr>,
        allow_parallel: bool,
        force_parallel: bool,
        join_nulls: bool,
        how: Wrap<JoinType>,
        suffix: String,
        validate: Wrap<JoinValidation>,
        maintain_order: Wrap<MaintainOrderJoin>,
        coalesce: Option<bool>,
    ) -> PyResult<Self> {
        let coalesce = match coalesce {
            None => JoinCoalesce::JoinSpecific,
            Some(true) => JoinCoalesce::CoalesceColumns,
            Some(false) => JoinCoalesce::KeepColumns,
        };
        let ldf = self.ldf.clone();
        let other = other.ldf;
        let left_on = left_on
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();
        let right_on = right_on
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();

        Ok(ldf
            .join_builder()
            .with(other)
            .left_on(left_on)
            .right_on(right_on)
            .allow_parallel(allow_parallel)
            .force_parallel(force_parallel)
            .join_nulls(join_nulls)
            .how(how.0)
            .suffix(suffix)
            .validate(validate.0)
            .coalesce(coalesce)
            .maintain_order(maintain_order.0)
            .finish()
            .into())
    }

    fn join_where(&self, other: Self, predicates: Vec<PyExpr>, suffix: String) -> PyResult<Self> {
        let ldf = self.ldf.clone();
        let other = other.ldf;

        let predicates = predicates.to_exprs();

        Ok(ldf
            .join_builder()
            .with(other)
            .suffix(suffix)
            .join_where(predicates)
            .into())
    }

    fn with_columns(&mut self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        ldf.with_columns(exprs.to_exprs()).into()
    }

    fn with_columns_seq(&mut self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        ldf.with_columns_seq(exprs.to_exprs()).into()
    }

    fn rename(&mut self, existing: Vec<String>, new: Vec<String>, strict: bool) -> Self {
        let ldf = self.ldf.clone();
        ldf.rename(existing, new, strict).into()
    }

    fn reverse(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.reverse().into()
    }

    #[pyo3(signature = (n, fill_value=None))]
    fn shift(&self, n: PyExpr, fill_value: Option<PyExpr>) -> Self {
        let lf = self.ldf.clone();
        let out = match fill_value {
            Some(v) => lf.shift_and_fill(n.inner, v.inner),
            None => lf.shift(n.inner),
        };
        out.into()
    }

    fn fill_nan(&self, fill_value: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.fill_nan(fill_value.inner).into()
    }

    fn min(&self) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.min();
        out.into()
    }

    fn max(&self) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.max();
        out.into()
    }

    fn sum(&self) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.sum();
        out.into()
    }

    fn mean(&self) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.mean();
        out.into()
    }

    fn std(&self, ddof: u8) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.std(ddof);
        out.into()
    }

    fn var(&self, ddof: u8) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.var(ddof);
        out.into()
    }

    fn median(&self) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.median();
        out.into()
    }

    fn quantile(&self, quantile: PyExpr, interpolation: Wrap<QuantileMethod>) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.quantile(quantile.inner, interpolation.0);
        out.into()
    }

    fn explode(&self, column: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        let column = column.to_exprs();
        ldf.explode(column).into()
    }

    fn null_count(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.null_count().into()
    }

    #[pyo3(signature = (maintain_order, subset, keep))]
    fn unique(
        &self,
        maintain_order: bool,
        subset: Option<Vec<PyExpr>>,
        keep: Wrap<UniqueKeepStrategy>,
    ) -> Self {
        let ldf = self.ldf.clone();
        let subset = subset.map(|e| e.to_exprs());
        match maintain_order {
            true => ldf.unique_stable_generic(subset, keep.0),
            false => ldf.unique_generic(subset, keep.0),
        }
        .into()
    }

    #[pyo3(signature = (subset=None))]
    fn drop_nans(&self, subset: Option<Vec<PyExpr>>) -> Self {
        let ldf = self.ldf.clone();
        let subset = subset.map(|e| e.to_exprs());
        ldf.drop_nans(subset).into()
    }

    #[pyo3(signature = (subset=None))]
    fn drop_nulls(&self, subset: Option<Vec<PyExpr>>) -> Self {
        let ldf = self.ldf.clone();
        let subset = subset.map(|e| e.to_exprs());
        ldf.drop_nulls(subset).into()
    }

    #[pyo3(signature = (offset, len=None))]
    fn slice(&self, offset: i64, len: Option<IdxSize>) -> Self {
        let ldf = self.ldf.clone();
        ldf.slice(offset, len.unwrap_or(IdxSize::MAX)).into()
    }

    fn tail(&self, n: IdxSize) -> Self {
        let ldf = self.ldf.clone();
        ldf.tail(n).into()
    }

    #[cfg(feature = "pivot")]
    #[pyo3(signature = (on, index, value_name, variable_name))]
    fn unpivot(
        &self,
        on: Vec<PyExpr>,
        index: Vec<PyExpr>,
        value_name: Option<String>,
        variable_name: Option<String>,
    ) -> Self {
        let args = UnpivotArgsDSL {
            on: on.into_iter().map(|e| e.inner.into()).collect(),
            index: index.into_iter().map(|e| e.inner.into()).collect(),
            value_name: value_name.map(|s| s.into()),
            variable_name: variable_name.map(|s| s.into()),
        };

        let ldf = self.ldf.clone();
        ldf.unpivot(args).into()
    }

    #[pyo3(signature = (name, offset=None))]
    fn with_row_index(&self, name: &str, offset: Option<IdxSize>) -> Self {
        let ldf = self.ldf.clone();
        ldf.with_row_index(name, offset).into()
    }

    #[pyo3(signature = (lambda, predicate_pushdown, projection_pushdown, slice_pushdown, streamable, schema, validate_output))]
    fn map_batches(
        &self,
        lambda: PyObject,
        predicate_pushdown: bool,
        projection_pushdown: bool,
        slice_pushdown: bool,
        streamable: bool,
        schema: Option<Wrap<Schema>>,
        validate_output: bool,
    ) -> Self {
        let mut opt = OptFlags::default();
        opt.set(OptFlags::PREDICATE_PUSHDOWN, predicate_pushdown);
        opt.set(OptFlags::PROJECTION_PUSHDOWN, projection_pushdown);
        opt.set(OptFlags::SLICE_PUSHDOWN, slice_pushdown);
        opt.set(OptFlags::STREAMING, streamable);

        self.ldf
            .clone()
            .map_python(
                lambda.into(),
                opt,
                schema.map(|s| Arc::new(s.0)),
                validate_output,
            )
            .into()
    }

    fn drop(&self, columns: Vec<PyExpr>, strict: bool) -> Self {
        let ldf = self.ldf.clone();
        let columns = columns.to_exprs();
        if strict {
            ldf.drop(columns)
        } else {
            ldf.drop_no_validate(columns)
        }
        .into()
    }

    fn cast(&self, dtypes: HashMap<PyBackedStr, Wrap<DataType>>, strict: bool) -> Self {
        let mut cast_map = PlHashMap::with_capacity(dtypes.len());
        cast_map.extend(dtypes.iter().map(|(k, v)| (k.as_ref(), v.0.clone())));
        self.ldf.clone().cast(cast_map, strict).into()
    }

    fn cast_all(&self, dtype: Wrap<DataType>, strict: bool) -> Self {
        self.ldf.clone().cast_all(dtype.0, strict).into()
    }

    fn clone(&self) -> Self {
        self.ldf.clone().into()
    }

    fn collect_schema(&mut self, py: Python) -> PyResult<PyObject> {
        let schema = py
            .allow_threads(|| self.ldf.collect_schema())
            .map_err(PyPolarsErr::from)?;

        let schema_dict = PyDict::new_bound(py);
        schema.iter_fields().for_each(|fld| {
            schema_dict
                .set_item(fld.name().as_str(), Wrap(fld.dtype().clone()))
                .unwrap()
        });
        Ok(schema_dict.to_object(py))
    }

    fn unnest(&self, columns: Vec<PyExpr>) -> Self {
        let columns = columns.to_exprs();
        self.ldf.clone().unnest(columns).into()
    }

    fn count(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.count().into()
    }

    #[cfg(feature = "merge_sorted")]
    fn merge_sorted(&self, other: Self, key: &str) -> PyResult<Self> {
        let out = self
            .ldf
            .clone()
            .merge_sorted(other.ldf, key)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }
}
