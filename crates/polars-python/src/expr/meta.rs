use std::sync::Arc;

use polars::prelude::{AggExpr, Expr, Schema, Selector};
use polars_error::{PolarsResult, polars_bail};
use pyo3::prelude::*;

use crate::PyExpr;
use crate::error::PyPolarsErr;
use crate::expr::ToPyExprs;
use crate::prelude::Wrap;

#[pymethods]
impl PyExpr {
    fn meta_eq(&self, other: Self) -> bool {
        self.inner == other.inner
    }

    fn meta_pop(&self, schema: Option<Wrap<Schema>>) -> PyResult<Vec<Self>> {
        let schema = schema.as_ref().map(|s| &s.0);
        let exprs = self
            .inner
            .clone()
            .meta()
            .pop(schema)
            .map_err(PyPolarsErr::from)?;
        Ok(exprs.to_pyexprs())
    }

    fn meta_root_names(&self) -> Vec<String> {
        self.inner
            .clone()
            .meta()
            .root_names()
            .iter()
            .map(|name| name.to_string())
            .collect()
    }

    fn meta_output_name(&self) -> PyResult<String> {
        let name = self
            .inner
            .clone()
            .meta()
            .output_name()
            .map_err(PyPolarsErr::from)?;
        Ok(name.to_string())
    }

    fn meta_undo_aliases(&self) -> Self {
        self.inner.clone().meta().undo_aliases().into()
    }

    fn meta_has_multiple_outputs(&self) -> bool {
        self.inner.clone().meta().has_multiple_outputs()
    }

    fn meta_is_column(&self) -> bool {
        self.inner.clone().meta().is_column()
    }

    fn meta_is_regex_projection(&self) -> bool {
        self.inner.clone().meta().is_regex_projection()
    }

    fn meta_is_column_selection(&self, allow_aliasing: bool) -> bool {
        self.inner
            .clone()
            .meta()
            .is_column_selection(allow_aliasing)
    }

    fn meta_is_literal(&self, allow_aliasing: bool) -> bool {
        self.inner.clone().meta().is_literal(allow_aliasing)
    }

    fn compute_tree_format(
        &self,
        display_as_dot: bool,
        schema: Option<Wrap<Schema>>,
    ) -> Result<String, PyErr> {
        let e = self
            .inner
            .clone()
            .meta()
            .into_tree_formatter(display_as_dot, schema.as_ref().map(|s| &s.0))
            .map_err(PyPolarsErr::from)?;
        Ok(format!("{e}"))
    }

    fn meta_tree_format(&self, schema: Option<Wrap<Schema>>) -> PyResult<String> {
        self.compute_tree_format(false, schema)
    }

    fn meta_show_graph(&self, schema: Option<Wrap<Schema>>) -> PyResult<String> {
        self.compute_tree_format(true, schema)
    }

    fn meta_replace_element(&self, expr: PyExpr) -> PyResult<PyExpr> {
        #[recursive::recursive]
        fn rec(expr: &Expr, with: Expr) -> PolarsResult<Expr> {
            use Expr as E;
            Ok(match expr.clone() {
                E::Element => with.clone(),
                E::Alias(expr, name) => E::Alias(Arc::new(rec(&expr, with)?), name),
                E::Column(n) if n.is_empty() => with.clone(),
                E::Column(_) => {
                    polars_bail!(InvalidOperation: "`col` is not allowed in this context")
                },
                E::Selector(Selector::ByIndex { indices, strict: _ })
                    if indices.len() == 1 && indices[0] == 0 =>
                {
                    with.clone()
                },
                e @ (E::Selector(_) | E::Literal(_) | E::SubPlan(..) | E::Field(..) | E::Len) => e,
                E::DataTypeFunction(e) => todo!(),
                E::BinaryExpr { left, op, right } => E::BinaryExpr {
                    left: Arc::new(rec(&left, with.clone())?),
                    op,
                    right: Arc::new(rec(&right, with)?),
                },
                E::Cast {
                    expr,
                    dtype,
                    options,
                } => E::Cast {
                    expr: Arc::new(rec(&expr, with)?),
                    dtype,
                    options,
                },
                E::Sort { expr, options } => E::Sort {
                    expr: Arc::new(rec(&expr, with)?),
                    options,
                },
                E::Gather {
                    expr,
                    idx,
                    returns_scalar,
                } => E::Gather {
                    expr: Arc::new(rec(&expr, with.clone())?),
                    idx: Arc::new(rec(&idx, with)?),
                    returns_scalar,
                },
                E::SortBy {
                    expr,
                    by,
                    sort_options,
                } => E::SortBy {
                    expr: Arc::new(rec(&expr, with.clone())?),
                    by: by
                        .iter()
                        .map(|by| rec(by, with.clone()))
                        .collect::<PolarsResult<Vec<_>>>()?,
                    sort_options,
                },
                E::Agg(mut agg_expr) => {
                    use AggExpr as A;
                    let input = match &mut agg_expr {
                        A::Min { input, .. }
                        | A::Max { input, .. }
                        | A::Median(input)
                        | A::NUnique(input)
                        | A::First(input)
                        | A::Last(input)
                        | A::Item(input)
                        | A::Mean(input)
                        | A::Implode(input)
                        | A::Count { input, .. }
                        | A::Sum(input)
                        | A::AggGroups(input)
                        | A::Std(input, _)
                        | A::Var(input, _) => input,
                        A::Quantile { expr, quantile, .. } => {
                            *quantile = Arc::new(rec(quantile, with.clone())?);
                            expr
                        },
                    };
                    *input = Arc::new(rec(input, with)?);
                    E::Agg(agg_expr)
                },
                E::Ternary {
                    predicate,
                    truthy,
                    falsy,
                } => E::Ternary {
                    predicate: Arc::new(rec(&predicate, with.clone())?),
                    truthy: Arc::new(rec(&truthy, with.clone())?),
                    falsy: Arc::new(rec(&falsy, with)?),
                },
                E::Function { input, function } => E::Function {
                    input: input
                        .iter()
                        .map(|i| rec(i, with.clone()))
                        .collect::<PolarsResult<Vec<_>>>()?,
                    function,
                },
                E::Explode { input, skip_empty } => E::Explode {
                    input: Arc::new(rec(&input, with)?),
                    skip_empty,
                },
                E::Filter { input, by } => E::Filter {
                    input: Arc::new(rec(&input, with.clone())?),
                    by: Arc::new(rec(&by, with)?),
                },
                E::Window {
                    function,
                    partition_by,
                    order_by,
                    options,
                } => E::Window {
                    function: Arc::new(rec(&function, with.clone())?),
                    partition_by: partition_by
                        .iter()
                        .map(|i| rec(i, with.clone()))
                        .collect::<PolarsResult<Vec<_>>>()?,
                    order_by: order_by
                        .map(|(e, opt)| PolarsResult::Ok((Arc::new(rec(&e, with)?), opt)))
                        .transpose()?,
                    options,
                },
                E::Slice {
                    input,
                    offset,
                    length,
                } => E::Slice {
                    input: Arc::new(rec(&input, with.clone())?),
                    offset: Arc::new(rec(&offset, with.clone())?),
                    length: Arc::new(rec(&length, with)?),
                },
                E::KeepName(expr) => E::KeepName(Arc::new(rec(&expr, with)?)),
                E::AnonymousFunction {
                    input,
                    function,
                    options,
                    fmt_str,
                } => E::AnonymousFunction {
                    input: input
                        .iter()
                        .map(|i| rec(i, with.clone()))
                        .collect::<PolarsResult<Vec<_>>>()?,
                    function,
                    options,
                    fmt_str,
                },
                E::Eval {
                    expr,
                    evaluation,
                    variant,
                } => E::Eval {
                    expr: Arc::new(rec(&expr, with)?),
                    evaluation,
                    variant,
                },
                E::RenameAlias { function, expr } => E::RenameAlias {
                    function,
                    expr: Arc::new(rec(&expr, with)?),
                },
            })
        }
        Ok(rec(&self.inner, expr.inner)
            .map_err(PyPolarsErr::from)?
            .into())
    }
}
