use super::*;
use crate::plans::conversion::functions::convert_functions;

pub fn to_expr_ir(expr: Expr, arena: &mut Arena<AExpr>) -> PolarsResult<ExprIR> {
    let mut state = ConversionContext::new();
    let node = to_aexpr_impl(expr, arena, &mut state)?;
    Ok(ExprIR::new(node, state.output_name))
}

pub(super) fn to_expr_irs(input: Vec<Expr>, arena: &mut Arena<AExpr>) -> PolarsResult<Vec<ExprIR>> {
    input.into_iter().map(|e| to_expr_ir(e, arena)).collect()
}

pub fn to_expr_ir_ignore_alias(expr: Expr, arena: &mut Arena<AExpr>) -> PolarsResult<ExprIR> {
    let mut state = ConversionContext::new();
    state.ignore_alias = true;
    let node = to_aexpr_impl_materialized_lit(expr, arena, &mut state)?;
    Ok(ExprIR::new(node, state.output_name))
}

pub(super) fn to_expr_irs_ignore_alias(
    input: Vec<Expr>,
    arena: &mut Arena<AExpr>,
) -> PolarsResult<Vec<ExprIR>> {
    input
        .into_iter()
        .map(|e| to_expr_ir_ignore_alias(e, arena))
        .collect()
}

/// converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation
pub fn to_aexpr(expr: Expr, arena: &mut Arena<AExpr>) -> PolarsResult<Node> {
    to_aexpr_impl_materialized_lit(
        expr,
        arena,
        &mut ConversionContext {
            prune_alias: false,
            ..Default::default()
        },
    )
}

#[derive(Default)]
pub(super) struct ConversionContext {
    pub(super) output_name: OutputName,
    /// Remove alias from the expressions and set as [`OutputName`].
    pub(super) prune_alias: bool,
    /// If an `alias` is encountered prune and ignore it.
    pub(super) ignore_alias: bool,
}

impl ConversionContext {
    fn new() -> Self {
        Self {
            prune_alias: true,
            ..Default::default()
        }
    }
}

fn to_aexprs(
    input: Vec<Expr>,
    arena: &mut Arena<AExpr>,
    state: &mut ConversionContext,
) -> PolarsResult<Vec<Node>> {
    input
        .into_iter()
        .map(|e| to_aexpr_impl_materialized_lit(e, arena, state))
        .collect()
}

pub(super) fn set_function_output_name<F>(
    e: &[ExprIR],
    state: &mut ConversionContext,
    function_fmt: F,
) where
    F: FnOnce() -> PlSmallStr,
{
    if state.output_name.is_none() {
        if e.is_empty() {
            let s = function_fmt();
            state.output_name = OutputName::LiteralLhs(s);
        } else {
            state.output_name = e[0].output_name_inner().clone();
        }
    }
}

fn to_aexpr_impl_materialized_lit(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    state: &mut ConversionContext,
) -> PolarsResult<Node> {
    // Already convert `Lit Float and Lit Int` expressions that are not used in a binary / function expression.
    // This means they can be materialized immediately
    let e = match expr {
        Expr::Literal(lv @ LiteralValue::Int(_) | lv @ LiteralValue::Float(_)) => {
            let av = lv.to_any_value().unwrap();
            Expr::Literal(LiteralValue::try_from(av).unwrap())
        },
        Expr::Alias(inner, name)
            if matches!(
                &*inner,
                Expr::Literal(LiteralValue::Int(_) | LiteralValue::Float(_))
            ) =>
        {
            let Expr::Literal(lv @ LiteralValue::Int(_) | lv @ LiteralValue::Float(_)) = &*inner
            else {
                unreachable!()
            };
            let av = lv.to_any_value().unwrap();
            Expr::Alias(
                Arc::new(Expr::Literal(LiteralValue::try_from(av).unwrap())),
                name,
            )
        },
        e => e,
    };
    to_aexpr_impl(e, arena, state)
}

/// Converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation.
#[recursive]
pub(super) fn to_aexpr_impl(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    state: &mut ConversionContext,
) -> PolarsResult<Node> {
    let owned = Arc::unwrap_or_clone;
    let v = match expr {
        Expr::Explode(expr) => AExpr::Explode(to_aexpr_impl(owned(expr), arena, state)?),
        Expr::Alias(e, name) => {
            if state.prune_alias {
                if state.output_name.is_none() && !state.ignore_alias {
                    state.output_name = OutputName::Alias(name);
                }
                let _ = to_aexpr_impl(owned(e), arena, state)?;
                arena.pop().unwrap()
            } else {
                AExpr::Alias(to_aexpr_impl(owned(e), arena, state)?, name)
            }
        },
        Expr::Literal(lv) => {
            if state.output_name.is_none() {
                state.output_name = OutputName::LiteralLhs(lv.output_column_name().clone());
            }
            AExpr::Literal(lv)
        },
        Expr::Column(name) => {
            if state.output_name.is_none() {
                state.output_name = OutputName::ColumnLhs(name.clone())
            }
            AExpr::Column(name)
        },
        Expr::BinaryExpr { left, op, right } => {
            let l = to_aexpr_impl(owned(left), arena, state)?;
            let r = to_aexpr_impl(owned(right), arena, state)?;
            AExpr::BinaryExpr {
                left: l,
                op,
                right: r,
            }
        },
        Expr::Cast {
            expr,
            dtype,
            options,
        } => AExpr::Cast {
            expr: to_aexpr_impl(owned(expr), arena, state)?,
            dtype,
            options,
        },
        Expr::Gather {
            expr,
            idx,
            returns_scalar,
        } => AExpr::Gather {
            expr: to_aexpr_impl(owned(expr), arena, state)?,
            idx: to_aexpr_impl_materialized_lit(owned(idx), arena, state)?,
            returns_scalar,
        },
        Expr::Sort { expr, options } => AExpr::Sort {
            expr: to_aexpr_impl(owned(expr), arena, state)?,
            options,
        },
        Expr::SortBy {
            expr,
            by,
            sort_options,
        } => AExpr::SortBy {
            expr: to_aexpr_impl(owned(expr), arena, state)?,
            by: by
                .into_iter()
                .map(|e| to_aexpr_impl(e, arena, state))
                .collect::<PolarsResult<_>>()?,
            sort_options,
        },
        Expr::Filter { input, by } => AExpr::Filter {
            input: to_aexpr_impl(owned(input), arena, state)?,
            by: to_aexpr_impl(owned(by), arena, state)?,
        },
        Expr::Agg(agg) => {
            let a_agg = match agg {
                AggExpr::Min {
                    input,
                    propagate_nans,
                } => IRAggExpr::Min {
                    input: to_aexpr_impl_materialized_lit(owned(input), arena, state)?,
                    propagate_nans,
                },
                AggExpr::Max {
                    input,
                    propagate_nans,
                } => IRAggExpr::Max {
                    input: to_aexpr_impl_materialized_lit(owned(input), arena, state)?,
                    propagate_nans,
                },
                AggExpr::Median(expr) => {
                    IRAggExpr::Median(to_aexpr_impl_materialized_lit(owned(expr), arena, state)?)
                },
                AggExpr::NUnique(expr) => {
                    IRAggExpr::NUnique(to_aexpr_impl_materialized_lit(owned(expr), arena, state)?)
                },
                AggExpr::First(expr) => {
                    IRAggExpr::First(to_aexpr_impl_materialized_lit(owned(expr), arena, state)?)
                },
                AggExpr::Last(expr) => {
                    IRAggExpr::Last(to_aexpr_impl_materialized_lit(owned(expr), arena, state)?)
                },
                AggExpr::Mean(expr) => {
                    IRAggExpr::Mean(to_aexpr_impl_materialized_lit(owned(expr), arena, state)?)
                },
                AggExpr::Implode(expr) => {
                    IRAggExpr::Implode(to_aexpr_impl_materialized_lit(owned(expr), arena, state)?)
                },
                AggExpr::Count(expr, include_nulls) => IRAggExpr::Count(
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?,
                    include_nulls,
                ),
                AggExpr::Quantile {
                    expr,
                    quantile,
                    interpol,
                } => IRAggExpr::Quantile {
                    expr: to_aexpr_impl_materialized_lit(owned(expr), arena, state)?,
                    quantile: to_aexpr_impl_materialized_lit(owned(quantile), arena, state)?,
                    interpol,
                },
                AggExpr::Sum(expr) => {
                    IRAggExpr::Sum(to_aexpr_impl_materialized_lit(owned(expr), arena, state)?)
                },
                AggExpr::Std(expr, ddof) => IRAggExpr::Std(
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?,
                    ddof,
                ),
                AggExpr::Var(expr, ddof) => IRAggExpr::Var(
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?,
                    ddof,
                ),
                AggExpr::AggGroups(expr) => {
                    IRAggExpr::AggGroups(to_aexpr_impl_materialized_lit(owned(expr), arena, state)?)
                },
                #[cfg(feature = "bitwise")]
                AggExpr::Bitwise(expr, f) => IRAggExpr::Bitwise(
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?,
                    f,
                ),
            };
            AExpr::Agg(a_agg)
        },
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            // Truthy must be resolved first to get the lhs name first set.
            let t = to_aexpr_impl(owned(truthy), arena, state)?;
            let p = to_aexpr_impl_materialized_lit(owned(predicate), arena, state)?;
            let f = to_aexpr_impl(owned(falsy), arena, state)?;
            AExpr::Ternary {
                predicate: p,
                truthy: t,
                falsy: f,
            }
        },
        Expr::AnonymousFunction {
            input,
            function,
            output_type,
            options,
        } => {
            let e = to_expr_irs(input, arena)?;
            set_function_output_name(&e, state, || PlSmallStr::from_static(options.fmt_str));
            AExpr::AnonymousFunction {
                input: e,
                function,
                output_type,
                options,
            }
        },
        Expr::Function {
            input,
            function,
            options,
        } => return convert_functions(input, function, options, arena, state),
        Expr::Window {
            function,
            partition_by,
            order_by,
            options,
        } => {
            // Process function first so name is correct.
            let function = to_aexpr_impl(owned(function), arena, state)?;
            let order_by = if let Some((e, options)) = order_by {
                Some((to_aexpr_impl(owned(e.clone()), arena, state)?, options))
            } else {
                None
            };

            AExpr::Window {
                function,
                partition_by: to_aexprs(partition_by, arena, state)?,
                order_by,
                options,
            }
        },
        Expr::Slice {
            input,
            offset,
            length,
        } => AExpr::Slice {
            input: to_aexpr_impl(owned(input), arena, state)?,
            offset: to_aexpr_impl_materialized_lit(owned(offset), arena, state)?,
            length: to_aexpr_impl_materialized_lit(owned(length), arena, state)?,
        },
        Expr::Len => {
            if state.output_name.is_none() {
                state.output_name = OutputName::LiteralLhs(get_len_name())
            }
            AExpr::Len
        },
        #[cfg(feature = "dtype-struct")]
        e @ Expr::Field(_) => {
            polars_bail!(InvalidOperation: "'Expr: {}' not allowed in this context/location", e)
        },
        e @ Expr::IndexColumn(_)
        | e @ Expr::Wildcard
        | e @ Expr::Nth(_)
        | e @ Expr::SubPlan { .. }
        | e @ Expr::KeepName(_)
        | e @ Expr::Exclude(_, _)
        | e @ Expr::RenameAlias { .. }
        | e @ Expr::Columns { .. }
        | e @ Expr::DtypeColumn { .. }
        | e @ Expr::Selector(_) => {
            polars_bail!(InvalidOperation: "'Expr: {}' not allowed in this context/location", e)
        },
    };
    Ok(arena.add(v))
}
