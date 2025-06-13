use super::*;
use crate::plans::conversion::functions::convert_functions;

pub fn to_expr_ir(expr: Expr, arena: &mut Arena<AExpr>, schema: &Schema) -> PolarsResult<ExprIR> {
    let (node, output_name) = to_aexpr_impl(expr, arena, schema)?;
    Ok(ExprIR::new(node, OutputName::Alias(output_name)))
}

pub fn to_expr_ir_materialized_lit(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<ExprIR> {
    let (node, output_name) = to_aexpr_impl_materialized_lit(expr, arena, schema)?;
    Ok(ExprIR::new(node, OutputName::Alias(output_name)))
}

pub(super) fn to_expr_irs(
    input: Vec<Expr>,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<Vec<ExprIR>> {
    input
        .into_iter()
        .map(|e| to_expr_ir(e, arena, schema))
        .collect()
}

fn to_aexpr_impl_materialized_lit(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<(Node, PlSmallStr)> {
    // Already convert `Lit Float and Lit Int` expressions that are not used in a binary / function expression.
    // This means they can be materialized immediately
    let e = match expr {
        Expr::Literal(lv @ LiteralValue::Dyn(_)) => Expr::Literal(lv.materialize()),
        Expr::Alias(inner, name) if matches!(&*inner, Expr::Literal(LiteralValue::Dyn(_))) => {
            let Expr::Literal(lv) = &*inner else {
                unreachable!()
            };
            Expr::Alias(Arc::new(Expr::Literal(lv.clone().materialize())), name)
        },
        e => e,
    };
    to_aexpr_impl(e, arena, schema)
}

/// Converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation.
#[recursive]
pub(super) fn to_aexpr_impl(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<(Node, PlSmallStr)> {
    let owned = Arc::unwrap_or_clone;
    let (v, output_name) = match expr {
        Expr::Explode { input, skip_empty } => {
            let (expr, output_name) = to_aexpr_impl(owned(input), arena, schema)?;
            (AExpr::Explode { expr, skip_empty }, output_name)
        },
        Expr::Alias(e, name) => return Ok((to_aexpr_impl(owned(e), arena, schema)?.0, name)),
        Expr::Literal(lv) => {
            let output_name = lv.output_column_name().clone();
            (AExpr::Literal(lv), output_name)
        },
        Expr::Column(name) => (AExpr::Column(name.clone()), name),
        Expr::BinaryExpr { left, op, right } => {
            let (l, output_name) = to_aexpr_impl(owned(left), arena, schema)?;
            let (r, _) = to_aexpr_impl(owned(right), arena, schema)?;
            (
                AExpr::BinaryExpr {
                    left: l,
                    op,
                    right: r,
                },
                output_name,
            )
        },
        Expr::Cast {
            expr,
            dtype,
            options,
        } => {
            let (expr, output_name) = to_aexpr_impl(owned(expr), arena, schema)?;
            (
                AExpr::Cast {
                    expr,
                    dtype: dtype.into_datatype(schema)?,
                    options,
                },
                output_name,
            )
        },
        Expr::Gather {
            expr,
            idx,
            returns_scalar,
        } => {
            let (expr, output_name) = to_aexpr_impl(owned(expr), arena, schema)?;
            let (idx, _) = to_aexpr_impl_materialized_lit(owned(idx), arena, schema)?;
            (
                AExpr::Gather {
                    expr,
                    idx,
                    returns_scalar,
                },
                output_name,
            )
        },
        Expr::Sort { expr, options } => {
            let (expr, output_name) = to_aexpr_impl(owned(expr), arena, schema)?;
            (AExpr::Sort { expr, options }, output_name)
        },
        Expr::SortBy {
            expr,
            by,
            sort_options,
        } => {
            let (expr, output_name) = to_aexpr_impl(owned(expr), arena, schema)?;
            let by = by
                .into_iter()
                .map(|e| Ok(to_aexpr_impl(e, arena, schema)?.0))
                .collect::<PolarsResult<_>>()?;

            (
                AExpr::SortBy {
                    expr,
                    by,
                    sort_options,
                },
                output_name,
            )
        },
        Expr::Filter { input, by } => {
            let (input, output_name) = to_aexpr_impl(owned(input), arena, schema)?;
            let (by, _) = to_aexpr_impl(owned(by), arena, schema)?;
            (AExpr::Filter { input, by }, output_name)
        },
        Expr::Agg(agg) => {
            let (a_agg, output_name) = match agg {
                AggExpr::Min {
                    input,
                    propagate_nans,
                } => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (
                        IRAggExpr::Min {
                            input,
                            propagate_nans,
                        },
                        output_name,
                    )
                },
                AggExpr::Max {
                    input,
                    propagate_nans,
                } => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (
                        IRAggExpr::Max {
                            input,
                            propagate_nans,
                        },
                        output_name,
                    )
                },
                AggExpr::Median(input) => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (IRAggExpr::Median(input), output_name)
                },
                AggExpr::NUnique(input) => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (IRAggExpr::NUnique(input), output_name)
                },
                AggExpr::First(input) => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (IRAggExpr::First(input), output_name)
                },
                AggExpr::Last(input) => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (IRAggExpr::Last(input), output_name)
                },
                AggExpr::Mean(input) => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (IRAggExpr::Mean(input), output_name)
                },
                AggExpr::Implode(input) => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (IRAggExpr::Implode(input), output_name)
                },
                AggExpr::Count(input, include_nulls) => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (IRAggExpr::Count(input, include_nulls), output_name)
                },
                AggExpr::Quantile {
                    expr,
                    quantile,
                    method,
                } => {
                    let (expr, output_name) =
                        to_aexpr_impl_materialized_lit(owned(expr), arena, schema)?;
                    let (quantile, _) =
                        to_aexpr_impl_materialized_lit(owned(quantile), arena, schema)?;
                    (
                        IRAggExpr::Quantile {
                            expr,
                            quantile,
                            method,
                        },
                        output_name,
                    )
                },
                AggExpr::Sum(input) => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (IRAggExpr::Sum(input), output_name)
                },
                AggExpr::Std(input, ddof) => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (IRAggExpr::Std(input, ddof), output_name)
                },
                AggExpr::Var(input, ddof) => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (IRAggExpr::Var(input, ddof), output_name)
                },
                AggExpr::AggGroups(input) => {
                    let (input, output_name) =
                        to_aexpr_impl_materialized_lit(owned(input), arena, schema)?;
                    (IRAggExpr::AggGroups(input), output_name)
                },
            };
            (AExpr::Agg(a_agg), output_name)
        },
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let (p, _) = to_aexpr_impl_materialized_lit(owned(predicate), arena, schema)?;
            let (t, output_name) = to_aexpr_impl(owned(truthy), arena, schema)?;
            let (f, _) = to_aexpr_impl(owned(falsy), arena, schema)?;
            (
                AExpr::Ternary {
                    predicate: p,
                    truthy: t,
                    falsy: f,
                },
                output_name,
            )
        },
        Expr::AnonymousFunction {
            input,
            function,
            output_type,
            options,
            fmt_str,
        } => {
            let e = to_expr_irs(input, arena, schema)?;
            let output_name = if e.is_empty() {
                fmt_str.as_ref().clone()
            } else {
                e[0].output_name().clone()
            };

            let function = function.materialize()?;
            let output_type = output_type.materialize()?;
            function.as_ref().resolve_dsl(schema)?;
            output_type.as_ref().resolve_dsl(schema)?;

            (
                AExpr::AnonymousFunction {
                    input: e,
                    function: LazySerde::Deserialized(function),
                    output_type: LazySerde::Deserialized(output_type),
                    options,
                    fmt_str,
                },
                output_name,
            )
        },
        Expr::Function { input, function } => {
            return convert_functions(input, function, arena, schema);
        },
        Expr::Window {
            function,
            partition_by,
            order_by,
            options,
        } => {
            let (function, output_name) = to_aexpr_impl(owned(function), arena, schema)?;
            let order_by = if let Some((e, options)) = order_by {
                Some((to_aexpr_impl(owned(e.clone()), arena, schema)?.0, options))
            } else {
                None
            };

            (
                AExpr::Window {
                    function,
                    partition_by: partition_by
                        .into_iter()
                        .map(|e| Ok(to_aexpr_impl_materialized_lit(e, arena, schema)?.0))
                        .collect::<PolarsResult<_>>()?,
                    order_by,
                    options,
                },
                output_name,
            )
        },
        Expr::Slice {
            input,
            offset,
            length,
        } => {
            let (input, output_name) = to_aexpr_impl(owned(input), arena, schema)?;
            let (offset, _) = to_aexpr_impl_materialized_lit(owned(offset), arena, schema)?;
            let (length, _) = to_aexpr_impl_materialized_lit(owned(length), arena, schema)?;
            (
                AExpr::Slice {
                    input,
                    offset,
                    length,
                },
                output_name,
            )
        },
        Expr::Eval {
            expr,
            evaluation,
            variant,
        } => {
            let (expr, output_name) = to_aexpr_impl(owned(expr), arena, schema)?;
            let expr_dtype = arena.get(expr).to_dtype(schema, Context::Default, arena)?;
            let element_dtype = variant.element_dtype(&expr_dtype)?;
            let evaluation_schema = Schema::from_iter([(PlSmallStr::EMPTY, element_dtype.clone())]);
            let (evaluation, _) = to_aexpr_impl(owned(evaluation), arena, &evaluation_schema)?;

            match variant {
                EvalVariant::List => {
                    for (_, e) in ArenaExprIter::iter(&&*arena, evaluation) {
                        match e {
                            #[cfg(feature = "dtype-categorical")]
                            AExpr::Cast {
                                dtype: DataType::Categorical(_, _) | DataType::Enum(_, _),
                                ..
                            } => {
                                polars_bail!(
                                    ComputeError: "casting to categorical not allowed in `list.eval`"
                                )
                            },
                            AExpr::Column(name) => {
                                polars_ensure!(
                                    name.is_empty(),
                                    ComputeError:
                                    "named columns are not allowed in `list.eval`; consider using `element` or `col(\"\")`"
                                );
                            },
                            _ => {},
                        }
                    }
                },
                EvalVariant::Cumulative { .. } => {
                    polars_ensure!(
                        is_scalar_ae(evaluation, arena),
                        InvalidOperation: "`cumulative_eval` is not allowed with non-scalar output"
                    )
                },
            }

            (
                AExpr::Eval {
                    expr,
                    evaluation,
                    variant,
                },
                output_name,
            )
        },
        Expr::Len => (AExpr::Len, get_len_name()),
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
    Ok((arena.add(v), output_name))
}
