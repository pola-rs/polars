use super::*;
use crate::plans::conversion::functions::convert_functions;

pub fn to_expr_ir(expr: Expr, arena: &mut Arena<AExpr>, schema: &Schema) -> PolarsResult<ExprIR> {
    let mut state = ConversionContext::new();
    let node = to_aexpr_impl(expr, arena, &mut state, schema)?;
    Ok(ExprIR::new(node, state.output_name))
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

pub fn to_expr_ir_ignore_alias(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<ExprIR> {
    let mut state = ConversionContext::new();
    state.ignore_alias = true;
    let node = to_aexpr_impl_materialized_lit(expr, arena, &mut state, schema)?;
    Ok(ExprIR::new(node, state.output_name))
}

pub(super) fn to_expr_irs_ignore_alias(
    input: Vec<Expr>,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<Vec<ExprIR>> {
    input
        .into_iter()
        .map(|e| to_expr_ir_ignore_alias(e, arena, schema))
        .collect()
}

/// converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation
pub fn to_aexpr(expr: Expr, arena: &mut Arena<AExpr>, schema: &Schema) -> PolarsResult<Node> {
    to_aexpr_impl_materialized_lit(
        expr,
        arena,
        &mut ConversionContext {
            prune_alias: false,
            ..Default::default()
        },
        schema,
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
    schema: &Schema,
) -> PolarsResult<Vec<Node>> {
    input
        .into_iter()
        .map(|e| to_aexpr_impl_materialized_lit(e, arena, state, schema))
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
    schema: &Schema,
) -> PolarsResult<Node> {
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
    to_aexpr_impl(e, arena, state, schema)
}

/// Converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation.
#[recursive]
pub(super) fn to_aexpr_impl(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    state: &mut ConversionContext,
    schema: &Schema,
) -> PolarsResult<Node> {
    let owned = Arc::unwrap_or_clone;
    let v = match expr {
        Expr::Explode { input, skip_empty } => AExpr::Explode {
            expr: to_aexpr_impl(owned(input), arena, state, schema)?,
            skip_empty,
        },
        Expr::Alias(e, name) => {
            if state.prune_alias {
                if state.output_name.is_none() && !state.ignore_alias {
                    state.output_name = OutputName::Alias(name);
                }
                let _ = to_aexpr_impl(owned(e), arena, state, schema)?;
                arena.pop().unwrap()
            } else {
                AExpr::Alias(to_aexpr_impl(owned(e), arena, state, schema)?, name)
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
            let l = to_aexpr_impl(owned(left), arena, state, schema)?;
            let r = to_aexpr_impl(owned(right), arena, state, schema)?;
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
            expr: to_aexpr_impl(owned(expr), arena, state, schema)?,
            dtype: dtype.into_datatype(schema)?,
            options,
        },
        Expr::Gather {
            expr,
            idx,
            returns_scalar,
        } => AExpr::Gather {
            expr: to_aexpr_impl(owned(expr), arena, state, schema)?,
            idx: to_aexpr_impl_materialized_lit(owned(idx), arena, state, schema)?,
            returns_scalar,
        },
        Expr::Sort { expr, options } => AExpr::Sort {
            expr: to_aexpr_impl(owned(expr), arena, state, schema)?,
            options,
        },
        Expr::SortBy {
            expr,
            by,
            sort_options,
        } => AExpr::SortBy {
            expr: to_aexpr_impl(owned(expr), arena, state, schema)?,
            by: by
                .into_iter()
                .map(|e| to_aexpr_impl(e, arena, state, schema))
                .collect::<PolarsResult<_>>()?,
            sort_options,
        },
        Expr::Filter { input, by } => AExpr::Filter {
            input: to_aexpr_impl(owned(input), arena, state, schema)?,
            by: to_aexpr_impl(owned(by), arena, state, schema)?,
        },
        Expr::Agg(agg) => {
            let a_agg =
                match agg {
                    AggExpr::Min {
                        input,
                        propagate_nans,
                    } => IRAggExpr::Min {
                        input: to_aexpr_impl_materialized_lit(owned(input), arena, state, schema)?,
                        propagate_nans,
                    },
                    AggExpr::Max {
                        input,
                        propagate_nans,
                    } => IRAggExpr::Max {
                        input: to_aexpr_impl_materialized_lit(owned(input), arena, state, schema)?,
                        propagate_nans,
                    },
                    AggExpr::Median(expr) => IRAggExpr::Median(to_aexpr_impl_materialized_lit(
                        owned(expr),
                        arena,
                        state,
                        schema,
                    )?),
                    AggExpr::NUnique(expr) => IRAggExpr::NUnique(to_aexpr_impl_materialized_lit(
                        owned(expr),
                        arena,
                        state,
                        schema,
                    )?),
                    AggExpr::First(expr) => IRAggExpr::First(to_aexpr_impl_materialized_lit(
                        owned(expr),
                        arena,
                        state,
                        schema,
                    )?),
                    AggExpr::Last(expr) => IRAggExpr::Last(to_aexpr_impl_materialized_lit(
                        owned(expr),
                        arena,
                        state,
                        schema,
                    )?),
                    AggExpr::Mean(expr) => IRAggExpr::Mean(to_aexpr_impl_materialized_lit(
                        owned(expr),
                        arena,
                        state,
                        schema,
                    )?),
                    AggExpr::Implode(expr) => IRAggExpr::Implode(to_aexpr_impl_materialized_lit(
                        owned(expr),
                        arena,
                        state,
                        schema,
                    )?),
                    AggExpr::Count(expr, include_nulls) => IRAggExpr::Count(
                        to_aexpr_impl_materialized_lit(owned(expr), arena, state, schema)?,
                        include_nulls,
                    ),
                    AggExpr::Quantile {
                        expr,
                        quantile,
                        method,
                    } => IRAggExpr::Quantile {
                        expr: to_aexpr_impl_materialized_lit(owned(expr), arena, state, schema)?,
                        quantile: to_aexpr_impl_materialized_lit(
                            owned(quantile),
                            arena,
                            state,
                            schema,
                        )?,
                        method,
                    },
                    AggExpr::Sum(expr) => IRAggExpr::Sum(to_aexpr_impl_materialized_lit(
                        owned(expr),
                        arena,
                        state,
                        schema,
                    )?),
                    AggExpr::Std(expr, ddof) => IRAggExpr::Std(
                        to_aexpr_impl_materialized_lit(owned(expr), arena, state, schema)?,
                        ddof,
                    ),
                    AggExpr::Var(expr, ddof) => IRAggExpr::Var(
                        to_aexpr_impl_materialized_lit(owned(expr), arena, state, schema)?,
                        ddof,
                    ),
                    AggExpr::AggGroups(expr) => IRAggExpr::AggGroups(
                        to_aexpr_impl_materialized_lit(owned(expr), arena, state, schema)?,
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
            let t = to_aexpr_impl(owned(truthy), arena, state, schema)?;
            let p = to_aexpr_impl_materialized_lit(owned(predicate), arena, state, schema)?;
            let f = to_aexpr_impl(owned(falsy), arena, state, schema)?;
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
            let e = to_expr_irs(input, arena, schema)?;
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
        } => return convert_functions(input, function, options, arena, state, schema),
        Expr::Window {
            function,
            partition_by,
            order_by,
            options,
        } => {
            // Process function first so name is correct.
            let function = to_aexpr_impl(owned(function), arena, state, schema)?;
            let order_by = if let Some((e, options)) = order_by {
                Some((
                    to_aexpr_impl(owned(e.clone()), arena, state, schema)?,
                    options,
                ))
            } else {
                None
            };

            AExpr::Window {
                function,
                partition_by: to_aexprs(partition_by, arena, state, schema)?,
                order_by,
                options,
            }
        },
        Expr::Slice {
            input,
            offset,
            length,
        } => AExpr::Slice {
            input: to_aexpr_impl(owned(input), arena, state, schema)?,
            offset: to_aexpr_impl_materialized_lit(owned(offset), arena, state, schema)?,
            length: to_aexpr_impl_materialized_lit(owned(length), arena, state, schema)?,
        },
        Expr::Eval {
            expr,
            evaluation,
            variant,
        } => {
            let mut evaluation_state = ConversionContext {
                output_name: OutputName::None,
                prune_alias: true,
                ignore_alias: true,
            };

            let expr = to_aexpr_impl(owned(expr), arena, state, schema)?;
            let evaluation =
                to_aexpr_impl(owned(evaluation), arena, &mut evaluation_state, schema)?;

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

            AExpr::Eval {
                expr,
                evaluation,
                variant,
            }
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
