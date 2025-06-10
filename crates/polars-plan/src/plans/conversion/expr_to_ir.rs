use super::*;
use crate::plans::conversion::functions::convert_functions;

pub(super) enum ExpandedDslToIrExpr {
    Simple(DslToIrExpr),
    Expanded(Vec<DslToIrExpr>),
}

pub(super) struct DslToIrExpr {
    pub node: Node,
    pub output_name: PlSmallStr,
}

impl DslToIrExpr {
    fn with_output_name(mut self, output_name: PlSmallStr) -> Self {
        self.output_name = output_name;
        self
    }

    pub fn map_expr(
        mut self,
        expr_arena: &mut Arena<AExpr>,
        f: impl FnOnce(Node) -> AExpr,
    ) -> Self {
        self.node = expr_arena.add(f(self.node));
        self
    }

    fn into_expr_ir(self) -> ExprIR {
        ExprIR::new(self.node, OutputName::Alias(self.output_name))
    }
}

impl ExpandedDslToIrExpr {
    fn with_output_name(self, output_name: PlSmallStr) -> Self {
        match self {
            Self::Simple(e) => Self::Simple(e.with_output_name(output_name)),
            Self::Expanded(e) => Self::Expanded(
                e.into_iter()
                    .map(|e| e.with_output_name(output_name.clone()))
                    .collect(),
            ),
        }
    }

    pub fn map_expr(self, expr_arena: &mut Arena<AExpr>, f: impl Fn(Node) -> AExpr + Copy) -> Self {
        match self {
            Self::Simple(e) => Self::Simple(e.map_expr(expr_arena, f)),
            Self::Expanded(e) => {
                Self::Expanded(e.into_iter().map(|e| e.map_expr(expr_arena, f)).collect())
            },
        }
    }

    pub fn combine_expr(
        self,
        expr_arena: &mut Arena<AExpr>,
        other: ExpandedDslToIrExpr,
        f: impl Fn(Node, Node) -> AExpr + Copy,
    ) -> PolarsResult<Self> {
        match (self, other) {
            (Self::Simple(l), Self::Simple(r)) => Ok(Self::Simple(DslToIrExpr {
                node: expr_arena.add(f(l.node, r.node)),
                output_name: l.output_name,
            })),
            (Self::Simple(l), Self::Expanded(r)) => Ok(Self::Expanded(
                r.into_iter()
                    .map(|r| DslToIrExpr {
                        node: expr_arena.add(f(l.node, r.node)),
                        output_name: l.output_name.clone(),
                    })
                    .collect(),
            )),
            (Self::Expanded(l), Self::Simple(r)) => Ok(Self::Expanded(
                l.into_iter()
                    .map(|l| DslToIrExpr {
                        node: expr_arena.add(f(l.node, r.node)),
                        output_name: l.output_name,
                    })
                    .collect(),
            )),
            (Self::Expanded(l), Self::Expanded(r)) => {
                polars_ensure!(l.len() == r.len(), InvalidOperation: "cannot combine selector-based expressions that expanded to a different numbers of columns");
                Ok(Self::Expanded(
                    l.into_iter()
                        .zip(r.into_iter())
                        .map(|(l, r)| DslToIrExpr {
                            node: expr_arena.add(f(l.node, r.node)),
                            output_name: l.output_name,
                        })
                        .collect(),
                ))
            },
        }
    }
}

pub fn to_expr_ir_no_selectors(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<Option<ExprIR>> {
    Ok(match to_aexpr_impl(expr, arena, schema)? {
        ExpandedDslToIrExpr::Simple(e) => Some(e.into_expr_ir()),
        ExpandedDslToIrExpr::Expanded(_) => None,
    })
}

pub fn to_expr_ir_materialized_lit_no_selectors(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<Option<ExprIR>> {
    Ok(match to_aexpr_impl_materialized_lit(expr, arena, schema)? {
        ExpandedDslToIrExpr::Simple(e) => Some(e.into_expr_ir()),
        ExpandedDslToIrExpr::Expanded(_) => None,
    })
}

pub fn extend_expr_ir(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
    expr_irs: &mut Vec<ExprIR>,
) -> PolarsResult<()> {
    match to_aexpr_impl(expr, arena, schema)? {
        ExpandedDslToIrExpr::Simple(e) => expr_irs.push(e.into_expr_ir()),
        ExpandedDslToIrExpr::Expanded(e) => {
            expr_irs.extend(e.into_iter().map(DslToIrExpr::into_expr_ir))
        },
    }
    Ok(())
}

pub fn extend_expr_ir_materialized_lit(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
    expr_irs: &mut Vec<ExprIR>,
) -> PolarsResult<()> {
    match to_aexpr_impl_materialized_lit(expr, arena, schema)? {
        ExpandedDslToIrExpr::Simple(e) => expr_irs.push(e.into_expr_ir()),
        ExpandedDslToIrExpr::Expanded(e) => {
            expr_irs.extend(e.into_iter().map(DslToIrExpr::into_expr_ir))
        },
    }
    Ok(())
}

pub fn expand_into_expr_irs(
    input: Vec<Expr>,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<Vec<ExprIR>> {
    let mut expanded = Vec::with_capacity(input.len());
    for e in input {
        extend_expr_ir(e, arena, schema, &mut expanded)?;
    }
    Ok(expanded)
}

pub(super) fn to_expr_irs(
    input: Vec<Expr>,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<Vec<ExprIR>> {
    let mut expanded = Vec::with_capacity(input.len());
    for e in input {
        extend_expr_ir(e, arena, schema, &mut expanded)?;
    }
    Ok(expanded)
}

fn to_aexpr_impl_materialized_lit(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<ExpandedDslToIrExpr> {
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
) -> PolarsResult<ExpandedDslToIrExpr> {
    let owned = Arc::unwrap_or_clone;
    Ok(match expr {
        Expr::Explode { input, skip_empty } => to_aexpr_impl(owned(input), arena, schema)?
            .map_expr(arena, |expr| AExpr::Explode { expr, skip_empty }),
        Expr::Alias(e, name) => to_aexpr_impl(owned(e), arena, schema)?.with_output_name(name),
        Expr::Literal(lv) => {
            let output_name = lv.output_column_name().clone();
            ExpandedDslToIrExpr::Simple(DslToIrExpr {
                node: arena.add(AExpr::Literal(lv)),
                output_name,
            })
        },
        Expr::Column(name) => ExpandedDslToIrExpr::Simple(DslToIrExpr {
            node: arena.add(AExpr::Column(name.clone())),
            output_name: name,
        }),
        Expr::BinaryExpr { left, op, right } => {
            let left = to_aexpr_impl(owned(left), arena, schema)?;
            let right = to_aexpr_impl(owned(right), arena, schema)?;

            left.combine_expr(arena, right, |left, right| AExpr::BinaryExpr {
                left,
                op,
                right,
            })?
        },
        Expr::Cast {
            expr,
            dtype,
            options,
        } => {
            let dtype = dtype.into_datatype(schema)?;
            to_aexpr_impl(owned(expr), arena, schema)?.map_expr(arena, |expr| AExpr::Cast {
                expr,
                dtype: dtype.clone(),
                options,
            })
        },
        Expr::Gather {
            expr,
            idx,
            returns_scalar,
        } => {
            let expr = to_aexpr_impl(owned(expr), arena, schema)?;
            let idx = to_aexpr_impl_materialized_lit(owned(idx), arena, schema)?;
            expr.combine_expr(arena, idx, |expr, idx| AExpr::Gather {
                expr,
                idx,
                returns_scalar,
            })?
        },
        Expr::Sort { expr, options } => to_aexpr_impl(owned(expr), arena, schema)?
            .map_expr(arena, |expr| AExpr::Sort { expr, options }),
        Expr::SortBy {
            expr,
            by,
            sort_options,
        } => {
            todo!()
            // let expr = to_aexpr_impl(owned(expr), arena, schema)?;
            // let by = by
            //     .into_iter()
            //     .map(|e| Ok(to_aexpr_impl(e, arena, schema)?.node))
            //     .collect::<PolarsResult<_>>()?;
            // expr.map_expr(arena, |expr| AExpr::SortBy {
            //     expr,
            //     by,
            //     sort_options,
            // })
        },
        Expr::Filter { input, by } => {
            let input = to_aexpr_impl(owned(input), arena, schema)?;
            let by = to_aexpr_impl(owned(by), arena, schema)?;
            input.combine_expr(arena, by, |input, by| AExpr::Filter { input, by })?
        },
        Expr::Agg(agg) => match agg {
            AggExpr::Min {
                input,
                propagate_nans,
            } => to_aexpr_impl_materialized_lit(owned(input), arena, schema)?.map_expr(
                arena,
                |input| {
                    AExpr::Agg(IRAggExpr::Min {
                        input,
                        propagate_nans,
                    })
                },
            ),
            AggExpr::Max {
                input,
                propagate_nans,
            } => to_aexpr_impl_materialized_lit(owned(input), arena, schema)?.map_expr(
                arena,
                |input| {
                    AExpr::Agg(IRAggExpr::Max {
                        input,
                        propagate_nans,
                    })
                },
            ),
            AggExpr::Median(input) => to_aexpr_impl_materialized_lit(owned(input), arena, schema)?
                .map_expr(arena, |input| AExpr::Agg(IRAggExpr::Median(input))),
            AggExpr::NUnique(input) => to_aexpr_impl_materialized_lit(owned(input), arena, schema)?
                .map_expr(arena, |input| AExpr::Agg(IRAggExpr::NUnique(input))),
            AggExpr::First(input) => to_aexpr_impl_materialized_lit(owned(input), arena, schema)?
                .map_expr(arena, |input| AExpr::Agg(IRAggExpr::First(input))),
            AggExpr::Last(input) => to_aexpr_impl_materialized_lit(owned(input), arena, schema)?
                .map_expr(arena, |input| AExpr::Agg(IRAggExpr::Last(input))),
            AggExpr::Mean(input) => to_aexpr_impl_materialized_lit(owned(input), arena, schema)?
                .map_expr(arena, |input| AExpr::Agg(IRAggExpr::Mean(input))),
            AggExpr::Implode(input) => to_aexpr_impl_materialized_lit(owned(input), arena, schema)?
                .map_expr(arena, |input| AExpr::Agg(IRAggExpr::Implode(input))),
            AggExpr::Count(input, include_nulls) => {
                to_aexpr_impl_materialized_lit(owned(input), arena, schema)?
                    .map_expr(arena, |input| {
                        AExpr::Agg(IRAggExpr::Count(input, include_nulls))
                    })
            },
            AggExpr::Quantile {
                expr,
                quantile,
                method,
            } => {
                let expr = to_aexpr_impl_materialized_lit(owned(expr), arena, schema)?;
                let quantile = to_aexpr_impl_materialized_lit(owned(quantile), arena, schema)?;

                expr.combine_expr(arena, quantile, |expr, quantile| {
                    AExpr::Agg(IRAggExpr::Quantile {
                        expr,
                        quantile,
                        method,
                    })
                })?
            },
            AggExpr::Sum(input) => to_aexpr_impl_materialized_lit(owned(input), arena, schema)?
                .map_expr(arena, |input| AExpr::Agg(IRAggExpr::Sum(input))),
            AggExpr::Std(input, ddof) => {
                to_aexpr_impl_materialized_lit(owned(input), arena, schema)?
                    .map_expr(arena, |input| AExpr::Agg(IRAggExpr::Std(input, ddof)))
            },
            AggExpr::Var(input, ddof) => {
                to_aexpr_impl_materialized_lit(owned(input), arena, schema)?
                    .map_expr(arena, |input| AExpr::Agg(IRAggExpr::Var(input, ddof)))
            },
            AggExpr::AggGroups(input) => {
                to_aexpr_impl_materialized_lit(owned(input), arena, schema)?
                    .map_expr(arena, |input| AExpr::Agg(IRAggExpr::AggGroups(input)))
            },
        },
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            todo!()
            // let predicate = to_aexpr_impl_materialized_lit(owned(predicate), arena, schema)?.node;
            // let truthy = to_aexpr_impl(owned(truthy), arena, schema)?;
            // let falsy = to_aexpr_impl(owned(falsy), arena, schema)?.node;
            //
            // truthy.map_expr(arena, |truthy| AExpr::Ternary {
            //     predicate,
            //     truthy,
            //     falsy,
            // })
        },
        Expr::AnonymousFunction {
            input,
            function,
            output_type,
            options,
        } => {
            let e = to_expr_irs(input, arena, schema)?;
            let output_name = if e.is_empty() {
                options.fmt_str.into()
            } else {
                e[0].output_name().clone()
            };

            let node = arena.add(AExpr::AnonymousFunction {
                input: e,
                function,
                output_type,
                options,
            });
            dbg!();
            ExpandedDslToIrExpr::Simple(DslToIrExpr { node, output_name })
        },
        Expr::Function {
            input,
            function,
            options,
        } => return convert_functions(input, function, options, arena, schema),
        Expr::Window {
            function,
            partition_by,
            order_by,
            options,
        } => {
            todo!();
            // let function = to_aexpr_impl(owned(function), arena, schema)?;
            // let order_by = if let Some((e, options)) = order_by {
            //     Some((
            //         to_aexpr_impl(owned(e.clone()), arena, schema)?.node,
            //         options,
            //     ))
            // } else {
            //     None
            // };
            // let partition_by = partition_by
            //     .into_iter()
            //     .map(|e| Ok(to_aexpr_impl_materialized_lit(e, arena, schema)?.node))
            //     .collect::<PolarsResult<_>>()?;
            //
            // function.map_expr(arena, |function| AExpr::Window {
            //     function,
            //     partition_by,
            //     order_by,
            //     options,
            // })
        },
        Expr::Slice {
            input,
            offset,
            length,
        } => {
            todo!()
            // let input = to_aexpr_impl(owned(input), arena, schema)?;
            // let offset = to_aexpr_impl_materialized_lit(owned(offset), arena, schema)?.node;
            // let length = to_aexpr_impl_materialized_lit(owned(length), arena, schema)?.node;
            //
            // input.map_expr(arena, |input| AExpr::Slice {
            //     input,
            //     offset,
            //     length,
            // })
        },
        Expr::Eval {
            expr,
            evaluation,
            variant,
        } => {
            let expr = to_aexpr_impl(owned(expr), arena, schema)?;
            let ExpandedDslToIrExpr::Simple(evaluation) =
                to_aexpr_impl(owned(evaluation), arena, schema)?
            else {
                polars_bail!(nyi = "selectors in `eval` expressions");
            };
            let evaluation = evaluation.node;

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

            expr.map_expr(arena, |expr| AExpr::Eval {
                expr,
                evaluation,
                variant,
            })
        },
        Expr::Len => ExpandedDslToIrExpr::Simple(DslToIrExpr {
            node: arena.add(AExpr::Len),
            output_name: get_len_name(),
        }),
        Expr::Selector(selector) => {
            let columns = selector.into_columns(schema)?;
            ExpandedDslToIrExpr::Expanded(
                columns
                    .into_iter()
                    .map(|c| DslToIrExpr {
                        node: arena.add(AExpr::Column(c.clone())),
                        output_name: c,
                    })
                    .collect(),
            )
        },
        #[cfg(feature = "dtype-struct")]
        e @ Expr::Field(_) => {
            polars_bail!(InvalidOperation: "'Expr: {}' not allowed in this context/location", e)
        },
        e @ Expr::SubPlan { .. } | e @ Expr::KeepName(_) | e @ Expr::RenameAlias { .. } => {
            polars_bail!(InvalidOperation: "'Expr: {}' not allowed in this context/location", e)
        },
    })
}
