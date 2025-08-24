use super::functions::convert_functions;
use super::*;
use crate::plans::iterator::ArenaExprIter;

pub fn to_expr_ir(expr: Expr, ctx: &mut ExprToIRContext) -> PolarsResult<ExprIR> {
    let (node, output_name) = to_aexpr_impl(expr, ctx)?;
    Ok(ExprIR::new(node, OutputName::Alias(output_name)))
}

pub fn to_expr_ir_materialized_lit(expr: Expr, ctx: &mut ExprToIRContext) -> PolarsResult<ExprIR> {
    let (node, output_name) = to_aexpr_impl_materialized_lit(expr, ctx)?;
    Ok(ExprIR::new(node, OutputName::Alias(output_name)))
}

pub(super) fn to_expr_irs(
    input: Vec<Expr>,
    ctx: &mut ExprToIRContext,
) -> PolarsResult<Vec<ExprIR>> {
    let original_with_fields = ctx.with_fields.clone();
    input
        .into_iter()
        .map(|e| {
            let e = to_expr_ir(e, ctx)?;
            ctx.with_fields = original_with_fields.clone();
            Ok(e)
        })
        .collect()
}

fn to_aexpr_impl_materialized_lit(
    expr: Expr,
    ctx: &mut ExprToIRContext,
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
    to_aexpr_impl(e, ctx)
}

pub struct ExprToIRContext<'a> {
    pub(super) with_fields: Option<(Node, Schema)>,
    pub arena: &'a mut Arena<AExpr>,
    pub schema: &'a Schema,

    pub allow_unknown: bool,
    /// Check whether mentioned column names exist in the schema.
    pub check_column_names: bool,
}

impl<'a> ExprToIRContext<'a> {
    pub fn new(arena: &'a mut Arena<AExpr>, schema: &'a Schema) -> Self {
        Self {
            with_fields: None,
            arena,
            schema,
            allow_unknown: false,
            check_column_names: true,
        }
    }

    pub fn new_with_opt_eager(
        arena: &'a mut Arena<AExpr>,
        schema: &'a Schema,
        optflags: &OptFlags,
    ) -> Self {
        let mut ctx = Self::new(arena, schema);
        ctx.allow_unknown = optflags.contains(OptFlags::EAGER);
        ctx
    }

    pub fn new_no_verification(arena: &'a mut Arena<AExpr>, schema: &'a Schema) -> Self {
        let mut ctx = Self::new(arena, schema);
        ctx.allow_unknown = true;
        ctx.check_column_names = false;
        ctx
    }
}

/// Converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation.
pub(super) fn to_aexpr_impl(
    expr: Expr,
    ctx: &mut ExprToIRContext,
) -> PolarsResult<(Node, PlSmallStr)> {
    let owned = Arc::unwrap_or_clone;

    macro_rules! recurse {
        ($input:expr) => {
            to_aexpr_impl($input, ctx)
        };
    }
    macro_rules! recurse_arc {
        ($input:expr) => {
            recurse!(owned($input))
        };
    }

    macro_rules! to_aexpr_mat_lit {
        ($input:expr) => {
            to_aexpr_impl_materialized_lit($input, ctx)
        };
    }

    macro_rules! to_aexpr_mat_lit_arc {
        ($input:expr) => {
            to_aexpr_mat_lit!(owned($input))
        };
    }

    let (v, output_name) = match expr {
        Expr::Explode { input, skip_empty } => {
            let (expr, output_name) = recurse_arc!(input)?;
            (AExpr::Explode { expr, skip_empty }, output_name)
        },
        Expr::Alias(e, name) => return Ok((recurse_arc!(e)?.0, name)),
        Expr::Literal(lv) => {
            let output_name = lv.output_column_name().clone();
            (AExpr::Literal(lv), output_name)
        },
        Expr::Column(name) => {
            if ctx.check_column_names {
                ctx.schema.try_index_of(&name)?;
            }
            (AExpr::Column(name.clone()), name)
        },
        Expr::BinaryExpr { left, op, right } => {
            let (l, output_name) = recurse_arc!(left)?;
            let (r, _) = recurse_arc!(right)?;
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
            let (expr, output_name) = recurse_arc!(expr)?;
            (
                AExpr::Cast {
                    expr,
                    dtype: dtype.into_datatype(ctx.schema)?,
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
            let (expr, output_name) = recurse_arc!(expr)?;
            let (idx, _) = to_aexpr_mat_lit_arc!(idx)?;
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
            let (expr, output_name) = recurse_arc!(expr)?;
            (AExpr::Sort { expr, options }, output_name)
        },
        Expr::SortBy {
            expr,
            by,
            sort_options,
        } => {
            let (expr, output_name) = recurse_arc!(expr)?;
            let by = by
                .into_iter()
                .map(|e| Ok(recurse!(e)?.0))
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
            let (input, output_name) = recurse_arc!(input)?;
            let (by, _) = recurse_arc!(by)?;
            (AExpr::Filter { input, by }, output_name)
        },
        Expr::Agg(agg) => {
            let (a_agg, output_name) = match agg {
                AggExpr::Min {
                    input,
                    propagate_nans,
                } => {
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
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
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
                    (
                        IRAggExpr::Max {
                            input,
                            propagate_nans,
                        },
                        output_name,
                    )
                },
                AggExpr::Median(input) => {
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
                    (IRAggExpr::Median(input), output_name)
                },
                AggExpr::NUnique(input) => {
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
                    (IRAggExpr::NUnique(input), output_name)
                },
                AggExpr::First(input) => {
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
                    (IRAggExpr::First(input), output_name)
                },
                AggExpr::Last(input) => {
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
                    (IRAggExpr::Last(input), output_name)
                },
                AggExpr::Mean(input) => {
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
                    (IRAggExpr::Mean(input), output_name)
                },
                AggExpr::Implode(input) => {
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
                    (IRAggExpr::Implode(input), output_name)
                },
                AggExpr::Count {
                    input,
                    include_nulls,
                } => {
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
                    (
                        IRAggExpr::Count {
                            input,
                            include_nulls,
                        },
                        output_name,
                    )
                },
                AggExpr::Quantile {
                    expr,
                    quantile,
                    method,
                } => {
                    let (expr, output_name) = to_aexpr_mat_lit_arc!(expr)?;
                    let (quantile, _) = to_aexpr_mat_lit_arc!(quantile)?;
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
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
                    (IRAggExpr::Sum(input), output_name)
                },
                AggExpr::Std(input, ddof) => {
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
                    (IRAggExpr::Std(input, ddof), output_name)
                },
                AggExpr::Var(input, ddof) => {
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
                    (IRAggExpr::Var(input, ddof), output_name)
                },
                AggExpr::AggGroups(input) => {
                    let (input, output_name) = to_aexpr_mat_lit_arc!(input)?;
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
            let (p, _) = to_aexpr_mat_lit_arc!(predicate)?;
            let (t, output_name) = recurse_arc!(truthy)?;
            let (f, _) = recurse_arc!(falsy)?;
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
            options,
            fmt_str,
        } => {
            let input = to_expr_irs(input, ctx)?;
            let output_name = if input.is_empty() {
                fmt_str.as_ref().clone()
            } else {
                input[0].output_name().clone()
            };

            let fields = input
                .iter()
                .map(|e| e.field(ctx.schema, ctx.arena))
                .collect::<PolarsResult<Vec<_>>>()?;

            let function = function.materialize()?;
            let out = function.get_field(ctx.schema, &fields)?;
            let output_dtype = out.dtype();

            assert!(
                output_dtype.is_known(),
                "output type of anonymous functions must bet set"
            );

            (
                AExpr::AnonymousFunction {
                    input,
                    function: LazySerde::Deserialized(function),
                    options,
                    fmt_str,
                },
                output_name,
            )
        },
        Expr::Function { input, function } => {
            return convert_functions(input, function, ctx);
        },
        Expr::Window {
            function,
            partition_by,
            order_by,
            options,
        } => {
            let (function, output_name) = recurse_arc!(function)?;
            let order_by = if let Some((e, options)) = order_by {
                Some((recurse_arc!(e)?.0, options))
            } else {
                None
            };

            (
                AExpr::Window {
                    function,
                    partition_by: partition_by
                        .into_iter()
                        .map(|e| Ok(to_aexpr_impl_materialized_lit(e, ctx)?.0))
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
            let (input, output_name) = recurse_arc!(input)?;
            let (offset, _) = to_aexpr_mat_lit_arc!(offset)?;
            let (length, _) = to_aexpr_mat_lit_arc!(length)?;
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
            let (expr, output_name) = recurse_arc!(expr)?;
            let expr_dtype = ctx.arena.get(expr).to_dtype(ctx.schema, ctx.arena)?;
            let element_dtype = variant.element_dtype(&expr_dtype)?;

            // Perform this before schema resolution so that we can better error messages.
            for e in evaluation.as_ref().into_iter() {
                if let Expr::Column(name) = e {
                    polars_ensure!(
                        name.is_empty(),
                        ComputeError:
                        "named columns are not allowed in `eval` functions; consider using `element`"
                    );
                }
            }

            let evaluation_schema = Schema::from_iter([(PlSmallStr::EMPTY, element_dtype.clone())]);
            let mut evaluation_ctx = ExprToIRContext {
                with_fields: None,
                schema: &evaluation_schema,
                arena: ctx.arena,
                allow_unknown: ctx.allow_unknown,
                check_column_names: ctx.check_column_names,
            };
            let (evaluation, _) = to_aexpr_impl(owned(evaluation), &mut evaluation_ctx)?;

            match variant {
                EvalVariant::List => {},
                EvalVariant::Cumulative { .. } => {
                    polars_ensure!(
                        is_scalar_ae(evaluation, ctx.arena),
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
        Expr::KeepName(expr) => {
            let (expr, _) = to_aexpr_impl(owned(expr), ctx)?;
            let name = ArenaExprIter::iter(ctx.arena, expr).find_map(|e| match e.1 {
                AExpr::Column(name) => Some(name.clone()),
                _ => None,
            });
            let Some(name) = name else {
                polars_bail!(
                    InvalidOperation:
                    "`name.keep_name` expected at least one column name"
                );
            };
            return Ok((expr, name));
        },
        Expr::DataTypeFunction(f) => super::datatype_fn_to_ir::datatype_fn_to_aexpr(f, ctx)?,
        Expr::RenameAlias { expr, function } => {
            let (expr, name) = to_aexpr_impl(owned(expr), ctx)?;
            let name = function.call(&name)?;
            return Ok((expr, name));
        },
        #[cfg(feature = "dtype-struct")]
        Expr::Field(name) => {
            assert_eq!(
                name.len(),
                1,
                "should have been handled in expression expansion"
            );
            let name = &name[0];

            let Some((input, with_fields)) = &ctx.with_fields else {
                polars_bail!(InvalidOperation: "`pl.field()` called outside of struct context");
            };

            if !with_fields.contains(name) {
                polars_bail!(
                    InvalidOperation: "field `{name}` does not exist on struct with fields {:?}",
                    with_fields.iter_names_cloned().collect::<Vec<_>>().as_slice()
                );
            }

            let function = IRFunctionExpr::StructExpr(IRStructFunction::FieldByName(name.clone()));
            let options = function.function_options();
            (
                AExpr::Function {
                    input: vec![ExprIR::new(*input, OutputName::Alias(PlSmallStr::EMPTY))],
                    function,
                    options,
                },
                name.clone(),
            )
        },

        e @ Expr::SubPlan { .. } | e @ Expr::Selector(_) => {
            polars_bail!(InvalidOperation: "'Expr: {}' not allowed in this context/location", e)
        },
    };
    Ok((ctx.arena.add(v), output_name))
}
