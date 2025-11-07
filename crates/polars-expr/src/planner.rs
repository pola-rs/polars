use polars_core::prelude::*;
use polars_plan::constants::PL_ELEMENT_NAME;
use polars_plan::prelude::expr_ir::ExprIR;
use polars_plan::prelude::*;
use recursive::recursive;

use crate::dispatch::{function_expr_to_groups_udf, function_expr_to_udf};
use crate::expressions as phys_expr;
use crate::expressions::*;

pub fn get_expr_depth_limit() -> PolarsResult<u16> {
    let depth = if let Ok(d) = std::env::var("POLARS_MAX_EXPR_DEPTH") {
        let v = d
            .parse::<u64>()
            .map_err(|_| polars_err!(ComputeError: "could not parse 'max_expr_depth': {}", d))?;
        u16::try_from(v).unwrap_or(0)
    } else {
        512
    };
    Ok(depth)
}

fn ok_checker(_i: usize, _state: &ExpressionConversionState) -> PolarsResult<()> {
    Ok(())
}

pub fn create_physical_expressions_from_irs(
    exprs: &[ExprIR],
    context: Context,
    expr_arena: &Arena<AExpr>,
    schema: &SchemaRef,
    state: &mut ExpressionConversionState,
) -> PolarsResult<Vec<Arc<dyn PhysicalExpr>>> {
    create_physical_expressions_check_state(exprs, context, expr_arena, schema, state, ok_checker)
}

pub(crate) fn create_physical_expressions_check_state<F>(
    exprs: &[ExprIR],
    context: Context,
    expr_arena: &Arena<AExpr>,
    schema: &SchemaRef,
    state: &mut ExpressionConversionState,
    checker: F,
) -> PolarsResult<Vec<Arc<dyn PhysicalExpr>>>
where
    F: Fn(usize, &ExpressionConversionState) -> PolarsResult<()>,
{
    exprs
        .iter()
        .enumerate()
        .map(|(i, e)| {
            state.reset();
            let out = create_physical_expr(e, context, expr_arena, schema, state);
            checker(i, state)?;
            out
        })
        .collect()
}

pub(crate) fn create_physical_expressions_from_nodes(
    exprs: &[Node],
    context: Context,
    expr_arena: &Arena<AExpr>,
    schema: &SchemaRef,
    state: &mut ExpressionConversionState,
) -> PolarsResult<Vec<Arc<dyn PhysicalExpr>>> {
    create_physical_expressions_from_nodes_check_state(
        exprs, context, expr_arena, schema, state, ok_checker,
    )
}

pub(crate) fn create_physical_expressions_from_nodes_check_state<F>(
    exprs: &[Node],
    context: Context,
    expr_arena: &Arena<AExpr>,
    schema: &SchemaRef,
    state: &mut ExpressionConversionState,
    checker: F,
) -> PolarsResult<Vec<Arc<dyn PhysicalExpr>>>
where
    F: Fn(usize, &ExpressionConversionState) -> PolarsResult<()>,
{
    exprs
        .iter()
        .enumerate()
        .map(|(i, e)| {
            state.reset();
            let out = create_physical_expr_inner(*e, context, expr_arena, schema, state);
            checker(i, state)?;
            out
        })
        .collect()
}

#[derive(Copy, Clone)]
pub struct ExpressionConversionState {
    // settings per context
    // they remain activate between
    // expressions
    pub allow_threading: bool,
    pub has_windows: bool,
    // settings per expression
    // those are reset every expression
    local: LocalConversionState,
}

#[derive(Copy, Clone, Default)]
struct LocalConversionState {
    has_implode: bool,
    has_window: bool,
    has_lit: bool,
}

impl ExpressionConversionState {
    pub fn new(allow_threading: bool) -> Self {
        Self {
            allow_threading,
            has_windows: false,
            local: LocalConversionState {
                ..Default::default()
            },
        }
    }

    fn reset(&mut self) {
        self.local = LocalConversionState::default();
    }

    fn has_implode(&self) -> bool {
        self.local.has_implode
    }

    fn set_window(&mut self) {
        self.has_windows = true;
        self.local.has_window = true;
    }
}

pub fn create_physical_expr(
    expr_ir: &ExprIR,
    ctxt: Context,
    expr_arena: &Arena<AExpr>,
    schema: &SchemaRef,
    state: &mut ExpressionConversionState,
) -> PolarsResult<Arc<dyn PhysicalExpr>> {
    let phys_expr = create_physical_expr_inner(expr_ir.node(), ctxt, expr_arena, schema, state)?;

    if let Some(name) = expr_ir.get_alias() {
        Ok(Arc::new(AliasExpr::new(
            phys_expr,
            name.clone(),
            node_to_expr(expr_ir.node(), expr_arena),
        )))
    } else {
        Ok(phys_expr)
    }
}

#[recursive]
fn create_physical_expr_inner(
    expression: Node,
    ctxt: Context,
    expr_arena: &Arena<AExpr>,
    schema: &SchemaRef,
    state: &mut ExpressionConversionState,
) -> PolarsResult<Arc<dyn PhysicalExpr>> {
    use AExpr::*;

    match expr_arena.get(expression) {
        Len => Ok(Arc::new(phys_expr::CountExpr::new())),
        #[cfg(feature = "dynamic_group_by")]
        aexpr @ Rolling {
            function,
            index_column,
            period,
            offset,
            closed_window,
        } => {
            let index_column = create_physical_expr_inner(
                *index_column,
                Context::Default,
                expr_arena,
                schema,
                state,
            )?;

            let output_field = aexpr.to_field(&ToFieldContext::new(expr_arena, schema))?;
            let function = *function;
            state.set_window();
            let phys_function =
                create_physical_expr_inner(function, Context::Default, expr_arena, schema, state)?;

            let expr = node_to_expr(expression, expr_arena);

            // set again as the state can be reset
            state.set_window();
            Ok(Arc::new(RollingExpr {
                phys_function,
                index_column,
                period: *period,
                offset: *offset,
                closed_window: *closed_window,
                expr,
                output_field,
            }))
        },
        aexpr @ Over {
            function,
            partition_by,
            order_by,
            mapping,
        } => {
            let output_field = aexpr.to_field(&ToFieldContext::new(expr_arena, schema))?;
            let function = *function;
            state.set_window();
            let phys_function =
                create_physical_expr_inner(function, Context::Default, expr_arena, schema, state)?;

            let order_by = order_by
                .map(|(node, options)| {
                    PolarsResult::Ok((
                        create_physical_expr_inner(
                            node,
                            Context::Default,
                            expr_arena,
                            schema,
                            state,
                        )?,
                        options,
                    ))
                })
                .transpose()?;

            let expr = node_to_expr(expression, expr_arena);

            // set again as the state can be reset
            state.set_window();
            // TODO! Order by
            let group_by = create_physical_expressions_from_nodes(
                partition_by,
                Context::Default,
                expr_arena,
                schema,
                state,
            )?;
            let mut apply_columns = aexpr_to_leaf_names(function, expr_arena);
            // sort and then dedup removes consecutive duplicates == all duplicates
            apply_columns.sort();
            apply_columns.dedup();

            if apply_columns.is_empty() {
                if has_aexpr(function, expr_arena, |e| matches!(e, AExpr::Literal(_))) {
                    apply_columns.push(PlSmallStr::from_static("literal"))
                } else if has_aexpr(function, expr_arena, |e| matches!(e, AExpr::Len)) {
                    apply_columns.push(PlSmallStr::from_static("len"))
                } else {
                    let e = node_to_expr(function, expr_arena);
                    polars_bail!(
                        ComputeError:
                        "cannot apply a window function, did not find a root column; \
                        this is likely due to a syntax error in this expression: {:?}", e
                    );
                }
            }

            // Check if the branches have an aggregation
            // when(a > sum)
            // then (foo)
            // otherwise(bar - sum)
            let mut has_arity = false;
            let mut agg_col = false;
            for (_, e) in expr_arena.iter(function) {
                match e {
                    AExpr::Ternary { .. } | AExpr::BinaryExpr { .. } => {
                        has_arity = true;
                    },
                    AExpr::Agg(_) => {
                        agg_col = true;
                    },
                    AExpr::Function { options, .. } | AExpr::AnonymousFunction { options, .. } => {
                        if options.flags.returns_scalar() {
                            agg_col = true;
                        }
                    },
                    _ => {},
                }
            }
            let has_different_group_sources = has_arity && agg_col;

            Ok(Arc::new(WindowExpr {
                group_by,
                order_by,
                apply_columns,
                phys_function,
                mapping: *mapping,
                expr,
                has_different_group_sources,
                output_field,
            }))
        },
        Literal(value) => {
            state.local.has_lit = true;
            Ok(Arc::new(LiteralExpr::new(
                value.clone(),
                node_to_expr(expression, expr_arena),
            )))
        },
        BinaryExpr { left, op, right } => {
            let output_field = expr_arena
                .get(expression)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;
            let is_scalar = is_scalar_ae(expression, expr_arena);
            let lhs = create_physical_expr_inner(*left, ctxt, expr_arena, schema, state)?;
            let rhs = create_physical_expr_inner(*right, ctxt, expr_arena, schema, state)?;
            Ok(Arc::new(phys_expr::BinaryExpr::new(
                lhs,
                *op,
                rhs,
                node_to_expr(expression, expr_arena),
                state.local.has_lit,
                state.allow_threading,
                is_scalar,
                output_field,
            )))
        },
        Column(column) => Ok(Arc::new(ColumnExpr::new(
            column.clone(),
            node_to_expr(expression, expr_arena),
            schema.clone(),
        ))),
        Element => Ok(Arc::new(ColumnExpr::new(
            PL_ELEMENT_NAME.clone(),
            node_to_expr(expression, expr_arena),
            schema.clone(),
        ))),
        Sort { expr, options } => {
            let phys_expr = create_physical_expr_inner(*expr, ctxt, expr_arena, schema, state)?;
            Ok(Arc::new(SortExpr::new(
                phys_expr,
                *options,
                node_to_expr(expression, expr_arena),
            )))
        },
        Gather {
            expr,
            idx,
            returns_scalar,
        } => {
            let phys_expr = create_physical_expr_inner(*expr, ctxt, expr_arena, schema, state)?;
            let phys_idx = create_physical_expr_inner(*idx, ctxt, expr_arena, schema, state)?;
            Ok(Arc::new(GatherExpr {
                phys_expr,
                idx: phys_idx,
                expr: node_to_expr(expression, expr_arena),
                returns_scalar: *returns_scalar,
            }))
        },
        SortBy {
            expr,
            by,
            sort_options,
        } => {
            let phys_expr = create_physical_expr_inner(*expr, ctxt, expr_arena, schema, state)?;
            let phys_by =
                create_physical_expressions_from_nodes(by, ctxt, expr_arena, schema, state)?;
            Ok(Arc::new(SortByExpr::new(
                phys_expr,
                phys_by,
                node_to_expr(expression, expr_arena),
                sort_options.clone(),
            )))
        },
        Filter { input, by } => {
            let phys_input = create_physical_expr_inner(*input, ctxt, expr_arena, schema, state)?;
            let phys_by = create_physical_expr_inner(*by, ctxt, expr_arena, schema, state)?;
            Ok(Arc::new(FilterExpr::new(
                phys_input,
                phys_by,
                node_to_expr(expression, expr_arena),
            )))
        },
        Agg(agg) => {
            let expr = agg.get_input().first();
            let input = create_physical_expr_inner(expr, ctxt, expr_arena, schema, state)?;
            polars_ensure!(!(state.has_implode() && matches!(ctxt, Context::Aggregation)), InvalidOperation: "'implode' followed by an aggregation is not allowed");
            state.local.has_implode |= matches!(agg, IRAggExpr::Implode(_));
            let allow_threading = state.allow_threading;

            match ctxt {
                Context::Default if !matches!(agg, IRAggExpr::Quantile { .. }) => {
                    use {GroupByMethod as GBM, IRAggExpr as I};

                    let output_field = expr_arena
                        .get(expression)
                        .to_field(&ToFieldContext::new(expr_arena, schema))?;
                    let groupby = match agg {
                        I::Min { propagate_nans, .. } if *propagate_nans => GBM::NanMin,
                        I::Min { .. } => GBM::Min,
                        I::Max { propagate_nans, .. } if *propagate_nans => GBM::NanMax,
                        I::Max { .. } => GBM::Max,
                        I::Median(_) => GBM::Median,
                        I::NUnique(_) => GBM::NUnique,
                        I::First(_) => GBM::First,
                        I::Last(_) => GBM::Last,
                        I::Item { allow_empty, .. } => GBM::Item {
                            allow_empty: *allow_empty,
                        },
                        I::Mean(_) => GBM::Mean,
                        I::Implode(_) => GBM::Implode,
                        I::Quantile { .. } => unreachable!(),
                        I::Sum(_) => GBM::Sum,
                        I::Count {
                            input: _,
                            include_nulls,
                        } => GBM::Count {
                            include_nulls: *include_nulls,
                        },
                        I::Std(_, ddof) => GBM::Std(*ddof),
                        I::Var(_, ddof) => GBM::Var(*ddof),
                        I::AggGroups(_) => {
                            polars_bail!(InvalidOperation: "agg groups expression only supported in aggregation context")
                        },
                    };

                    let agg_type = AggregationType {
                        groupby,
                        allow_threading,
                    };

                    Ok(Arc::new(AggregationExpr::new(
                        input,
                        agg_type,
                        output_field,
                    )))
                },
                _ => {
                    if let IRAggExpr::Quantile {
                        quantile,
                        method: interpol,
                        ..
                    } = agg
                    {
                        let quantile =
                            create_physical_expr_inner(*quantile, ctxt, expr_arena, schema, state)?;
                        return Ok(Arc::new(AggQuantileExpr::new(input, quantile, *interpol)));
                    }

                    let mut output_field = expr_arena
                        .get(expression)
                        .to_field(&ToFieldContext::new(expr_arena, schema))?;

                    if matches!(ctxt, Context::Aggregation) && !is_scalar_ae(expression, expr_arena)
                    {
                        output_field.coerce(output_field.dtype.clone().implode());
                    }

                    let groupby = GroupByMethod::from(agg.clone());
                    let agg_type = AggregationType {
                        groupby,
                        allow_threading: false,
                    };
                    Ok(Arc::new(AggregationExpr::new(
                        input,
                        agg_type,
                        output_field,
                    )))
                },
            }
        },
        Cast {
            expr,
            dtype,
            options,
        } => {
            let phys_expr = create_physical_expr_inner(*expr, ctxt, expr_arena, schema, state)?;
            Ok(Arc::new(CastExpr {
                input: phys_expr,
                dtype: dtype.clone(),
                expr: node_to_expr(expression, expr_arena),
                options: *options,
            }))
        },
        Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let is_scalar = is_scalar_ae(expression, expr_arena);
            let mut lit_count = 0u8;
            state.reset();
            let predicate =
                create_physical_expr_inner(*predicate, ctxt, expr_arena, schema, state)?;
            lit_count += state.local.has_lit as u8;
            state.reset();
            let truthy = create_physical_expr_inner(*truthy, ctxt, expr_arena, schema, state)?;
            lit_count += state.local.has_lit as u8;
            state.reset();
            let falsy = create_physical_expr_inner(*falsy, ctxt, expr_arena, schema, state)?;
            lit_count += state.local.has_lit as u8;
            Ok(Arc::new(TernaryExpr::new(
                predicate,
                truthy,
                falsy,
                node_to_expr(expression, expr_arena),
                state.allow_threading && lit_count < 2,
                is_scalar,
            )))
        },
        AnonymousFunction {
            input,
            function,
            options,
            fmt_str: _,
        } => {
            let is_scalar = is_scalar_ae(expression, expr_arena);
            let output_field = expr_arena
                .get(expression)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;

            let input =
                create_physical_expressions_from_irs(input, ctxt, expr_arena, schema, state)?;

            let function = function.clone().materialize()?;
            let function = function.into_inner().as_column_udf();

            Ok(Arc::new(ApplyExpr::new(
                input,
                SpecialEq::new(function),
                None,
                node_to_expr(expression, expr_arena),
                *options,
                state.allow_threading,
                schema.clone(),
                output_field,
                is_scalar,
                true,
            )))
        },
        Eval {
            expr,
            evaluation,
            variant,
        } => {
            let is_scalar = is_scalar_ae(expression, expr_arena);
            let evaluation_is_scalar = is_scalar_ae(*evaluation, expr_arena);
            let evaluation_is_elementwise = is_elementwise_rec(*evaluation, expr_arena);
            // @NOTE: This is actually also something the downstream apply code should care about.
            let mut pd_group = ExprPushdownGroup::Pushable;
            pd_group.update_with_expr_rec(expr_arena.get(*evaluation), expr_arena, None);
            let evaluation_is_fallible = matches!(pd_group, ExprPushdownGroup::Fallible);

            let output_field = expr_arena
                .get(expression)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;
            let input_field = expr_arena
                .get(*expr)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;
            let expr =
                create_physical_expr_inner(*expr, Context::Default, expr_arena, schema, state)?;

            let element_dtype = variant.element_dtype(&input_field.dtype)?;
            let mut eval_schema = schema.as_ref().clone();
            eval_schema.insert(PL_ELEMENT_NAME.clone(), element_dtype.clone());
            let evaluation = create_physical_expr_inner(
                *evaluation,
                // @Hack. Since EvalVariant::Array uses `evaluate_on_groups` to determine the
                // output and that expects to be outputting a list, we need to pretend like we are
                // aggregating here.
                //
                // EvalVariant::List also has this problem but that has a List datatype, so that
                // goes wrong by pure change and some black magic.
                if matches!(variant, EvalVariant::Array { .. }) && !evaluation_is_elementwise {
                    Context::Aggregation
                } else {
                    Context::Default
                },
                expr_arena,
                &Arc::new(eval_schema),
                state,
            )?;

            Ok(Arc::new(EvalExpr::new(
                expr,
                evaluation,
                *variant,
                node_to_expr(expression, expr_arena),
                output_field,
                is_scalar,
                evaluation_is_scalar,
                evaluation_is_elementwise,
                evaluation_is_fallible,
            )))
        },
        Function {
            input,
            function,
            options,
        } => {
            let is_scalar = is_scalar_ae(expression, expr_arena);
            let output_field = expr_arena
                .get(expression)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;
            let input =
                create_physical_expressions_from_irs(input, ctxt, expr_arena, schema, state)?;
            let is_fallible = expr_arena.get(expression).is_fallible_top_level(expr_arena);

            Ok(Arc::new(ApplyExpr::new(
                input,
                function_expr_to_udf(function.clone()),
                function_expr_to_groups_udf(function),
                node_to_expr(expression, expr_arena),
                *options,
                state.allow_threading,
                schema.clone(),
                output_field,
                is_scalar,
                is_fallible,
            )))
        },
        Slice {
            input,
            offset,
            length,
        } => {
            let input = create_physical_expr_inner(*input, ctxt, expr_arena, schema, state)?;
            let offset = create_physical_expr_inner(*offset, ctxt, expr_arena, schema, state)?;
            let length = create_physical_expr_inner(*length, ctxt, expr_arena, schema, state)?;
            polars_ensure!(!(state.has_implode() && matches!(ctxt, Context::Aggregation)),
                InvalidOperation: "'implode' followed by a slice during aggregation is not allowed");
            Ok(Arc::new(SliceExpr {
                input,
                offset,
                length,
                expr: node_to_expr(expression, expr_arena),
            }))
        },
        Explode { expr, skip_empty } => {
            let input = create_physical_expr_inner(*expr, ctxt, expr_arena, schema, state)?;
            let skip_empty = *skip_empty;
            let function = SpecialEq::new(Arc::new(
                move |c: &mut [polars_core::frame::column::Column]| c[0].explode(skip_empty),
            ) as Arc<dyn ColumnsUdf>);

            let output_field = expr_arena
                .get(expression)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;

            Ok(Arc::new(ApplyExpr::new(
                vec![input],
                function,
                None,
                node_to_expr(expression, expr_arena),
                FunctionOptions::groupwise(),
                state.allow_threading,
                schema.clone(),
                output_field,
                false,
                false,
            )))
        },
        AnonymousStreamingAgg { .. } => {
            polars_bail!(ComputeError: "anonymous agg not supported in in-memory engine")
        },
    }
}
