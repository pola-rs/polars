use polars_core::prelude::*;
use polars_plan::prelude::expr_ir::ExprIR;
use polars_plan::prelude::*;
use recursive::recursive;

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
        Window {
            function,
            partition_by,
            order_by,
            options,
        } => {
            let mut function = *function;
            state.set_window();
            let phys_function = create_physical_expr_inner(
                function,
                Context::Aggregation,
                expr_arena,
                schema,
                state,
            )?;

            let order_by = order_by
                .map(|(node, options)| {
                    PolarsResult::Ok((
                        create_physical_expr_inner(
                            node,
                            Context::Aggregation,
                            expr_arena,
                            schema,
                            state,
                        )?,
                        options,
                    ))
                })
                .transpose()?;

            let mut out_name = None;
            if let Alias(expr, name) = expr_arena.get(function) {
                function = *expr;
                out_name = Some(name.clone());
            };
            let function_expr = node_to_expr(function, expr_arena);
            let expr = node_to_expr(expression, expr_arena);

            // set again as the state can be reset
            state.set_window();
            match options {
                WindowType::Over(mapping) => {
                    // TODO! Order by
                    let group_by = create_physical_expressions_from_nodes(
                        partition_by,
                        Context::Aggregation,
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

                    Ok(Arc::new(WindowExpr {
                        group_by,
                        order_by,
                        apply_columns,
                        out_name,
                        function: function_expr,
                        phys_function,
                        mapping: *mapping,
                        expr,
                    }))
                },
                #[cfg(feature = "dynamic_group_by")]
                WindowType::Rolling(options) => Ok(Arc::new(RollingExpr {
                    function: function_expr,
                    phys_function,
                    out_name,
                    options: options.clone(),
                    expr,
                })),
            }
        },
        Literal(value) => {
            state.local.has_lit = true;
            Ok(Arc::new(LiteralExpr::new(
                value.clone(),
                node_to_expr(expression, expr_arena),
            )))
        },
        BinaryExpr { left, op, right } => {
            let output_field =
                expr_arena
                    .get(expression)
                    .to_field(schema, Context::Default, expr_arena)?;
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

                    let groupby = match agg {
                        I::Min { propagate_nans, .. } if *propagate_nans => GBM::NanMin,
                        I::Min { .. } => GBM::Min,
                        I::Max { propagate_nans, .. } if *propagate_nans => GBM::NanMax,
                        I::Max { .. } => GBM::Max,
                        I::Median(_) => GBM::Median,
                        I::NUnique(_) => GBM::NUnique,
                        I::First(_) => GBM::First,
                        I::Last(_) => GBM::Last,
                        I::Mean(_) => GBM::Mean,
                        I::Implode(_) => GBM::Implode,
                        I::Quantile { .. } => unreachable!(),
                        I::Sum(_) => GBM::Sum,
                        I::Count(_, include_nulls) => GBM::Count {
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

                    Ok(Arc::new(AggregationExpr::new(input, agg_type, None)))
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

                    let field = expr_arena.get(expression).to_field(
                        schema,
                        Context::Aggregation,
                        expr_arena,
                    )?;

                    let groupby = GroupByMethod::from(agg.clone());
                    let agg_type = AggregationType {
                        groupby,
                        allow_threading: false,
                    };
                    Ok(Arc::new(AggregationExpr::new(input, agg_type, Some(field))))
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
                inlined_eval: Default::default(),
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
            output_type: _,
            options,
        } => {
            let is_scalar = is_scalar_ae(expression, expr_arena);
            let output_field = expr_arena
                .get(expression)
                .to_field(schema, ctxt, expr_arena)?;

            let input =
                create_physical_expressions_from_irs(input, ctxt, expr_arena, schema, state)?;

            Ok(Arc::new(ApplyExpr::new(
                input,
                function.clone().materialize()?,
                node_to_expr(expression, expr_arena),
                *options,
                state.allow_threading,
                schema.clone(),
                output_field,
                is_scalar,
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
                .to_field(schema, ctxt, expr_arena)?;
            let input =
                create_physical_expressions_from_irs(input, ctxt, expr_arena, schema, state)?;

            Ok(Arc::new(ApplyExpr::new(
                input,
                function.clone().into(),
                node_to_expr(expression, expr_arena),
                *options,
                state.allow_threading,
                schema.clone(),
                output_field,
                is_scalar,
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
            polars_ensure!(!(state.has_implode() && matches!(ctxt, Context::Aggregation)), InvalidOperation: "'implode' followed by a slice during aggregation is not allowed");
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
                move |c: &mut [polars_core::frame::column::Column]| {
                    c[0].explode(skip_empty).map(Some)
                },
            ) as Arc<dyn ColumnsUdf>);

            let field = expr_arena
                .get(expression)
                .to_field(schema, ctxt, expr_arena)?;

            Ok(Arc::new(ApplyExpr::new(
                vec![input],
                function,
                node_to_expr(expression, expr_arena),
                FunctionOptions::groupwise(),
                state.allow_threading,
                schema.clone(),
                field,
                false,
            )))
        },
        Alias(input, name) => {
            let phys_expr = create_physical_expr_inner(*input, ctxt, expr_arena, schema, state)?;
            Ok(Arc::new(AliasExpr::new(
                phys_expr,
                name.clone(),
                node_to_expr(*input, expr_arena),
            )))
        },
    }
}
