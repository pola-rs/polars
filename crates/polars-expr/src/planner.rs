use polars_core::prelude::*;
use polars_plan::prelude::expr_ir::ExprIR;
use polars_plan::prelude::*;

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

fn ok_checker(_state: &ExpressionConversionState) -> PolarsResult<()> {
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
    F: Fn(&ExpressionConversionState) -> PolarsResult<()>,
{
    exprs
        .iter()
        .map(|e| {
            state.reset();
            let out = create_physical_expr(e, context, expr_arena, schema, state);
            checker(state)?;
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
    F: Fn(&ExpressionConversionState) -> PolarsResult<()>,
{
    exprs
        .iter()
        .map(|e| {
            state.reset();
            let out = create_physical_expr_inner(*e, context, expr_arena, schema, state);
            checker(state)?;
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
    depth_limit: u16,
}

#[derive(Copy, Clone)]
struct LocalConversionState {
    has_implode: bool,
    has_window: bool,
    has_lit: bool,
    // Max depth an expression may have.
    // 0 is unlimited.
    depth_limit: u16,
}

impl Default for LocalConversionState {
    fn default() -> Self {
        Self {
            has_lit: false,
            has_implode: false,
            has_window: false,
            depth_limit: 500,
        }
    }
}

impl ExpressionConversionState {
    pub fn new(allow_threading: bool, depth_limit: u16) -> Self {
        Self {
            depth_limit,
            allow_threading,
            has_windows: false,
            local: LocalConversionState {
                depth_limit,
                ..Default::default()
            },
        }
    }
    fn reset(&mut self) {
        self.local = LocalConversionState {
            depth_limit: self.depth_limit,
            ..Default::default()
        }
    }

    fn has_implode(&self) -> bool {
        self.local.has_implode
    }

    fn set_window(&mut self) {
        self.has_windows = true;
        self.local.has_window = true;
    }

    fn check_depth(&mut self) {
        if self.local.depth_limit > 0 {
            self.local.depth_limit -= 1;

            if self.local.depth_limit == 0 {
                let depth = get_expr_depth_limit().unwrap();
                polars_warn!(format!("encountered expression deeper than {depth} elements; this may overflow the stack, consider refactoring"))
            }
        }
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

fn create_physical_expr_inner(
    expression: Node,
    ctxt: Context,
    expr_arena: &Arena<AExpr>,
    schema: &SchemaRef,
    state: &mut ExpressionConversionState,
) -> PolarsResult<Arc<dyn PhysicalExpr>> {
    use AExpr::*;

    state.check_depth();

    match expr_arena.get(expression) {
        Len => Ok(Arc::new(phys_expr::CountExpr::new())),
        Window {
            mut function,
            partition_by,
            order_by,
            options,
        } => {
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
                        #[cfg(feature = "bitwise")]
                        I::Bitwise(_, f) => GBM::Bitwise((*f).into()),
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

            let is_reducing_aggregation = options.flags.contains(FunctionFlags::RETURNS_SCALAR)
                && matches!(options.collect_groups, ApplyOptions::GroupWise);
            // Will be reset in the function so get that here.
            let has_window = state.local.has_window;
            let input = create_physical_expressions_check_state(
                input,
                ctxt,
                expr_arena,
                schema,
                state,
                |state| {
                    polars_ensure!(!((is_reducing_aggregation || has_window) && state.has_implode() && matches!(ctxt, Context::Aggregation)), InvalidOperation: "'implode' followed by an aggregation is not allowed");
                    Ok(())
                },
            )?;

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
            let is_reducing_aggregation = options.flags.contains(FunctionFlags::RETURNS_SCALAR)
                && matches!(options.collect_groups, ApplyOptions::GroupWise);
            // Will be reset in the function so get that here.
            let has_window = state.local.has_window;
            let input = create_physical_expressions_check_state(
                input,
                ctxt,
                expr_arena,
                schema,
                state,
                |state| {
                    polars_ensure!(!((is_reducing_aggregation || has_window) && state.has_implode() && matches!(ctxt, Context::Aggregation)), InvalidOperation: "'implode' followed by an aggregation is not allowed");
                    Ok(())
                },
            )?;

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
        Explode(expr) => {
            let input = create_physical_expr_inner(*expr, ctxt, expr_arena, schema, state)?;
            let function = SpecialEq::new(Arc::new(
                move |c: &mut [polars_core::frame::column::Column]| c[0].explode().map(Some),
            ) as Arc<dyn ColumnsUdf>);

            let field = expr_arena
                .get(expression)
                .to_field(schema, ctxt, expr_arena)?;

            Ok(Arc::new(ApplyExpr::new(
                vec![input],
                function,
                node_to_expr(expression, expr_arena),
                FunctionOptions {
                    collect_groups: ApplyOptions::GroupWise,
                    fmt_str: "",
                    cast_to_supertypes: None,
                    check_lengths: Default::default(),
                    flags: Default::default(),
                },
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
