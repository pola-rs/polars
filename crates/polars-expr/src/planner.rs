use polars_core::prelude::*;
use polars_plan::constants::{get_literal_name, get_pl_element_name, get_pl_structfields_name};
use polars_plan::prelude::expr_ir::ExprIR;
use polars_plan::prelude::*;
use recursive::recursive;

use crate::dispatch::{function_expr_to_groups_udf, function_expr_to_udf};
use crate::expressions as phys_expr;
use crate::expressions::*;
use crate::reduce::GroupedReduction;

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
    expr_arena: &mut Arena<AExpr>,
    schema: &SchemaRef,
    state: &mut ExpressionConversionState,
) -> PolarsResult<Vec<Arc<dyn PhysicalExpr>>> {
    create_physical_expressions_check_state(exprs, expr_arena, schema, state, ok_checker)
}

pub(crate) fn create_physical_expressions_check_state<F>(
    exprs: &[ExprIR],
    expr_arena: &mut Arena<AExpr>,
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
            let out = create_physical_expr(e, expr_arena, schema, state);
            checker(i, state)?;
            out
        })
        .collect()
}

pub(crate) fn create_physical_expressions_from_nodes(
    exprs: &[Node],
    expr_arena: &mut Arena<AExpr>,
    schema: &SchemaRef,
    state: &mut ExpressionConversionState,
) -> PolarsResult<Vec<Arc<dyn PhysicalExpr>>> {
    create_physical_expressions_from_nodes_check_state(exprs, expr_arena, schema, state, ok_checker)
}

pub(crate) fn create_physical_expressions_from_nodes_check_state<F>(
    exprs: &[Node],
    expr_arena: &mut Arena<AExpr>,
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
            let out = create_physical_expr_inner(*e, expr_arena, schema, state);
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

    fn set_window(&mut self) {
        self.has_windows = true;
        self.local.has_window = true;
    }
}

pub fn create_physical_expr(
    expr_ir: &ExprIR,
    expr_arena: &mut Arena<AExpr>,
    schema: &SchemaRef, // Schema of the input.
    state: &mut ExpressionConversionState,
) -> PolarsResult<Arc<dyn PhysicalExpr>> {
    let phys_expr = create_physical_expr_inner(expr_ir.node(), expr_arena, schema, state)?;

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
    expr_arena: &mut Arena<AExpr>,
    schema: &SchemaRef, // Schema of the input.
    state: &mut ExpressionConversionState,
) -> PolarsResult<Arc<dyn PhysicalExpr>> {
    use AExpr::*;

    let aexpr = expr_arena.get(expression);
    match aexpr.clone() {
        Len => Ok(Arc::new(phys_expr::CountExpr::new())),
        #[cfg(feature = "dynamic_group_by")]
        Rolling {
            function,
            index_column,
            period,
            offset,
            closed_window,
        } => {
            let output_field = aexpr.to_field(&ToFieldContext::new(expr_arena, schema))?;
            let index_column = create_physical_expr_inner(index_column, expr_arena, schema, state)?;

            state.set_window();
            let phys_function = create_physical_expr_inner(function, expr_arena, schema, state)?;
            let expr = node_to_expr(expression, expr_arena);

            // set again as the state can be reset
            state.set_window();
            Ok(Arc::new(RollingExpr {
                phys_function,
                index_column,
                period,
                offset,
                closed_window,
                expr,
                output_field,
            }))
        },
        Over {
            function,
            partition_by,
            order_by,
            mapping,
        } => {
            let output_field = aexpr.to_field(&ToFieldContext::new(expr_arena, schema))?;
            state.set_window();
            let phys_function = create_physical_expr_inner(function, expr_arena, schema, state)?;

            let mut order_by_is_elementwise = false;
            let order_by = order_by
                .map(|(node, options)| {
                    order_by_is_elementwise |= is_elementwise_rec(node, expr_arena);
                    PolarsResult::Ok((
                        create_physical_expr_inner(node, expr_arena, schema, state)?,
                        options,
                    ))
                })
                .transpose()?;

            let expr = node_to_expr(expression, expr_arena);

            // set again as the state can be reset
            state.set_window();
            let all_group_by_are_elementwise = partition_by
                .iter()
                .all(|n| is_elementwise_rec(*n, expr_arena));
            let group_by =
                create_physical_expressions_from_nodes(&partition_by, expr_arena, schema, state)?;
            let mut apply_columns = aexpr_to_leaf_names(function, expr_arena);
            // sort and then dedup removes consecutive duplicates == all duplicates
            apply_columns.sort();
            apply_columns.dedup();

            if apply_columns.is_empty() {
                if has_aexpr(function, expr_arena, |e| matches!(e, AExpr::Literal(_))) {
                    apply_columns.push(get_literal_name())
                } else if has_aexpr(function, expr_arena, |e| matches!(e, AExpr::Len)) {
                    apply_columns.push(PlSmallStr::from_static("len"))
                } else if has_aexpr(function, expr_arena, |e| matches!(e, AExpr::Element)) {
                    apply_columns.push(PlSmallStr::from_static("element"))
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
                mapping,
                expr,
                has_different_group_sources,
                output_field,

                order_by_is_elementwise,
                all_group_by_are_elementwise,
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
            let lhs = create_physical_expr_inner(left, expr_arena, schema, state)?;
            let rhs = create_physical_expr_inner(right, expr_arena, schema, state)?;
            Ok(Arc::new(phys_expr::BinaryExpr::new(
                lhs,
                op,
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
        Element => {
            let output_field = expr_arena
                .get(expression)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;

            Ok(Arc::new(ElementExpr::new(output_field)))
        },
        #[cfg(feature = "dtype-struct")]
        StructField(field) => {
            let output_field = expr_arena
                .get(expression)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;

            Ok(Arc::new(FieldExpr::new(
                field.clone(),
                node_to_expr(expression, expr_arena),
                output_field,
            )))
        },
        Sort { expr, options } => {
            let phys_expr = create_physical_expr_inner(expr, expr_arena, schema, state)?;
            Ok(Arc::new(SortExpr::new(
                phys_expr,
                options,
                node_to_expr(expression, expr_arena),
            )))
        },
        Gather {
            expr,
            idx,
            returns_scalar,
            null_on_oob,
        } => {
            let phys_expr = create_physical_expr_inner(expr, expr_arena, schema, state)?;
            let phys_idx = create_physical_expr_inner(idx, expr_arena, schema, state)?;
            Ok(Arc::new(GatherExpr {
                phys_expr,
                idx: phys_idx,
                expr: node_to_expr(expression, expr_arena),
                returns_scalar,
                null_on_oob,
            }))
        },
        SortBy {
            expr,
            by,
            sort_options,
        } => {
            let phys_expr = create_physical_expr_inner(expr, expr_arena, schema, state)?;
            let phys_by = create_physical_expressions_from_nodes(&by, expr_arena, schema, state)?;
            Ok(Arc::new(SortByExpr::new(
                phys_expr,
                phys_by,
                node_to_expr(expression, expr_arena),
                sort_options.clone(),
            )))
        },
        Filter { input, by } => {
            let phys_input = create_physical_expr_inner(input, expr_arena, schema, state)?;
            let phys_by = create_physical_expr_inner(by, expr_arena, schema, state)?;
            Ok(Arc::new(FilterExpr::new(
                phys_input,
                phys_by,
                node_to_expr(expression, expr_arena),
            )))
        },
        Agg(agg) => {
            let expr = agg.get_input().first();
            let input = create_physical_expr_inner(expr, expr_arena, schema, state)?;
            let allow_threading = state.allow_threading;

            let output_field = expr_arena
                .get(expression)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;

            // Special case: Quantile supports multiple inputs.
            // TODO refactor to FunctionExpr.
            if let IRAggExpr::Quantile {
                quantile, method, ..
            } = agg
            {
                let quantile = create_physical_expr_inner(quantile, expr_arena, schema, state)?;
                return Ok(Arc::new(AggQuantileExpr::new(input, quantile, method)));
            }

            let groupby = GroupByMethod::from(agg.clone());

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
        Function {
            input,
            function: function @ (IRFunctionExpr::ArgMin | IRFunctionExpr::ArgMax),
            options: _,
        } => {
            let phys_input =
                create_physical_expr_inner(input[0].node(), expr_arena, schema, state)?;

            let mut output_field = expr_arena
                .get(expression)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;
            output_field = Field::new(output_field.name().clone(), IDX_DTYPE.clone());

            let groupby = match function {
                IRFunctionExpr::ArgMin => GroupByMethod::ArgMin,
                IRFunctionExpr::ArgMax => GroupByMethod::ArgMax,
                _ => unreachable!(), // guaranteed by pattern
            };

            let agg_type = AggregationType {
                groupby,
                allow_threading: state.allow_threading,
            };

            Ok(Arc::new(AggregationExpr::new(
                phys_input,
                agg_type,
                output_field,
            )))
        },
        Function {
            input: inputs,
            function: function @ (IRFunctionExpr::MinBy | IRFunctionExpr::MaxBy),
            options: _,
        } => {
            assert!(inputs.len() == 2);
            let new_minmax_by = match function {
                IRFunctionExpr::MinBy => AggMinMaxByExpr::new_min_by,
                IRFunctionExpr::MaxBy => AggMinMaxByExpr::new_max_by,
                _ => unreachable!(), // guaranteed by pattern
            };
            let input = create_physical_expr_inner(inputs[0].node(), expr_arena, schema, state)?;
            let by = create_physical_expr_inner(inputs[1].node(), expr_arena, schema, state)?;
            return Ok(Arc::new(new_minmax_by(input, by)));
        },
        Cast {
            expr,
            dtype,
            options,
        } => {
            let phys_expr = create_physical_expr_inner(expr, expr_arena, schema, state)?;
            Ok(Arc::new(CastExpr {
                input: phys_expr,
                dtype: dtype.clone(),
                expr: node_to_expr(expression, expr_arena),
                options,
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
            let predicate = create_physical_expr_inner(predicate, expr_arena, schema, state)?;
            lit_count += state.local.has_lit as u8;
            state.reset();
            let truthy = create_physical_expr_inner(truthy, expr_arena, schema, state)?;
            lit_count += state.local.has_lit as u8;
            state.reset();
            let falsy = create_physical_expr_inner(falsy, expr_arena, schema, state)?;
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
        AExpr::AnonymousAgg {
            input,
            fmt_str: _,
            function,
        } => {
            let output_field = expr_arena
                .get(expression)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;

            let inputs = create_physical_expressions_from_irs(&input, expr_arena, schema, state)?;
            let grouped_reduction = function
                .clone()
                .materialize()?
                .as_any()
                .downcast_ref::<Box<dyn GroupedReduction>>()
                .unwrap()
                .new_empty();

            Ok(Arc::new(AnonymousAggregationExpr::new(
                inputs,
                grouped_reduction,
                output_field,
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

            let input = create_physical_expressions_from_irs(&input, expr_arena, schema, state)?;

            let function = function.clone().materialize()?;
            let function = function.into_inner().as_column_udf();

            Ok(Arc::new(ApplyExpr::new(
                input,
                SpecialEq::new(function),
                None,
                node_to_expr(expression, expr_arena),
                options,
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
            let evaluation_is_scalar = is_scalar_ae(evaluation, expr_arena);
            let evaluation_is_elementwise = is_elementwise_rec(evaluation, expr_arena);
            // @NOTE: This is actually also something the downstream apply code should care about.
            let mut pd_group = ExprPushdownGroup::Pushable;
            pd_group.update_with_expr_rec(expr_arena.get(evaluation), expr_arena, None);
            let evaluation_is_fallible = matches!(pd_group, ExprPushdownGroup::Fallible);

            let output_field = expr_arena
                .get(expression)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;
            let input_field = expr_arena
                .get(expr)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;
            let expr = create_physical_expr_inner(expr, expr_arena, schema, state)?;

            let element_dtype = variant.element_dtype(&input_field.dtype)?;
            let mut eval_schema = schema.as_ref().clone();
            eval_schema.insert(get_pl_element_name(), element_dtype.clone());
            let evaluation =
                create_physical_expr_inner(evaluation, expr_arena, &Arc::new(eval_schema), state)?;

            Ok(Arc::new(EvalExpr::new(
                expr,
                evaluation,
                variant,
                node_to_expr(expression, expr_arena),
                output_field,
                is_scalar,
                evaluation_is_scalar,
                evaluation_is_elementwise,
                evaluation_is_fallible,
            )))
        },
        #[cfg(feature = "dtype-struct")]
        StructEval { expr, evaluation } => {
            let is_scalar = is_scalar_ae(expression, expr_arena);
            let output_field = expr_arena
                .get(expression)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;
            let input_field = expr_arena
                .get(expr)
                .to_field(&ToFieldContext::new(expr_arena, schema))?;

            let input = create_physical_expr_inner(expr, expr_arena, schema, state)?;

            let mut eval_schema = schema.as_ref().clone();
            eval_schema.insert(get_pl_structfields_name(), input_field.dtype().clone());
            let eval_schema = Arc::new(eval_schema);

            let evaluation = evaluation
                .iter()
                .map(|e| create_physical_expr(e, expr_arena, &eval_schema, state))
                .collect::<PolarsResult<Vec<_>>>()?;

            Ok(Arc::new(StructEvalExpr::new(
                input,
                evaluation,
                node_to_expr(expression, expr_arena),
                output_field,
                is_scalar,
                state.allow_threading,
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

            let input = create_physical_expressions_from_irs(&input, expr_arena, schema, state)?;
            let is_fallible = expr_arena.get(expression).is_fallible_top_level(expr_arena);

            Ok(Arc::new(ApplyExpr::new(
                input,
                function_expr_to_udf(function.clone()),
                function_expr_to_groups_udf(&function),
                node_to_expr(expression, expr_arena),
                options,
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
            let input = create_physical_expr_inner(input, expr_arena, schema, state)?;
            let offset = create_physical_expr_inner(offset, expr_arena, schema, state)?;
            let length = create_physical_expr_inner(length, expr_arena, schema, state)?;
            Ok(Arc::new(SliceExpr {
                input,
                offset,
                length,
                expr: node_to_expr(expression, expr_arena),
            }))
        },
        Explode { expr, options } => {
            let input = create_physical_expr_inner(expr, expr_arena, schema, state)?;
            let function = SpecialEq::new(Arc::new(
                move |c: &mut [polars_core::frame::column::Column]| c[0].explode(options),
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
    }
}
