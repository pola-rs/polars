use polars_core::prelude::*;
use polars_utils::vec::ConvertVec;

use crate::constants::get_len_name;
use crate::prelude::*;

pub fn to_expr_ir(expr: Expr, arena: &mut Arena<AExpr>) -> ExprIR {
    let mut state = ConversionState::new();
    let node = to_aexpr_impl(expr, arena, &mut state);
    ExprIR::new(node, state.output_name)
}

fn to_expr_irs(input: Vec<Expr>, arena: &mut Arena<AExpr>) -> Vec<ExprIR> {
    input.convert_owned(|e| to_expr_ir(e, arena))
}

pub fn to_expr_ir_ignore_alias(expr: Expr, arena: &mut Arena<AExpr>) -> ExprIR {
    let mut state = ConversionState::new();
    state.ignore_alias = true;
    let node = to_aexpr_impl(expr, arena, &mut state);
    ExprIR::new(node, state.output_name)
}

fn to_expr_irs_ignore_alias(input: Vec<Expr>, arena: &mut Arena<AExpr>) -> Vec<ExprIR> {
    input.convert_owned(|e| to_expr_ir_ignore_alias(e, arena))
}

/// converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation
pub fn to_aexpr(expr: Expr, arena: &mut Arena<AExpr>) -> Node {
    to_aexpr_impl(
        expr,
        arena,
        &mut ConversionState {
            prune_alias: false,
            ..Default::default()
        },
    )
}

#[derive(Default)]
struct ConversionState {
    output_name: OutputName,
    /// Remove alias from the expressions and set as [`OutputName`].
    prune_alias: bool,
    /// If an `alias` is encountered prune and ignore it.
    ignore_alias: bool,
}

impl ConversionState {
    fn new() -> Self {
        Self {
            prune_alias: true,
            ..Default::default()
        }
    }
}

fn to_aexprs(input: Vec<Expr>, arena: &mut Arena<AExpr>, state: &mut ConversionState) -> Vec<Node> {
    input
        .into_iter()
        .map(|e| to_aexpr_impl(e, arena, state))
        .collect()
}

/// converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation
fn to_aexpr_impl(expr: Expr, arena: &mut Arena<AExpr>, state: &mut ConversionState) -> Node {
    let v = match expr {
        Expr::Explode(expr) => AExpr::Explode(to_aexpr_impl(*expr, arena, state)),
        Expr::Alias(e, name) => {
            if state.prune_alias {
                if state.output_name.is_none() && !state.ignore_alias {
                    state.output_name = OutputName::Alias(name);
                }
                to_aexpr_impl(*e, arena, state);
                arena.pop().unwrap()
            } else {
                AExpr::Alias(to_aexpr_impl(*e, arena, state), name)
            }
        },
        Expr::Literal(lv) => {
            if state.output_name.is_none() {
                state.output_name = OutputName::LiteralLhs(lv.output_column_name());
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
            let l = to_aexpr_impl(*left, arena, state);
            let r = to_aexpr_impl(*right, arena, state);
            AExpr::BinaryExpr {
                left: l,
                op,
                right: r,
            }
        },
        Expr::Cast {
            expr,
            data_type,
            strict,
        } => AExpr::Cast {
            expr: to_aexpr_impl(*expr, arena, state),
            data_type,
            strict,
        },
        Expr::Gather {
            expr,
            idx,
            returns_scalar,
        } => AExpr::Gather {
            expr: to_aexpr_impl(*expr, arena, state),
            idx: to_aexpr_impl(*idx, arena, state),
            returns_scalar,
        },
        Expr::Sort { expr, options } => AExpr::Sort {
            expr: to_aexpr_impl(*expr, arena, state),
            options,
        },
        Expr::SortBy {
            expr,
            by,
            descending,
        } => AExpr::SortBy {
            expr: to_aexpr_impl(*expr, arena, state),
            by: by
                .into_iter()
                .map(|e| to_aexpr_impl(e, arena, state))
                .collect(),
            descending,
        },
        Expr::Filter { input, by } => AExpr::Filter {
            input: to_aexpr_impl(*input, arena, state),
            by: to_aexpr_impl(*by, arena, state),
        },
        Expr::Agg(agg) => {
            let a_agg = match agg {
                AggExpr::Min {
                    input,
                    propagate_nans,
                } => AAggExpr::Min {
                    input: to_aexpr_impl(*input, arena, state),
                    propagate_nans,
                },
                AggExpr::Max {
                    input,
                    propagate_nans,
                } => AAggExpr::Max {
                    input: to_aexpr_impl(*input, arena, state),
                    propagate_nans,
                },
                AggExpr::Median(expr) => AAggExpr::Median(to_aexpr_impl(*expr, arena, state)),
                AggExpr::NUnique(expr) => AAggExpr::NUnique(to_aexpr_impl(*expr, arena, state)),
                AggExpr::First(expr) => AAggExpr::First(to_aexpr_impl(*expr, arena, state)),
                AggExpr::Last(expr) => AAggExpr::Last(to_aexpr_impl(*expr, arena, state)),
                AggExpr::Mean(expr) => AAggExpr::Mean(to_aexpr_impl(*expr, arena, state)),
                AggExpr::Implode(expr) => AAggExpr::Implode(to_aexpr_impl(*expr, arena, state)),
                AggExpr::Count(expr, include_nulls) => {
                    AAggExpr::Count(to_aexpr_impl(*expr, arena, state), include_nulls)
                },
                AggExpr::Quantile {
                    expr,
                    quantile,
                    interpol,
                } => AAggExpr::Quantile {
                    expr: to_aexpr_impl(*expr, arena, state),
                    quantile: to_aexpr_impl(*quantile, arena, state),
                    interpol,
                },
                AggExpr::Sum(expr) => AAggExpr::Sum(to_aexpr_impl(*expr, arena, state)),
                AggExpr::Std(expr, ddof) => AAggExpr::Std(to_aexpr_impl(*expr, arena, state), ddof),
                AggExpr::Var(expr, ddof) => AAggExpr::Var(to_aexpr_impl(*expr, arena, state), ddof),
                AggExpr::AggGroups(expr) => AAggExpr::AggGroups(to_aexpr_impl(*expr, arena, state)),
            };
            AExpr::Agg(a_agg)
        },
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            // Truthy must be resolved first to get the lhs name first set.
            let t = to_aexpr_impl(*truthy, arena, state);
            let p = to_aexpr_impl(*predicate, arena, state);
            let f = to_aexpr_impl(*falsy, arena, state);
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
        } => AExpr::AnonymousFunction {
            input: to_aexprs(input, arena, state),
            function,
            output_type,
            options,
        },
        Expr::Function {
            input,
            function,
            options,
        } => {
            match function {
                #[cfg(feature = "dtype-struct")]
                FunctionExpr::AsStruct => {
                    state.prune_alias = false;
                },
                _ => {},
            }
            AExpr::Function {
                input: to_aexprs(input, arena, state),
                function,
                options,
            }
        },
        Expr::Window {
            function,
            partition_by,
            options,
        } => AExpr::Window {
            function: to_aexpr_impl(*function, arena, state),
            partition_by: to_aexprs(partition_by, arena, state),
            options,
        },
        Expr::Slice {
            input,
            offset,
            length,
        } => AExpr::Slice {
            input: to_aexpr_impl(*input, arena, state),
            offset: to_aexpr_impl(*offset, arena, state),
            length: to_aexpr_impl(*length, arena, state),
        },
        Expr::Len => {
            if state.output_name.is_none() {
                state.output_name = OutputName::LiteralLhs(get_len_name())
            }
            AExpr::Len
        },
        Expr::Nth(i) => AExpr::Nth(i),
        Expr::Wildcard => AExpr::Wildcard,
        Expr::SubPlan { .. } => panic!("no SQLSubquery expected at this point"),
        Expr::KeepName(_) => panic!("no `name.keep` expected at this point"),
        Expr::Exclude(_, _) => panic!("no `exclude` expected at this point"),
        Expr::RenameAlias { .. } => panic!("no `rename_alias` expected at this point"),
        Expr::Columns { .. } => panic!("no `columns` expected at this point"),
        Expr::DtypeColumn { .. } => panic!("no `dtype-columns` expected at this point"),
        Expr::Selector(_) => panic!("no `selector` expected at this point"),
    };
    arena.add(v)
}

/// converts LogicalPlan to ALogicalPlan
/// it adds expressions & lps to the respective arenas as it traverses the plan
/// finally it returns the top node of the logical plan
pub fn to_alp(
    lp: LogicalPlan,
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<ALogicalPlan>,
) -> PolarsResult<Node> {
    let v = match lp {
        LogicalPlan::Scan {
            file_info,
            paths,
            predicate,
            scan_type,
            file_options: options,
        } => ALogicalPlan::Scan {
            file_info,
            paths,
            output_schema: None,
            predicate: predicate.map(|expr| to_expr_ir(expr, expr_arena)),
            scan_type,
            file_options: options,
        },
        #[cfg(feature = "python")]
        LogicalPlan::PythonScan { options } => ALogicalPlan::PythonScan {
            options,
            predicate: None,
        },
        LogicalPlan::Union { inputs, options } => {
            let inputs = inputs
                .into_iter()
                .map(|lp| to_alp(lp, expr_arena, lp_arena))
                .collect::<PolarsResult<_>>()?;
            ALogicalPlan::Union { inputs, options }
        },
        LogicalPlan::HConcat {
            inputs,
            schema,
            options,
        } => {
            let inputs = inputs
                .into_iter()
                .map(|lp| to_alp(lp, expr_arena, lp_arena))
                .collect::<PolarsResult<_>>()?;
            ALogicalPlan::HConcat {
                inputs,
                schema,
                options,
            }
        },
        LogicalPlan::Selection { input, predicate } => {
            let i = to_alp(*input, expr_arena, lp_arena)?;
            let p = to_expr_ir(predicate, expr_arena);
            ALogicalPlan::Selection {
                input: i,
                predicate: p,
            }
        },
        LogicalPlan::Slice { input, offset, len } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::Slice { input, offset, len }
        },
        LogicalPlan::DataFrameScan {
            df,
            schema,
            output_schema,
            projection,
            selection,
        } => ALogicalPlan::DataFrameScan {
            df,
            schema,
            output_schema,
            projection,
            selection: selection.map(|expr| to_expr_ir(expr, expr_arena)),
        },
        LogicalPlan::Projection {
            expr,
            input,
            schema,
            options,
        } => {
            let eirs = to_expr_irs(expr, expr_arena);
            let expr = eirs.into();
            let i = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::Projection {
                expr,
                input: i,
                schema,
                options,
            }
        },
        LogicalPlan::Sort {
            input,
            by_column,
            args,
        } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            let by_column = to_expr_irs(by_column, expr_arena);
            ALogicalPlan::Sort {
                input,
                by_column,
                args,
            }
        },
        LogicalPlan::Cache {
            input,
            id,
            cache_hits,
        } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::Cache {
                input,
                id,
                cache_hits,
            }
        },
        LogicalPlan::Aggregate {
            input,
            keys,
            aggs,
            schema,
            apply,
            maintain_order,
            options,
        } => {
            let i = to_alp(*input, expr_arena, lp_arena)?;
            let aggs = to_expr_irs(aggs, expr_arena);
            let keys = keys.convert(|e| to_expr_ir(e.clone(), expr_arena));

            ALogicalPlan::Aggregate {
                input: i,
                keys,
                aggs,
                schema,
                apply,
                maintain_order,
                options,
            }
        },
        LogicalPlan::Join {
            input_left,
            input_right,
            schema,
            left_on,
            right_on,
            options,
        } => {
            let input_left = to_alp(*input_left, expr_arena, lp_arena)?;
            let input_right = to_alp(*input_right, expr_arena, lp_arena)?;

            let left_on = to_expr_irs_ignore_alias(left_on, expr_arena);
            let right_on = to_expr_irs_ignore_alias(right_on, expr_arena);

            ALogicalPlan::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            }
        },
        LogicalPlan::HStack {
            input,
            exprs,
            schema,
            options,
        } => {
            let eirs = to_expr_irs(exprs, expr_arena);
            let exprs = eirs.into();
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::HStack {
                input,
                exprs,
                schema,
                options,
            }
        },
        LogicalPlan::Distinct { input, options } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::Distinct { input, options }
        },
        LogicalPlan::MapFunction { input, function } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::MapFunction { input, function }
        },
        LogicalPlan::Error { err, .. } => {
            // We just take the error. The LogicalPlan should not be used anymore once this
            // is taken.
            return Err(err.take());
        },
        LogicalPlan::ExtContext {
            input,
            contexts,
            schema,
        } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            let contexts = contexts
                .into_iter()
                .map(|lp| to_alp(lp, expr_arena, lp_arena))
                .collect::<PolarsResult<_>>()?;
            ALogicalPlan::ExtContext {
                input,
                contexts,
                schema,
            }
        },
        LogicalPlan::Sink { input, payload } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::Sink { input, payload }
        },
    };
    Ok(lp_arena.add(v))
}

/// converts a node from the AExpr arena to Expr
pub fn node_to_expr(node: Node, expr_arena: &Arena<AExpr>) -> Expr {
    let expr = expr_arena.get(node).clone();

    match expr {
        AExpr::Explode(node) => Expr::Explode(Box::new(node_to_expr(node, expr_arena))),
        AExpr::Alias(expr, name) => {
            let exp = node_to_expr(expr, expr_arena);
            Expr::Alias(Box::new(exp), name)
        },
        AExpr::Column(a) => Expr::Column(a),
        AExpr::Literal(s) => Expr::Literal(s),
        AExpr::BinaryExpr { left, op, right } => {
            let l = node_to_expr(left, expr_arena);
            let r = node_to_expr(right, expr_arena);
            Expr::BinaryExpr {
                left: Box::new(l),
                op,
                right: Box::new(r),
            }
        },
        AExpr::Cast {
            expr,
            data_type,
            strict,
        } => {
            let exp = node_to_expr(expr, expr_arena);
            Expr::Cast {
                expr: Box::new(exp),
                data_type,
                strict,
            }
        },
        AExpr::Sort { expr, options } => {
            let exp = node_to_expr(expr, expr_arena);
            Expr::Sort {
                expr: Box::new(exp),
                options,
            }
        },
        AExpr::Gather {
            expr,
            idx,
            returns_scalar,
        } => {
            let expr = node_to_expr(expr, expr_arena);
            let idx = node_to_expr(idx, expr_arena);
            Expr::Gather {
                expr: Box::new(expr),
                idx: Box::new(idx),
                returns_scalar,
            }
        },
        AExpr::SortBy {
            expr,
            by,
            descending,
        } => {
            let expr = node_to_expr(expr, expr_arena);
            let by = by
                .iter()
                .map(|node| node_to_expr(*node, expr_arena))
                .collect();
            Expr::SortBy {
                expr: Box::new(expr),
                by,
                descending,
            }
        },
        AExpr::Filter { input, by } => {
            let input = node_to_expr(input, expr_arena);
            let by = node_to_expr(by, expr_arena);
            Expr::Filter {
                input: Box::new(input),
                by: Box::new(by),
            }
        },
        AExpr::Agg(agg) => match agg {
            AAggExpr::Min {
                input,
                propagate_nans,
            } => {
                let exp = node_to_expr(input, expr_arena);
                AggExpr::Min {
                    input: Box::new(exp),
                    propagate_nans,
                }
                .into()
            },
            AAggExpr::Max {
                input,
                propagate_nans,
            } => {
                let exp = node_to_expr(input, expr_arena);
                AggExpr::Max {
                    input: Box::new(exp),
                    propagate_nans,
                }
                .into()
            },

            AAggExpr::Median(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Median(Box::new(exp)).into()
            },
            AAggExpr::NUnique(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::NUnique(Box::new(exp)).into()
            },
            AAggExpr::First(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::First(Box::new(exp)).into()
            },
            AAggExpr::Last(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Last(Box::new(exp)).into()
            },
            AAggExpr::Mean(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Mean(Box::new(exp)).into()
            },
            AAggExpr::Implode(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Implode(Box::new(exp)).into()
            },
            AAggExpr::Quantile {
                expr,
                quantile,
                interpol,
            } => {
                let expr = node_to_expr(expr, expr_arena);
                let quantile = node_to_expr(quantile, expr_arena);
                AggExpr::Quantile {
                    expr: Box::new(expr),
                    quantile: Box::new(quantile),
                    interpol,
                }
                .into()
            },
            AAggExpr::Sum(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Sum(Box::new(exp)).into()
            },
            AAggExpr::Std(expr, ddof) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Std(Box::new(exp), ddof).into()
            },
            AAggExpr::Var(expr, ddof) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Var(Box::new(exp), ddof).into()
            },
            AAggExpr::AggGroups(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::AggGroups(Box::new(exp)).into()
            },
            AAggExpr::Count(expr, include_nulls) => {
                let expr = node_to_expr(expr, expr_arena);
                AggExpr::Count(Box::new(expr), include_nulls).into()
            },
        },
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let p = node_to_expr(predicate, expr_arena);
            let t = node_to_expr(truthy, expr_arena);
            let f = node_to_expr(falsy, expr_arena);

            Expr::Ternary {
                predicate: Box::new(p),
                truthy: Box::new(t),
                falsy: Box::new(f),
            }
        },
        AExpr::AnonymousFunction {
            input,
            function,
            output_type,
            options,
        } => Expr::AnonymousFunction {
            input: nodes_to_exprs(&input, expr_arena),
            function,
            output_type,
            options,
        },
        AExpr::Function {
            input,
            function,
            options,
        } => Expr::Function {
            input: nodes_to_exprs(&input, expr_arena),
            function,
            options,
        },
        AExpr::Window {
            function,
            partition_by,
            options,
        } => {
            let function = Box::new(node_to_expr(function, expr_arena));
            let partition_by = nodes_to_exprs(&partition_by, expr_arena);
            Expr::Window {
                function,
                partition_by,
                options,
            }
        },
        AExpr::Slice {
            input,
            offset,
            length,
        } => Expr::Slice {
            input: Box::new(node_to_expr(input, expr_arena)),
            offset: Box::new(node_to_expr(offset, expr_arena)),
            length: Box::new(node_to_expr(length, expr_arena)),
        },
        AExpr::Len => Expr::Len,
        AExpr::Nth(i) => Expr::Nth(i),
        AExpr::Wildcard => Expr::Wildcard,
    }
}

fn nodes_to_exprs(nodes: &[Node], expr_arena: &Arena<AExpr>) -> Vec<Expr> {
    nodes.iter().map(|n| node_to_expr(*n, expr_arena)).collect()
}

fn expr_irs_to_exprs(expr_irs: Vec<ExprIR>, expr_arena: &Arena<AExpr>) -> Vec<Expr> {
    expr_irs.convert_owned(|e| e.to_expr(expr_arena))
}

impl ALogicalPlan {
    fn into_lp<F, LPA>(
        self,
        conversion_fn: &F,
        lp_arena: &mut LPA,
        expr_arena: &Arena<AExpr>,
    ) -> LogicalPlan
    where
        F: Fn(Node, &mut LPA) -> ALogicalPlan,
    {
        let lp = self;
        let convert_to_lp = |node: Node, lp_arena: &mut LPA| {
            conversion_fn(node, lp_arena).into_lp(conversion_fn, lp_arena, expr_arena)
        };
        match lp {
            ALogicalPlan::Scan {
                paths,
                file_info,
                predicate,
                scan_type,
                output_schema: _,
                file_options: options,
            } => LogicalPlan::Scan {
                paths,
                file_info,
                predicate: predicate.map(|e| e.to_expr(expr_arena)),
                scan_type,
                file_options: options,
            },
            #[cfg(feature = "python")]
            ALogicalPlan::PythonScan { options, .. } => LogicalPlan::PythonScan { options },
            ALogicalPlan::Union { inputs, options } => {
                let inputs = inputs
                    .into_iter()
                    .map(|node| convert_to_lp(node, lp_arena))
                    .collect();
                LogicalPlan::Union { inputs, options }
            },
            ALogicalPlan::HConcat {
                inputs,
                schema,
                options,
            } => {
                let inputs = inputs
                    .into_iter()
                    .map(|node| convert_to_lp(node, lp_arena))
                    .collect();
                LogicalPlan::HConcat {
                    inputs,
                    schema: schema.clone(),
                    options,
                }
            },
            ALogicalPlan::Slice { input, offset, len } => {
                let lp = convert_to_lp(input, lp_arena);
                LogicalPlan::Slice {
                    input: Box::new(lp),
                    offset,
                    len,
                }
            },
            ALogicalPlan::Selection { input, predicate } => {
                let lp = convert_to_lp(input, lp_arena);
                let predicate = predicate.to_expr(expr_arena);
                LogicalPlan::Selection {
                    input: Box::new(lp),
                    predicate,
                }
            },
            ALogicalPlan::DataFrameScan {
                df,
                schema,
                output_schema,
                projection,
                selection,
            } => LogicalPlan::DataFrameScan {
                df,
                schema,
                output_schema,
                projection,
                selection: selection.map(|e| e.to_expr(expr_arena)),
            },
            ALogicalPlan::Projection {
                expr,
                input,
                schema,
                options,
            } => {
                let i = convert_to_lp(input, lp_arena);
                let expr = expr_irs_to_exprs(expr.all_exprs(), expr_arena);
                LogicalPlan::Projection {
                    expr,
                    input: Box::new(i),
                    schema,
                    options,
                }
            },
            ALogicalPlan::SimpleProjection { input, columns, .. } => {
                let input = convert_to_lp(input, lp_arena);
                let expr = columns
                    .iter_names()
                    .map(|name| Expr::Column(ColumnName::from(name.as_str())))
                    .collect::<Vec<_>>();
                LogicalPlan::Projection {
                    expr,
                    input: Box::new(input),
                    schema: columns.clone(),
                    options: Default::default(),
                }
            },
            ALogicalPlan::Sort {
                input,
                by_column,
                args,
            } => {
                let input = Box::new(convert_to_lp(input, lp_arena));
                let by_column = expr_irs_to_exprs(by_column, expr_arena);
                LogicalPlan::Sort {
                    input,
                    by_column,
                    args,
                }
            },
            ALogicalPlan::Cache {
                input,
                id,
                cache_hits,
            } => {
                let input = Box::new(convert_to_lp(input, lp_arena));
                LogicalPlan::Cache {
                    input,
                    id,
                    cache_hits,
                }
            },
            ALogicalPlan::Aggregate {
                input,
                keys,
                aggs,
                schema,
                apply,
                maintain_order,
                options: dynamic_options,
            } => {
                let i = convert_to_lp(input, lp_arena);
                let keys = Arc::new(expr_irs_to_exprs(keys, expr_arena));
                let aggs = expr_irs_to_exprs(aggs, expr_arena);

                LogicalPlan::Aggregate {
                    input: Box::new(i),
                    keys,
                    aggs,
                    schema,
                    apply,
                    maintain_order,
                    options: dynamic_options,
                }
            },
            ALogicalPlan::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            } => {
                let i_l = convert_to_lp(input_left, lp_arena);
                let i_r = convert_to_lp(input_right, lp_arena);

                let left_on = expr_irs_to_exprs(left_on, expr_arena);
                let right_on = expr_irs_to_exprs(right_on, expr_arena);

                LogicalPlan::Join {
                    input_left: Box::new(i_l),
                    input_right: Box::new(i_r),
                    schema,
                    left_on,
                    right_on,
                    options,
                }
            },
            ALogicalPlan::HStack {
                input,
                exprs,
                schema,
                options,
            } => {
                let i = convert_to_lp(input, lp_arena);
                let exprs = expr_irs_to_exprs(exprs.all_exprs(), expr_arena);

                LogicalPlan::HStack {
                    input: Box::new(i),
                    exprs,
                    schema,
                    options,
                }
            },
            ALogicalPlan::Distinct { input, options } => {
                let i = convert_to_lp(input, lp_arena);
                LogicalPlan::Distinct {
                    input: Box::new(i),
                    options,
                }
            },
            ALogicalPlan::MapFunction { input, function } => {
                let input = Box::new(convert_to_lp(input, lp_arena));
                LogicalPlan::MapFunction { input, function }
            },
            ALogicalPlan::ExtContext {
                input,
                contexts,
                schema,
            } => {
                let input = Box::new(convert_to_lp(input, lp_arena));
                let contexts = contexts
                    .into_iter()
                    .map(|node| convert_to_lp(node, lp_arena))
                    .collect();
                LogicalPlan::ExtContext {
                    input,
                    contexts,
                    schema,
                }
            },
            ALogicalPlan::Sink { input, payload } => {
                let input = Box::new(convert_to_lp(input, lp_arena));
                LogicalPlan::Sink { input, payload }
            },
            ALogicalPlan::Invalid => unreachable!(),
        }
    }
}

pub fn node_to_lp_cloned(
    node: Node,
    expr_arena: &Arena<AExpr>,
    mut lp_arena: &Arena<ALogicalPlan>,
) -> LogicalPlan {
    // we borrow again mutably only to make the types happy
    // we want to initialize `to_lp` from a mutable and a immutable lp_arena
    // by borrowing an immutable mutably, we still are immutable down the line.
    let alp = lp_arena.get(node).clone();
    alp.into_lp(
        &|node, lp_arena: &mut &Arena<ALogicalPlan>| lp_arena.get(node).clone(),
        &mut lp_arena,
        expr_arena,
    )
}

/// converts a node from the ALogicalPlan arena to a LogicalPlan
pub fn node_to_lp(
    node: Node,
    expr_arena: &Arena<AExpr>,
    lp_arena: &mut Arena<ALogicalPlan>,
) -> LogicalPlan {
    let alp = lp_arena.get_mut(node);
    let alp = std::mem::take(alp);
    alp.into_lp(
        &|node, lp_arena: &mut Arena<ALogicalPlan>| {
            let lp = lp_arena.get_mut(node);
            std::mem::take(lp)
        },
        lp_arena,
        expr_arena,
    )
}
