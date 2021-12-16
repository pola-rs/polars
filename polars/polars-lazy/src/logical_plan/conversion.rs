use crate::prelude::*;
use polars_core::prelude::*;

fn to_aexprs(input: Vec<Expr>, arena: &mut Arena<AExpr>) -> Vec<Node> {
    input.into_iter().map(|e| to_aexpr(e, arena)).collect()
}

// converts expression to AExpr, which uses an arena (Vec) for allocation
pub(crate) fn to_aexpr(expr: Expr, arena: &mut Arena<AExpr>) -> Node {
    let v = match expr {
        Expr::IsUnique(expr) => AExpr::IsUnique(to_aexpr(*expr, arena)),
        Expr::Duplicated(expr) => AExpr::Duplicated(to_aexpr(*expr, arena)),
        Expr::Reverse(expr) => AExpr::Reverse(to_aexpr(*expr, arena)),
        Expr::Explode(expr) => AExpr::Explode(to_aexpr(*expr, arena)),
        Expr::Alias(e, name) => AExpr::Alias(to_aexpr(*e, arena), name),
        Expr::Literal(value) => AExpr::Literal(value),
        Expr::Column(s) => AExpr::Column(s),
        Expr::BinaryExpr { left, op, right } => {
            let l = to_aexpr(*left, arena);
            let r = to_aexpr(*right, arena);
            AExpr::BinaryExpr {
                left: l,
                op,
                right: r,
            }
        }
        Expr::Not(e) => AExpr::Not(to_aexpr(*e, arena)),
        Expr::IsNotNull(e) => AExpr::IsNotNull(to_aexpr(*e, arena)),
        Expr::IsNull(e) => AExpr::IsNull(to_aexpr(*e, arena)),

        Expr::Cast {
            expr,
            data_type,
            strict,
        } => AExpr::Cast {
            expr: to_aexpr(*expr, arena),
            data_type,
            strict,
        },
        Expr::Take { expr, idx } => AExpr::Take {
            expr: to_aexpr(*expr, arena),
            idx: to_aexpr(*idx, arena),
        },
        Expr::Sort { expr, options } => AExpr::Sort {
            expr: to_aexpr(*expr, arena),
            options,
        },
        Expr::SortBy { expr, by, reverse } => AExpr::SortBy {
            expr: to_aexpr(*expr, arena),
            by: by.into_iter().map(|e| to_aexpr(e, arena)).collect(),
            reverse,
        },
        Expr::Filter { input, by } => AExpr::Filter {
            input: to_aexpr(*input, arena),
            by: to_aexpr(*by, arena),
        },
        Expr::Agg(agg) => {
            let a_agg = match agg {
                AggExpr::Min(expr) => AAggExpr::Min(to_aexpr(*expr, arena)),
                AggExpr::Max(expr) => AAggExpr::Max(to_aexpr(*expr, arena)),
                AggExpr::Median(expr) => AAggExpr::Median(to_aexpr(*expr, arena)),
                AggExpr::NUnique(expr) => AAggExpr::NUnique(to_aexpr(*expr, arena)),
                AggExpr::First(expr) => AAggExpr::First(to_aexpr(*expr, arena)),
                AggExpr::Last(expr) => AAggExpr::Last(to_aexpr(*expr, arena)),
                AggExpr::Mean(expr) => AAggExpr::Mean(to_aexpr(*expr, arena)),
                AggExpr::List(expr) => AAggExpr::List(to_aexpr(*expr, arena)),
                AggExpr::Count(expr) => AAggExpr::Count(to_aexpr(*expr, arena)),
                AggExpr::Quantile {
                    expr,
                    quantile,
                    interpol,
                } => AAggExpr::Quantile {
                    expr: to_aexpr(*expr, arena),
                    quantile,
                    interpol,
                },
                AggExpr::Sum(expr) => AAggExpr::Sum(to_aexpr(*expr, arena)),
                AggExpr::Std(expr) => AAggExpr::Std(to_aexpr(*expr, arena)),
                AggExpr::Var(expr) => AAggExpr::Var(to_aexpr(*expr, arena)),
                AggExpr::AggGroups(expr) => AAggExpr::AggGroups(to_aexpr(*expr, arena)),
            };
            AExpr::Agg(a_agg)
        }
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let p = to_aexpr(*predicate, arena);
            let t = to_aexpr(*truthy, arena);
            let f = to_aexpr(*falsy, arena);
            AExpr::Ternary {
                predicate: p,
                truthy: t,
                falsy: f,
            }
        }
        Expr::Function {
            input,
            function,
            output_type,
            options,
        } => AExpr::Function {
            input: to_aexprs(input, arena),
            function,
            output_type,
            options,
        },
        Expr::BinaryFunction {
            input_a,
            input_b,
            function,
            output_field,
        } => AExpr::BinaryFunction {
            input_a: to_aexpr(*input_a, arena),
            input_b: to_aexpr(*input_b, arena),
            function,
            output_field,
        },
        Expr::Shift { input, periods } => AExpr::Shift {
            input: to_aexpr(*input, arena),
            periods,
        },
        Expr::Window {
            function,
            partition_by,
            order_by,
            options,
        } => AExpr::Window {
            function: to_aexpr(*function, arena),
            partition_by: to_aexprs(partition_by, arena),
            order_by: order_by.map(|ob| to_aexpr(*ob, arena)),
            options,
        },
        Expr::Slice {
            input,
            offset,
            length,
        } => AExpr::Slice {
            input: to_aexpr(*input, arena),
            offset,
            length,
        },
        Expr::Wildcard => AExpr::Wildcard,
        Expr::KeepName(_) => panic!("no keep_name expected at this point"),
        Expr::Exclude(_, _) => panic!("no exclude expected at this point"),
        Expr::SufPreFix { .. } => panic!("no `suffix/prefix` expected at this point"),
        Expr::Columns { .. } => panic!("no `columns` expected at this point"),
        Expr::DtypeColumn { .. } => panic!("no `dtype-columns` expected at this point"),
    };
    arena.add(v)
}

pub(crate) fn to_alp(
    lp: LogicalPlan,
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<ALogicalPlan>,
) -> Node {
    let v = match lp {
        LogicalPlan::Union { inputs } => {
            let inputs = inputs
                .into_iter()
                .map(|lp| to_alp(lp, expr_arena, lp_arena))
                .collect();
            ALogicalPlan::Union { inputs }
        }
        LogicalPlan::Selection { input, predicate } => {
            let i = to_alp(*input, expr_arena, lp_arena);
            let p = to_aexpr(predicate, expr_arena);
            ALogicalPlan::Selection {
                input: i,
                predicate: p,
            }
        }
        LogicalPlan::Slice { input, offset, len } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Slice { input, offset, len }
        }
        LogicalPlan::Melt {
            input,
            id_vars,
            value_vars,
            schema,
        } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Melt {
                input,
                id_vars,
                value_vars,
                schema,
            }
        }
        #[cfg(feature = "csv-file")]
        LogicalPlan::CsvScan {
            path,
            schema,
            options,
            predicate,
            aggregate,
        } => ALogicalPlan::CsvScan {
            path,
            schema,
            output_schema: None,
            options,
            predicate: predicate.map(|expr| to_aexpr(expr, expr_arena)),
            aggregate: aggregate
                .into_iter()
                .map(|expr| to_aexpr(expr, expr_arena))
                .collect(),
        },
        #[cfg(feature = "ipc")]
        LogicalPlan::IpcScan {
            path,
            schema,
            predicate,
            aggregate,
            options,
        } => ALogicalPlan::IpcScan {
            path,
            schema,
            output_schema: None,
            predicate: predicate.map(|expr| to_aexpr(expr, expr_arena)),
            aggregate: aggregate
                .into_iter()
                .map(|expr| to_aexpr(expr, expr_arena))
                .collect(),
            options,
        },
        #[cfg(feature = "parquet")]
        LogicalPlan::ParquetScan {
            path,
            schema,
            with_columns,
            predicate,
            aggregate,
            n_rows,
            cache,
        } => ALogicalPlan::ParquetScan {
            path,
            schema,
            output_schema: None,
            with_columns,
            predicate: predicate.map(|expr| to_aexpr(expr, expr_arena)),
            aggregate: aggregate
                .into_iter()
                .map(|expr| to_aexpr(expr, expr_arena))
                .collect(),
            n_rows,
            cache,
        },
        LogicalPlan::DataFrameScan {
            df,
            schema,
            projection,
            selection,
        } => ALogicalPlan::DataFrameScan {
            df,
            schema,
            projection: projection
                .map(|exprs| exprs.into_iter().map(|x| to_aexpr(x, expr_arena)).collect()),
            selection: selection.map(|expr| to_aexpr(expr, expr_arena)),
        },
        LogicalPlan::Projection {
            expr,
            input,
            schema,
        } => {
            let exp = expr.into_iter().map(|x| to_aexpr(x, expr_arena)).collect();
            let i = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Projection {
                expr: exp,
                input: i,
                schema,
            }
        }
        LogicalPlan::LocalProjection {
            expr,
            input,
            schema,
        } => {
            let exp = expr.into_iter().map(|x| to_aexpr(x, expr_arena)).collect();
            let i = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::LocalProjection {
                expr: exp,
                input: i,
                schema,
            }
        }
        LogicalPlan::Sort {
            input,
            by_column,
            reverse,
        } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            let by_column = by_column
                .into_iter()
                .map(|x| to_aexpr(x, expr_arena))
                .collect();
            ALogicalPlan::Sort {
                input,
                by_column,
                reverse,
            }
        }
        LogicalPlan::Explode { input, columns } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Explode { input, columns }
        }
        LogicalPlan::Cache { input } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Cache { input }
        }
        LogicalPlan::Aggregate {
            input,
            keys,
            aggs,
            schema,
            apply,
            maintain_order,
            dynamic_options,
        } => {
            let i = to_alp(*input, expr_arena, lp_arena);
            let aggs_new = aggs.into_iter().map(|x| to_aexpr(x, expr_arena)).collect();
            let keys_new = keys
                .iter()
                .map(|x| to_aexpr(x.clone(), expr_arena))
                .collect();

            ALogicalPlan::Aggregate {
                input: i,
                keys: keys_new,
                aggs: aggs_new,
                schema,
                apply,
                maintain_order,
                dynamic_options,
            }
        }
        LogicalPlan::Join {
            input_left,
            input_right,
            schema,
            left_on,
            right_on,
            options,
        } => {
            let i_l = to_alp(*input_left, expr_arena, lp_arena);
            let i_r = to_alp(*input_right, expr_arena, lp_arena);

            let l_on = left_on
                .into_iter()
                .map(|x| to_aexpr(x, expr_arena))
                .collect();
            let r_on = right_on
                .into_iter()
                .map(|x| to_aexpr(x, expr_arena))
                .collect();

            ALogicalPlan::Join {
                input_left: i_l,
                input_right: i_r,
                schema,
                left_on: l_on,
                right_on: r_on,
                options,
            }
        }
        LogicalPlan::HStack {
            input,
            exprs,
            schema,
        } => {
            let exp = exprs.into_iter().map(|x| to_aexpr(x, expr_arena)).collect();
            let i = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::HStack {
                input: i,
                exprs: exp,
                schema,
            }
        }
        LogicalPlan::Distinct {
            input,
            maintain_order,
            subset,
        } => {
            let i = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Distinct {
                input: i,
                maintain_order,
                subset,
            }
        }
        LogicalPlan::Udf {
            input,
            function,
            projection_pd,
            predicate_pd,
            schema,
        } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Udf {
                input,
                function,
                projection_pd,
                predicate_pd,
                schema,
            }
        }
    };
    lp_arena.add(v)
}

pub(crate) fn node_to_exp(node: Node, expr_arena: &Arena<AExpr>) -> Expr {
    let expr = expr_arena.get(node).clone();

    match expr {
        AExpr::Duplicated(node) => Expr::Duplicated(Box::new(node_to_exp(node, expr_arena))),
        AExpr::IsUnique(node) => Expr::IsUnique(Box::new(node_to_exp(node, expr_arena))),
        AExpr::Reverse(node) => Expr::Reverse(Box::new(node_to_exp(node, expr_arena))),
        AExpr::Explode(node) => Expr::Explode(Box::new(node_to_exp(node, expr_arena))),
        AExpr::Alias(expr, name) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Alias(Box::new(exp), name)
        }
        AExpr::Column(a) => Expr::Column(a),
        AExpr::Literal(s) => Expr::Literal(s),
        AExpr::BinaryExpr { left, op, right } => {
            let l = node_to_exp(left, expr_arena);
            let r = node_to_exp(right, expr_arena);
            Expr::BinaryExpr {
                left: Box::new(l),
                op,
                right: Box::new(r),
            }
        }
        AExpr::Not(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Not(Box::new(exp))
        }
        AExpr::IsNotNull(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::IsNotNull(Box::new(exp))
        }
        AExpr::IsNull(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::IsNull(Box::new(exp))
        }
        AExpr::Cast {
            expr,
            data_type,
            strict,
        } => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Cast {
                expr: Box::new(exp),
                data_type,
                strict,
            }
        }
        AExpr::Sort { expr, options } => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Sort {
                expr: Box::new(exp),
                options,
            }
        }
        AExpr::Take { expr, idx } => {
            let expr = node_to_exp(expr, expr_arena);
            let idx = node_to_exp(idx, expr_arena);
            Expr::Take {
                expr: Box::new(expr),
                idx: Box::new(idx),
            }
        }
        AExpr::SortBy { expr, by, reverse } => {
            let expr = node_to_exp(expr, expr_arena);
            let by = by
                .iter()
                .map(|node| node_to_exp(*node, expr_arena))
                .collect();
            Expr::SortBy {
                expr: Box::new(expr),
                by,
                reverse,
            }
        }
        AExpr::Filter { input, by } => {
            let input = node_to_exp(input, expr_arena);
            let by = node_to_exp(by, expr_arena);
            Expr::Filter {
                input: Box::new(input),
                by: Box::new(by),
            }
        }
        AExpr::Agg(agg) => match agg {
            AAggExpr::Min(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Min(Box::new(exp)).into()
            }
            AAggExpr::Max(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Max(Box::new(exp)).into()
            }

            AAggExpr::Median(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Median(Box::new(exp)).into()
            }
            AAggExpr::NUnique(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::NUnique(Box::new(exp)).into()
            }
            AAggExpr::First(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::First(Box::new(exp)).into()
            }
            AAggExpr::Last(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Last(Box::new(exp)).into()
            }
            AAggExpr::Mean(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Mean(Box::new(exp)).into()
            }
            AAggExpr::List(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::List(Box::new(exp)).into()
            }
            AAggExpr::Quantile {
                expr,
                quantile,
                interpol,
            } => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Quantile {
                    expr: Box::new(exp),
                    quantile,
                    interpol,
                }
                .into()
            }
            AAggExpr::Sum(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Sum(Box::new(exp)).into()
            }
            AAggExpr::Std(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Std(Box::new(exp)).into()
            }
            AAggExpr::Var(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Var(Box::new(exp)).into()
            }
            AAggExpr::AggGroups(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::AggGroups(Box::new(exp)).into()
            }
            AAggExpr::Count(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Count(Box::new(exp)).into()
            }
        },
        AExpr::Shift { input, periods } => {
            let e = node_to_exp(input, expr_arena);
            Expr::Shift {
                input: Box::new(e),
                periods,
            }
        }
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let p = node_to_exp(predicate, expr_arena);
            let t = node_to_exp(truthy, expr_arena);
            let f = node_to_exp(falsy, expr_arena);

            Expr::Ternary {
                predicate: Box::new(p),
                truthy: Box::new(t),
                falsy: Box::new(f),
            }
        }
        AExpr::Function {
            input,
            function,
            output_type,
            options,
        } => Expr::Function {
            input: nodes_to_exprs(&input, expr_arena),
            function,
            output_type,
            options,
        },
        AExpr::BinaryFunction {
            input_a,
            input_b,
            function,
            output_field,
        } => Expr::BinaryFunction {
            input_a: Box::new(node_to_exp(input_a, expr_arena)),
            input_b: Box::new(node_to_exp(input_b, expr_arena)),
            function,
            output_field,
        },
        AExpr::Window {
            function,
            partition_by,
            order_by,
            options,
        } => {
            let function = Box::new(node_to_exp(function, expr_arena));
            let partition_by = nodes_to_exprs(&partition_by, expr_arena);
            let order_by = order_by.map(|ob| Box::new(node_to_exp(ob, expr_arena)));
            Expr::Window {
                function,
                partition_by,
                order_by,
                options,
            }
        }
        AExpr::Slice {
            input,
            offset,
            length,
        } => Expr::Slice {
            input: Box::new(node_to_exp(input, expr_arena)),
            offset,
            length,
        },
        AExpr::Wildcard => Expr::Wildcard,
    }
}

fn nodes_to_exprs(nodes: &[Node], expr_arena: &Arena<AExpr>) -> Vec<Expr> {
    nodes.iter().map(|n| node_to_exp(*n, expr_arena)).collect()
}

pub(crate) fn node_to_lp(
    node: Node,
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<ALogicalPlan>,
) -> LogicalPlan {
    let lp = lp_arena.get_mut(node);
    let lp = std::mem::take(lp);

    match lp {
        ALogicalPlan::Union { inputs } => {
            let inputs = inputs
                .into_iter()
                .map(|node| node_to_lp(node, expr_arena, lp_arena))
                .collect();
            LogicalPlan::Union { inputs }
        }
        ALogicalPlan::Slice { input, offset, len } => {
            let lp = node_to_lp(input, expr_arena, lp_arena);
            LogicalPlan::Slice {
                input: Box::new(lp),
                offset,
                len,
            }
        }
        ALogicalPlan::Selection { input, predicate } => {
            let lp = node_to_lp(input, expr_arena, lp_arena);
            let p = node_to_exp(predicate, expr_arena);
            LogicalPlan::Selection {
                input: Box::new(lp),
                predicate: p,
            }
        }
        #[cfg(feature = "csv-file")]
        ALogicalPlan::CsvScan {
            path,
            schema,
            output_schema: _,
            options,
            predicate,
            aggregate,
        } => LogicalPlan::CsvScan {
            path,
            schema,
            options,
            predicate: predicate.map(|n| node_to_exp(n, expr_arena)),
            aggregate: nodes_to_exprs(&aggregate, expr_arena),
        },
        #[cfg(feature = "ipc")]
        ALogicalPlan::IpcScan {
            path,
            schema,
            output_schema: _,
            predicate,
            aggregate,
            options,
        } => LogicalPlan::IpcScan {
            path,
            schema,
            predicate: predicate.map(|n| node_to_exp(n, expr_arena)),
            aggregate: nodes_to_exprs(&aggregate, expr_arena),
            options,
        },
        #[cfg(feature = "parquet")]
        ALogicalPlan::ParquetScan {
            path,
            schema,
            output_schema: _,
            with_columns,
            predicate,
            aggregate,
            n_rows,
            cache,
        } => LogicalPlan::ParquetScan {
            path,
            schema,
            with_columns,
            predicate: predicate.map(|n| node_to_exp(n, expr_arena)),
            aggregate: nodes_to_exprs(&aggregate, expr_arena),
            n_rows,
            cache,
        },
        ALogicalPlan::DataFrameScan {
            df,
            schema,
            projection,
            selection,
        } => LogicalPlan::DataFrameScan {
            df,
            schema,
            projection: projection
                .as_ref()
                .map(|nodes| nodes.iter().map(|n| node_to_exp(*n, expr_arena)).collect()),
            selection: selection.map(|n| node_to_exp(n, expr_arena)),
        },
        ALogicalPlan::Projection {
            expr,
            input,
            schema,
        } => {
            let i = node_to_lp(input, expr_arena, lp_arena);

            LogicalPlan::Projection {
                expr: nodes_to_exprs(&expr, expr_arena),
                input: Box::new(i),
                schema,
            }
        }
        ALogicalPlan::LocalProjection {
            expr,
            input,
            schema,
        } => {
            let i = node_to_lp(input, expr_arena, lp_arena);

            LogicalPlan::LocalProjection {
                expr: nodes_to_exprs(&expr, expr_arena),
                input: Box::new(i),
                schema,
            }
        }
        ALogicalPlan::Sort {
            input,
            by_column,
            reverse,
        } => {
            let input = Box::new(node_to_lp(input, expr_arena, lp_arena));
            LogicalPlan::Sort {
                input,
                by_column: nodes_to_exprs(&by_column, expr_arena),
                reverse,
            }
        }
        ALogicalPlan::Explode { input, columns } => {
            let input = Box::new(node_to_lp(input, expr_arena, lp_arena));
            LogicalPlan::Explode { input, columns }
        }
        ALogicalPlan::Cache { input } => {
            let input = Box::new(node_to_lp(input, expr_arena, lp_arena));
            LogicalPlan::Cache { input }
        }
        ALogicalPlan::Aggregate {
            input,
            keys,
            aggs,
            schema,
            apply,
            maintain_order,
            dynamic_options,
        } => {
            let i = node_to_lp(input, expr_arena, lp_arena);

            LogicalPlan::Aggregate {
                input: Box::new(i),
                keys: Arc::new(nodes_to_exprs(&keys, expr_arena)),
                aggs: nodes_to_exprs(&aggs, expr_arena),
                schema,
                apply,
                maintain_order,
                dynamic_options,
            }
        }
        ALogicalPlan::Join {
            input_left,
            input_right,
            schema,
            left_on,
            right_on,
            options,
        } => {
            let i_l = node_to_lp(input_left, expr_arena, lp_arena);
            let i_r = node_to_lp(input_right, expr_arena, lp_arena);

            LogicalPlan::Join {
                input_left: Box::new(i_l),
                input_right: Box::new(i_r),
                schema,
                left_on: nodes_to_exprs(&left_on, expr_arena),
                right_on: nodes_to_exprs(&right_on, expr_arena),
                options,
            }
        }
        ALogicalPlan::HStack {
            input,
            exprs,
            schema,
        } => {
            let i = node_to_lp(input, expr_arena, lp_arena);

            LogicalPlan::HStack {
                input: Box::new(i),
                exprs: nodes_to_exprs(&exprs, expr_arena),
                schema,
            }
        }
        ALogicalPlan::Distinct {
            input,
            maintain_order,
            subset,
        } => {
            let i = node_to_lp(input, expr_arena, lp_arena);
            LogicalPlan::Distinct {
                input: Box::new(i),
                maintain_order,
                subset,
            }
        }
        ALogicalPlan::Melt {
            input,
            id_vars,
            value_vars,
            schema,
        } => {
            let input = node_to_lp(input, expr_arena, lp_arena);
            LogicalPlan::Melt {
                input: Box::new(input),
                id_vars,
                value_vars,
                schema,
            }
        }
        ALogicalPlan::Udf {
            input,
            function,
            predicate_pd,
            projection_pd,
            schema,
        } => {
            let input = Box::new(node_to_lp(input, expr_arena, lp_arena));
            LogicalPlan::Udf {
                input,
                function,
                predicate_pd,
                projection_pd,
                schema,
            }
        }
    }
}
