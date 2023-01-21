use polars_core::prelude::*;

use crate::prelude::*;

fn to_aexprs(input: Vec<Expr>, arena: &mut Arena<AExpr>) -> Vec<Node> {
    input.into_iter().map(|e| to_aexpr(e, arena)).collect()
}

/// converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation
pub fn to_aexpr(expr: Expr, arena: &mut Arena<AExpr>) -> Node {
    let v = match expr {
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
                AggExpr::Min {
                    input,
                    propagate_nans,
                } => AAggExpr::Min {
                    input: to_aexpr(*input, arena),
                    propagate_nans,
                },
                AggExpr::Max {
                    input,
                    propagate_nans,
                } => AAggExpr::Max {
                    input: to_aexpr(*input, arena),
                    propagate_nans,
                },
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
                    quantile: to_aexpr(*quantile, arena),
                    interpol,
                },
                AggExpr::Sum(expr) => AAggExpr::Sum(to_aexpr(*expr, arena)),
                AggExpr::Std(expr, ddof) => AAggExpr::Std(to_aexpr(*expr, arena), ddof),
                AggExpr::Var(expr, ddof) => AAggExpr::Var(to_aexpr(*expr, arena), ddof),
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
        Expr::AnonymousFunction {
            input,
            function,
            output_type,
            options,
        } => AExpr::AnonymousFunction {
            input: to_aexprs(input, arena),
            function,
            output_type,
            options,
        },
        Expr::Function {
            input,
            function,
            options,
        } => AExpr::Function {
            input: to_aexprs(input, arena),
            function,
            options,
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
            offset: to_aexpr(*offset, arena),
            length: to_aexpr(*length, arena),
        },
        Expr::Wildcard => AExpr::Wildcard,
        Expr::Count => AExpr::Count,
        Expr::Nth(i) => AExpr::Nth(i),
        Expr::KeepName(_) => panic!("no keep_name expected at this point"),
        Expr::Exclude(_, _) => panic!("no exclude expected at this point"),
        Expr::RenameAlias { .. } => panic!("no `rename_alias` expected at this point"),
        Expr::Columns { .. } => panic!("no `columns` expected at this point"),
        Expr::DtypeColumn { .. } => panic!("no `dtype-columns` expected at this point"),
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
        LogicalPlan::AnonymousScan {
            function,
            file_info,
            predicate,
            options,
        } => ALogicalPlan::AnonymousScan {
            function,
            file_info,
            output_schema: None,
            predicate: predicate.map(|expr| to_aexpr(expr, expr_arena)),
            options,
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
        }
        LogicalPlan::Selection { input, predicate } => {
            let i = to_alp(*input, expr_arena, lp_arena)?;
            let p = to_aexpr(predicate, expr_arena);
            ALogicalPlan::Selection {
                input: i,
                predicate: p,
            }
        }
        LogicalPlan::Slice { input, offset, len } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::Slice { input, offset, len }
        }
        LogicalPlan::Melt {
            input,
            args,
            schema,
        } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::Melt {
                input,
                args,
                schema,
            }
        }
        #[cfg(feature = "csv-file")]
        LogicalPlan::CsvScan {
            path,
            file_info,
            options,
            predicate,
        } => ALogicalPlan::CsvScan {
            path,
            file_info,
            output_schema: None,
            options,
            predicate: predicate.map(|expr| to_aexpr(expr, expr_arena)),
        },
        #[cfg(feature = "ipc")]
        LogicalPlan::IpcScan {
            path,
            file_info,
            predicate,
            options,
        } => ALogicalPlan::IpcScan {
            path,
            file_info,
            output_schema: None,
            predicate: predicate.map(|expr| to_aexpr(expr, expr_arena)),
            options,
        },
        #[cfg(feature = "parquet")]
        LogicalPlan::ParquetScan {
            path,
            file_info,
            predicate,
            options,
            cloud_options,
        } => ALogicalPlan::ParquetScan {
            path,
            file_info,
            output_schema: None,
            predicate: predicate.map(|expr| to_aexpr(expr, expr_arena)),
            options,
            cloud_options,
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
            selection: selection.map(|expr| to_aexpr(expr, expr_arena)),
        },
        LogicalPlan::Projection {
            expr,
            input,
            schema,
        } => {
            let exp = expr.into_iter().map(|x| to_aexpr(x, expr_arena)).collect();
            let i = to_alp(*input, expr_arena, lp_arena)?;
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
            let i = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::LocalProjection {
                expr: exp,
                input: i,
                schema,
            }
        }
        LogicalPlan::Sort {
            input,
            by_column,
            args,
        } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            let by_column = by_column
                .into_iter()
                .map(|x| to_aexpr(x, expr_arena))
                .collect();
            ALogicalPlan::Sort {
                input,
                by_column,
                args,
            }
        }
        LogicalPlan::Explode {
            input,
            columns,
            schema,
        } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::Explode {
                input,
                columns,
                schema,
            }
        }
        LogicalPlan::Cache { input, id, count } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::Cache { input, id, count }
        }
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
                options,
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
            let input_left = to_alp(*input_left, expr_arena, lp_arena)?;
            let input_right = to_alp(*input_right, expr_arena, lp_arena)?;

            let l_on = left_on
                .into_iter()
                .map(|x| to_aexpr(x, expr_arena))
                .collect();
            let r_on = right_on
                .into_iter()
                .map(|x| to_aexpr(x, expr_arena))
                .collect();

            ALogicalPlan::Join {
                input_left,
                input_right,
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
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::HStack {
                input,
                exprs: exp,
                schema,
            }
        }
        LogicalPlan::Distinct { input, options } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::Distinct { input, options }
        }
        LogicalPlan::MapFunction { input, function } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::MapFunction { input, function }
        }
        LogicalPlan::Error { err, .. } => {
            // We just take the error. The LogicalPlan should not be used anymore once this
            // is taken.
            return Err(err.take());
        }
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
        }
        LogicalPlan::FileSink { input, payload } => {
            let input = to_alp(*input, expr_arena, lp_arena)?;
            ALogicalPlan::FileSink { input, payload }
        }
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
        }
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
        }
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
        }
        AExpr::Sort { expr, options } => {
            let exp = node_to_expr(expr, expr_arena);
            Expr::Sort {
                expr: Box::new(exp),
                options,
            }
        }
        AExpr::Take { expr, idx } => {
            let expr = node_to_expr(expr, expr_arena);
            let idx = node_to_expr(idx, expr_arena);
            Expr::Take {
                expr: Box::new(expr),
                idx: Box::new(idx),
            }
        }
        AExpr::SortBy { expr, by, reverse } => {
            let expr = node_to_expr(expr, expr_arena);
            let by = by
                .iter()
                .map(|node| node_to_expr(*node, expr_arena))
                .collect();
            Expr::SortBy {
                expr: Box::new(expr),
                by,
                reverse,
            }
        }
        AExpr::Filter { input, by } => {
            let input = node_to_expr(input, expr_arena);
            let by = node_to_expr(by, expr_arena);
            Expr::Filter {
                input: Box::new(input),
                by: Box::new(by),
            }
        }
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
            }
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
            }

            AAggExpr::Median(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Median(Box::new(exp)).into()
            }
            AAggExpr::NUnique(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::NUnique(Box::new(exp)).into()
            }
            AAggExpr::First(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::First(Box::new(exp)).into()
            }
            AAggExpr::Last(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Last(Box::new(exp)).into()
            }
            AAggExpr::Mean(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Mean(Box::new(exp)).into()
            }
            AAggExpr::List(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::List(Box::new(exp)).into()
            }
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
            }
            AAggExpr::Sum(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Sum(Box::new(exp)).into()
            }
            AAggExpr::Std(expr, ddof) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Std(Box::new(exp), ddof).into()
            }
            AAggExpr::Var(expr, ddof) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Var(Box::new(exp), ddof).into()
            }
            AAggExpr::AggGroups(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::AggGroups(Box::new(exp)).into()
            }
            AAggExpr::Count(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Count(Box::new(exp)).into()
            }
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
        }
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
            order_by,
            options,
        } => {
            let function = Box::new(node_to_expr(function, expr_arena));
            let partition_by = nodes_to_exprs(&partition_by, expr_arena);
            let order_by = order_by.map(|ob| Box::new(node_to_expr(ob, expr_arena)));
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
            input: Box::new(node_to_expr(input, expr_arena)),
            offset: Box::new(node_to_expr(offset, expr_arena)),
            length: Box::new(node_to_expr(length, expr_arena)),
        },
        AExpr::Count => Expr::Count,
        AExpr::Nth(i) => Expr::Nth(i),
        AExpr::Wildcard => Expr::Wildcard,
    }
}

fn nodes_to_exprs(nodes: &[Node], expr_arena: &Arena<AExpr>) -> Vec<Expr> {
    nodes.iter().map(|n| node_to_expr(*n, expr_arena)).collect()
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
            ALogicalPlan::AnonymousScan {
                function,
                file_info,
                output_schema: _,
                predicate,
                options,
            } => LogicalPlan::AnonymousScan {
                function,
                file_info,
                predicate: predicate.map(|n| node_to_expr(n, expr_arena)),
                options,
            },
            #[cfg(feature = "python")]
            ALogicalPlan::PythonScan { options, .. } => LogicalPlan::PythonScan { options },
            ALogicalPlan::Union { inputs, options } => {
                let inputs = inputs
                    .into_iter()
                    .map(|node| convert_to_lp(node, lp_arena))
                    .collect();
                LogicalPlan::Union { inputs, options }
            }
            ALogicalPlan::Slice { input, offset, len } => {
                let lp = convert_to_lp(input, lp_arena);
                LogicalPlan::Slice {
                    input: Box::new(lp),
                    offset,
                    len,
                }
            }
            ALogicalPlan::Selection { input, predicate } => {
                let lp = convert_to_lp(input, lp_arena);
                let p = node_to_expr(predicate, expr_arena);
                LogicalPlan::Selection {
                    input: Box::new(lp),
                    predicate: p,
                }
            }
            #[cfg(feature = "csv-file")]
            ALogicalPlan::CsvScan {
                path,
                file_info,
                output_schema: _,
                options,
                predicate,
            } => LogicalPlan::CsvScan {
                path,
                file_info,
                options,
                predicate: predicate.map(|n| node_to_expr(n, expr_arena)),
            },
            #[cfg(feature = "ipc")]
            ALogicalPlan::IpcScan {
                path,
                file_info,
                output_schema: _,
                predicate,
                options,
            } => LogicalPlan::IpcScan {
                path,
                file_info,
                predicate: predicate.map(|n| node_to_expr(n, expr_arena)),
                options,
            },
            #[cfg(feature = "parquet")]
            ALogicalPlan::ParquetScan {
                path,
                file_info,
                output_schema: _,
                predicate,
                options,
                cloud_options,
            } => LogicalPlan::ParquetScan {
                path,
                file_info,
                predicate: predicate.map(|n| node_to_expr(n, expr_arena)),
                options,
                cloud_options,
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
                selection: selection.map(|n| node_to_expr(n, expr_arena)),
            },
            ALogicalPlan::Projection {
                expr,
                input,
                schema,
            } => {
                let i = convert_to_lp(input, lp_arena);

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
                let i = convert_to_lp(input, lp_arena);

                LogicalPlan::LocalProjection {
                    expr: nodes_to_exprs(&expr, expr_arena),
                    input: Box::new(i),
                    schema,
                }
            }
            ALogicalPlan::Sort {
                input,
                by_column,
                args,
            } => {
                let input = Box::new(convert_to_lp(input, lp_arena));
                LogicalPlan::Sort {
                    input,
                    by_column: nodes_to_exprs(&by_column, expr_arena),
                    args,
                }
            }
            ALogicalPlan::Explode {
                input,
                columns,
                schema,
            } => {
                let input = Box::new(convert_to_lp(input, lp_arena));
                LogicalPlan::Explode {
                    input,
                    columns,
                    schema,
                }
            }
            ALogicalPlan::Cache { input, id, count } => {
                let input = Box::new(convert_to_lp(input, lp_arena));
                LogicalPlan::Cache { input, id, count }
            }
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

                LogicalPlan::Aggregate {
                    input: Box::new(i),
                    keys: Arc::new(nodes_to_exprs(&keys, expr_arena)),
                    aggs: nodes_to_exprs(&aggs, expr_arena),
                    schema,
                    apply,
                    maintain_order,
                    options: dynamic_options,
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
                let i_l = convert_to_lp(input_left, lp_arena);
                let i_r = convert_to_lp(input_right, lp_arena);

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
                let i = convert_to_lp(input, lp_arena);

                LogicalPlan::HStack {
                    input: Box::new(i),
                    exprs: nodes_to_exprs(&exprs, expr_arena),
                    schema,
                }
            }
            ALogicalPlan::Distinct { input, options } => {
                let i = convert_to_lp(input, lp_arena);
                LogicalPlan::Distinct {
                    input: Box::new(i),
                    options,
                }
            }
            ALogicalPlan::Melt {
                input,
                args,
                schema,
            } => {
                let input = convert_to_lp(input, lp_arena);
                LogicalPlan::Melt {
                    input: Box::new(input),
                    args,
                    schema,
                }
            }
            ALogicalPlan::MapFunction { input, function } => {
                let input = Box::new(convert_to_lp(input, lp_arena));
                LogicalPlan::MapFunction { input, function }
            }
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
            }
            ALogicalPlan::FileSink { input, payload } => {
                let input = Box::new(convert_to_lp(input, lp_arena));
                LogicalPlan::FileSink { input, payload }
            }
        }
    }
}

pub fn node_to_lp_cloned(
    node: Node,
    expr_arena: &Arena<AExpr>,
    mut lp_arena: &Arena<ALogicalPlan>,
) -> LogicalPlan {
    // we borrow again mutably only to make the types happy
    // we want to intialize `to_lp` from a mutable and a immutable lp_arena
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
