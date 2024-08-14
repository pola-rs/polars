use super::*;

/// converts a node from the AExpr arena to Expr
#[recursive]
pub fn node_to_expr(node: Node, expr_arena: &Arena<AExpr>) -> Expr {
    let expr = expr_arena.get(node).clone();

    match expr {
        AExpr::Explode(node) => Expr::Explode(Arc::new(node_to_expr(node, expr_arena))),
        AExpr::Alias(expr, name) => {
            let exp = node_to_expr(expr, expr_arena);
            Expr::Alias(Arc::new(exp), name)
        },
        AExpr::Column(a) => Expr::Column(a),
        AExpr::Literal(s) => Expr::Literal(s),
        AExpr::BinaryExpr { left, op, right } => {
            let l = node_to_expr(left, expr_arena);
            let r = node_to_expr(right, expr_arena);
            Expr::BinaryExpr {
                left: Arc::new(l),
                op,
                right: Arc::new(r),
            }
        },
        AExpr::Cast {
            expr,
            data_type,
            options: strict,
        } => {
            let exp = node_to_expr(expr, expr_arena);
            Expr::Cast {
                expr: Arc::new(exp),
                data_type,
                options: strict,
            }
        },
        AExpr::Sort { expr, options } => {
            let exp = node_to_expr(expr, expr_arena);
            Expr::Sort {
                expr: Arc::new(exp),
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
                expr: Arc::new(expr),
                idx: Arc::new(idx),
                returns_scalar,
            }
        },
        AExpr::SortBy {
            expr,
            by,
            sort_options,
        } => {
            let expr = node_to_expr(expr, expr_arena);
            let by = by
                .iter()
                .map(|node| node_to_expr(*node, expr_arena))
                .collect();
            Expr::SortBy {
                expr: Arc::new(expr),
                by,
                sort_options,
            }
        },
        AExpr::Filter { input, by } => {
            let input = node_to_expr(input, expr_arena);
            let by = node_to_expr(by, expr_arena);
            Expr::Filter {
                input: Arc::new(input),
                by: Arc::new(by),
            }
        },
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Min {
                input,
                propagate_nans,
            } => {
                let exp = node_to_expr(input, expr_arena);
                AggExpr::Min {
                    input: Arc::new(exp),
                    propagate_nans,
                }
                .into()
            },
            IRAggExpr::Max {
                input,
                propagate_nans,
            } => {
                let exp = node_to_expr(input, expr_arena);
                AggExpr::Max {
                    input: Arc::new(exp),
                    propagate_nans,
                }
                .into()
            },

            IRAggExpr::Median(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Median(Arc::new(exp)).into()
            },
            IRAggExpr::NUnique(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::NUnique(Arc::new(exp)).into()
            },
            IRAggExpr::First(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::First(Arc::new(exp)).into()
            },
            IRAggExpr::Last(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Last(Arc::new(exp)).into()
            },
            IRAggExpr::Mean(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Mean(Arc::new(exp)).into()
            },
            IRAggExpr::Implode(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Implode(Arc::new(exp)).into()
            },
            IRAggExpr::Quantile {
                expr,
                quantile,
                interpol,
            } => {
                let expr = node_to_expr(expr, expr_arena);
                let quantile = node_to_expr(quantile, expr_arena);
                AggExpr::Quantile {
                    expr: Arc::new(expr),
                    quantile: Arc::new(quantile),
                    interpol,
                }
                .into()
            },
            IRAggExpr::Sum(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Sum(Arc::new(exp)).into()
            },
            IRAggExpr::Std(expr, ddof) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Std(Arc::new(exp), ddof).into()
            },
            IRAggExpr::Var(expr, ddof) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Var(Arc::new(exp), ddof).into()
            },
            IRAggExpr::AggGroups(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::AggGroups(Arc::new(exp)).into()
            },
            IRAggExpr::Count(expr, include_nulls) => {
                let expr = node_to_expr(expr, expr_arena);
                AggExpr::Count(Arc::new(expr), include_nulls).into()
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
                predicate: Arc::new(p),
                truthy: Arc::new(t),
                falsy: Arc::new(f),
            }
        },
        AExpr::AnonymousFunction {
            input,
            function,
            output_type,
            options,
        } => Expr::AnonymousFunction {
            input: expr_irs_to_exprs(input, expr_arena),
            function,
            output_type,
            options,
        },
        AExpr::Function {
            input,
            function,
            options,
        } => Expr::Function {
            input: expr_irs_to_exprs(input, expr_arena),
            function,
            options,
        },
        AExpr::Window {
            function,
            partition_by,
            order_by,
            options,
        } => {
            let function = Arc::new(node_to_expr(function, expr_arena));
            let partition_by = nodes_to_exprs(&partition_by, expr_arena);
            let order_by =
                order_by.map(|(n, options)| (Arc::new(node_to_expr(n, expr_arena)), options));
            Expr::Window {
                function,
                partition_by,
                order_by,
                options,
            }
        },
        AExpr::Slice {
            input,
            offset,
            length,
        } => Expr::Slice {
            input: Arc::new(node_to_expr(input, expr_arena)),
            offset: Arc::new(node_to_expr(offset, expr_arena)),
            length: Arc::new(node_to_expr(length, expr_arena)),
        },
        AExpr::Len => Expr::Len,
    }
}

fn nodes_to_exprs(nodes: &[Node], expr_arena: &Arena<AExpr>) -> Vec<Expr> {
    nodes.iter().map(|n| node_to_expr(*n, expr_arena)).collect()
}

pub fn node_to_lp_cloned(
    node: Node,
    expr_arena: &Arena<AExpr>,
    mut lp_arena: &Arena<IR>,
) -> DslPlan {
    // we borrow again mutably only to make the types happy
    // we want to initialize `to_lp` from a mutable and a immutable lp_arena
    // by borrowing an immutable mutably, we still are immutable down the line.
    let alp = lp_arena.get(node).clone();
    alp.into_lp(
        &|node, lp_arena: &mut &Arena<IR>| lp_arena.get(node).clone(),
        &mut lp_arena,
        expr_arena,
    )
}

/// converts a node from the IR arena to a LogicalPlan
pub fn node_to_lp(node: Node, expr_arena: &Arena<AExpr>, lp_arena: &mut Arena<IR>) -> DslPlan {
    let alp = lp_arena.get_mut(node);
    let alp = std::mem::take(alp);
    alp.into_lp(
        &|node, lp_arena: &mut Arena<IR>| {
            let lp = lp_arena.get_mut(node);
            std::mem::take(lp)
        },
        lp_arena,
        expr_arena,
    )
}
