//! Optimization that collapses several a join with several filters into faster join.
//!
//! For example, `join(how='cross').filter(pl.col.l == pl.col.r)` can be collapsed to
//! `join(how='inner', left_on=pl.col.l, right_on=pl.col.r)`.

use std::sync::Arc;

use polars_core::schema::SchemaRef;
use polars_ops::frame::{IEJoinOptions, InequalityOperator, JoinCoalesce, JoinType};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::{aexpr_to_leaf_names_iter, AExpr, IR};
use crate::dsl::Operator;
use crate::plans::{ExprIR, OutputName};

/// Join origin of an expression
#[derive(Debug, Clone, Copy)]
enum ExprOrigin {
    /// Utilizes no columns
    None,
    /// Utilizes columns from the left side of the join
    Left,
    /// Utilizes columns from the right side of the join
    Right,
    /// Utilizes columns from both sides of the join
    Both,
}

fn get_origin(
    root: Node,
    expr_arena: &Arena<AExpr>,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    suffix: &str,
) -> ExprOrigin {
    let mut expr_origin = ExprOrigin::None;

    for name in aexpr_to_leaf_names_iter(root, expr_arena) {
        let in_left = left_schema.contains(name.as_str());
        let in_right = right_schema.contains(name.as_str());
        let has_suffix = name.as_str().ends_with(suffix);
        let in_right = in_right
            | (has_suffix && right_schema.contains(&name.as_str()[..name.len() - suffix.len()]));

        let name_origin = match (in_left, in_right, has_suffix) {
            (true, false, _) | (true, true, false) => ExprOrigin::Left,
            (false, true, _) | (true, true, true) => ExprOrigin::Right,
            (false, false, _) => {
                unreachable!("Invalid filter column should have been filtered before")
            },
        };

        use ExprOrigin as O;
        expr_origin = match (expr_origin, name_origin) {
            (O::None, other) | (other, O::None) => other,
            (O::Left, O::Left) => O::Left,
            (O::Right, O::Right) => O::Right,
            _ => O::Both,
        };
    }

    expr_origin
}

/// Remove the join suffixes from a list of expressions
fn remove_suffix(
    exprs: &mut Vec<ExprIR>,
    expr_arena: &mut Arena<AExpr>,
    schema: &SchemaRef,
    suffix: &str,
) {
    let mut stack = Vec::new();

    for expr in exprs {
        if let OutputName::ColumnLhs(colname) = expr.output_name_inner() {
            if colname.ends_with(suffix) && !schema.contains(colname.as_str()) {
                expr.set_columnlhs(PlSmallStr::from(&colname[..colname.len() - suffix.len()]));
            }
        }

        stack.clear();
        stack.push(expr.node());
        while let Some(node) = stack.pop() {
            let expr = expr_arena.get_mut(node);
            expr.nodes(&mut stack);

            let AExpr::Column(colname) = expr else {
                continue;
            };

            if !colname.ends_with(suffix) || schema.contains(colname.as_str()) {
                continue;
            }

            *colname = PlSmallStr::from(&colname[..colname.len() - suffix.len()]);
        }
    }
}

/// An iterator over all the minterms in a boolean expression boolean.
///
/// In other words, all the terms that can `AND` together to form this expression.
///
/// # Example
///
/// ```
/// a & (b | c) & (b & (c | (a & c)))
/// ```
///
/// Gives terms:
///
/// ```
/// a
/// b | c
/// b
/// c | (a & c)
/// ```
struct MintermIter<'a> {
    stack: Vec<Node>,
    expr_arena: &'a Arena<AExpr>,
}

impl<'a> Iterator for MintermIter<'a> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        let mut top = self.stack.pop()?;

        while let AExpr::BinaryExpr {
            left,
            op: Operator::And,
            right,
        } = self.expr_arena.get(top)
        {
            self.stack.push(*right);
            top = *left;
        }

        Some(top)
    }
}

impl<'a> MintermIter<'a> {
    fn new(root: Node, expr_arena: &'a Arena<AExpr>) -> Self {
        Self {
            stack: vec![root],
            expr_arena,
        }
    }
}

fn and_expr(left: Node, right: Node, expr_arena: &mut Arena<AExpr>) -> Node {
    expr_arena.add(AExpr::BinaryExpr {
        left,
        op: Operator::And,
        right,
    })
}

pub fn optimize(root: Node, lp_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>) {
    let mut predicates = Vec::with_capacity(4);

    // Partition to:
    // - equality predicates
    // - IEjoin supported inequality predicates
    // - remaining predicates
    let mut ie_op = Vec::new();
    let mut ie_exprs = Vec::new();
    let mut remaining_predicates = Vec::new();

    let mut ir_stack = Vec::with_capacity(16);
    ir_stack.push(root);

    while let Some(current) = ir_stack.pop() {
        let current_ir = lp_arena.get(current);
        current_ir.copy_inputs(&mut ir_stack);

        match current_ir {
            IR::Filter {
                input: _,
                predicate,
            } => {
                predicates.push((current, predicate.node()));
            },
            IR::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            } if matches!(options.args.how, JoinType::Cross) => {
                if predicates.is_empty() {
                    continue;
                }

                let suffix = options.args.suffix();

                debug_assert!(left_on.is_empty());
                debug_assert!(right_on.is_empty());

                let mut eq_left_on = Vec::new();
                let mut eq_right_on = Vec::new();

                let mut ie_left_on = Vec::new();
                let mut ie_right_on = Vec::new();

                ie_op.clear();
                ie_exprs.clear();

                remaining_predicates.clear();

                fn to_inequality_operator(op: &Operator) -> Option<InequalityOperator> {
                    match op {
                        Operator::Lt => Some(InequalityOperator::Lt),
                        Operator::LtEq => Some(InequalityOperator::LtEq),
                        Operator::Gt => Some(InequalityOperator::Gt),
                        Operator::GtEq => Some(InequalityOperator::GtEq),
                        _ => None,
                    }
                }

                let left_schema = lp_arena.get(*input_left).schema(lp_arena);
                let right_schema = lp_arena.get(*input_right).schema(lp_arena);

                let left_schema = left_schema.as_ref();
                let right_schema = right_schema.as_ref();

                for (_, predicate_node) in &predicates {
                    for node in MintermIter::new(*predicate_node, expr_arena) {
                        let AExpr::BinaryExpr { left, op, right } = expr_arena.get(node) else {
                            remaining_predicates.push(node);
                            continue;
                        };

                        if !op.is_comparison() {
                            // @NOTE: This is not a valid predicate, but we should not handle that
                            // here.
                            remaining_predicates.push(node);
                            continue;
                        }

                        let mut left = *left;
                        let mut op = *op;
                        let mut right = *right;

                        let left_origin = get_origin(
                            left,
                            expr_arena,
                            left_schema,
                            right_schema,
                            suffix.as_str(),
                        );
                        let right_origin = get_origin(
                            right,
                            expr_arena,
                            left_schema,
                            right_schema,
                            suffix.as_str(),
                        );

                        use ExprOrigin as EO;

                        // We can only join if both sides of the binary expression stem from
                        // different sides of the join.
                        match (left_origin, right_origin) {
                            (EO::Both, _) | (_, EO::Both) => {
                                // If either expression originates from the both sides, we need to
                                // filter it afterwards.
                                remaining_predicates.push(node);
                                continue;
                            },
                            (EO::None, _) | (_, EO::None) => {
                                // @TODO: This should probably be pushed down
                                remaining_predicates.push(node);
                                continue;
                            },
                            (EO::Left, EO::Left) | (EO::Right, EO::Right) => {
                                // @TODO: This can probably be pushed down in the predicate
                                // pushdown, but for now just take it as is.
                                remaining_predicates.push(node);
                                continue;
                            },
                            (EO::Right, EO::Left) => {
                                // Swap around the expressions so they match with the left_on and
                                // right_on.
                                std::mem::swap(&mut left, &mut right);
                                op = op.swap_operands();
                            },
                            (EO::Left, EO::Right) => {},
                        }

                        if let Some(ie_op_) = to_inequality_operator(&op) {
                            // We already have an IEjoin or an Inner join, push to remaining
                            if ie_op.len() >= 2 || !eq_left_on.is_empty() {
                                remaining_predicates.push(node);
                            } else {
                                ie_left_on.push(ExprIR::from_node(left, expr_arena));
                                ie_right_on.push(ExprIR::from_node(right, expr_arena));
                                ie_op.push(ie_op_);
                                ie_exprs.push(node);
                            }
                        } else if matches!(op, Operator::Eq) {
                            eq_left_on.push(ExprIR::from_node(left, expr_arena));
                            eq_right_on.push(ExprIR::from_node(right, expr_arena));
                        } else {
                            remaining_predicates.push(node);
                        }
                    }
                }

                if !eq_left_on.is_empty() {
                    let mut options = options.as_ref().clone();
                    options.args.how = JoinType::Inner;
                    // We need to make sure not to delete any columns
                    options.args.coalesce = JoinCoalesce::KeepColumns;

                    let input_left = *input_left;
                    let input_right = *input_right;
                    let schema = schema.clone();

                    remove_suffix(&mut eq_right_on, expr_arena, right_schema, suffix.as_str());

                    let new_join = IR::Join {
                        input_left,
                        input_right,
                        schema,
                        left_on: eq_left_on,
                        right_on: eq_right_on,
                        options: Arc::new(options),
                    };

                    lp_arena.replace(current, new_join);

                    let remaining_predicates_sum = remaining_predicates
                        .iter()
                        .copied()
                        .reduce(|left, right| and_expr(left, right, expr_arena));

                    let ie_sum = ie_exprs
                        .iter()
                        .copied()
                        .reduce(|left, right| and_expr(left, right, expr_arena));

                    let predicate_sum = match (remaining_predicates_sum, ie_sum) {
                        (None, None) => None,
                        (Some(l), Some(r)) => Some(and_expr(l, r, expr_arena)),
                        (Some(l), None) | (None, Some(l)) => Some(l),
                    };
                    if let Some(sum_predicate) = predicate_sum {
                        let IR::Filter { input, predicate } = lp_arena.get_mut(predicates[0].0)
                        else {
                            unreachable!();
                        };

                        *input = current;
                        predicate.set_node(sum_predicate);
                    } else {
                        // There are no predicates anymore. Just put the JOIN at the filters
                        // position.
                        lp_arena.swap(current, predicates[0].0);
                    }
                } else if ie_exprs.len() >= 2 {
                    debug_assert_eq!(ie_exprs.len(), 2);

                    // Do an IEjoin.
                    let mut options = options.as_ref().clone();
                    options.args.how = JoinType::IEJoin(IEJoinOptions {
                        operator1: ie_op[0],
                        operator2: ie_op[1],
                    });
                    // We need to make sure not to delete any columns
                    options.args.coalesce = JoinCoalesce::KeepColumns;

                    let input_left = *input_left;
                    let input_right = *input_right;
                    let schema = schema.clone();

                    remove_suffix(&mut ie_right_on, expr_arena, right_schema, suffix.as_str());

                    let new_join = IR::Join {
                        input_left,
                        input_right,
                        schema,
                        left_on: ie_left_on,
                        right_on: ie_right_on,
                        options: Arc::new(options),
                    };

                    lp_arena.replace(current, new_join);

                    let remaining_predicates_sum = remaining_predicates
                        .iter()
                        .copied()
                        .reduce(|left, right| and_expr(left, right, expr_arena));

                    if let Some(predicates_sum) = remaining_predicates_sum {
                        let IR::Filter { input, predicate } = lp_arena.get_mut(predicates[0].0)
                        else {
                            unreachable!();
                        };

                        *input = current;
                        predicate.set_node(predicates_sum);
                    } else {
                        // There are no predicates anymore. Just put the JOIN at the filters
                        // position.
                        lp_arena.swap(current, predicates[0].0);
                    }
                }

                predicates.clear();
            },
            _ => {
                predicates.clear();
            },
        }
    }
}
