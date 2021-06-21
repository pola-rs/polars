use polars_core::prelude::*;
use polars_core::utils::get_supertype;

use crate::logical_plan::optimizer::stack_opt::OptimizationRule;
use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::is_scan;

pub struct TypeCoercionRule {}

impl OptimizationRule for TypeCoercionRule {
    fn optimize_expr(
        &self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        lp_arena: &Arena<ALogicalPlan>,
        lp_node: Node,
    ) -> Option<AExpr> {
        let expr = expr_arena.get(expr_node);
        match *expr {
            AExpr::Ternary {
                truthy: truthy_node,
                falsy: falsy_node,
                predicate,
            } => {
                let plan = lp_arena.get(lp_node);
                let mut inputs = [None, None];

                // Used to get the schema of the input.
                if is_scan(plan) {
                    inputs[0] = Some(lp_node);
                } else {
                    plan.copy_inputs(&mut inputs);
                };

                if let Some(input) = inputs[0] {
                    let input_schema = lp_arena.get(input).schema(lp_arena);
                    let truthy = expr_arena.get(truthy_node);
                    let falsy = expr_arena.get(falsy_node);
                    let type_true = truthy
                        .get_type(input_schema, Context::Default, expr_arena)
                        .expect("could not dtype");
                    let type_false = falsy
                        .get_type(input_schema, Context::Default, expr_arena)
                        .expect("could not dtype");

                    if type_true == type_false {
                        None
                    } else {
                        let st = get_supertype(&type_true, &type_false).expect("supertype");
                        let new_node_truthy = expr_arena.add(AExpr::Cast {
                            expr: truthy_node,
                            data_type: st.clone(),
                        });
                        let new_node_falsy = expr_arena.add(AExpr::Cast {
                            expr: falsy_node,
                            data_type: st,
                        });
                        Some(AExpr::Ternary {
                            truthy: new_node_truthy,
                            falsy: new_node_falsy,
                            predicate,
                        })
                    }
                } else {
                    None
                }
            }
            AExpr::BinaryExpr {
                left: node_left,
                op,
                right: node_right,
            } => {
                let plan = lp_arena.get(lp_node);
                let mut inputs = [None, None];

                if is_scan(plan) {
                    inputs[0] = Some(lp_node);
                } else {
                    plan.copy_inputs(&mut inputs);
                };

                if let Some(input) = inputs[0] {
                    let input_schema = lp_arena.get(input).schema(lp_arena);

                    let left = expr_arena.get(node_left);
                    let right = expr_arena.get(node_right);

                    let type_left = left
                        .get_type(input_schema, Context::Default, expr_arena)
                        .expect("could not get dtype");
                    let type_right = right
                        .get_type(input_schema, Context::Default, expr_arena)
                        .expect("could not get dtype");

                    let compare_cat_to_string = matches!(
                        op,
                        Operator::Eq
                            | Operator::NotEq
                            | Operator::Gt
                            | Operator::Lt
                            | Operator::GtEq
                            | Operator::LtEq
                    ) && ((type_left == DataType::Categorical
                        && type_right == DataType::Utf8)
                        || (type_left == DataType::Utf8 && type_right == DataType::Categorical));

                    if type_left == type_right || compare_cat_to_string {
                        None
                    } else {
                        let st = get_supertype(&type_left, &type_right)
                            .expect("could not find supertype of binary expr");
                        let new_node_left = expr_arena.add(AExpr::Cast {
                            expr: node_left,
                            data_type: st.clone(),
                        });
                        let new_node_right = expr_arena.add(AExpr::Cast {
                            expr: node_right,
                            data_type: st,
                        });

                        Some(AExpr::BinaryExpr {
                            left: new_node_left,
                            op,
                            right: new_node_right,
                        })
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
