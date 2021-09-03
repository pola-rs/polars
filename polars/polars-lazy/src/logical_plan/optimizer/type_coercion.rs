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
                        let mut st = get_supertype(&type_true, &type_false).expect("supertype");

                        match (truthy, falsy) {
                            // do nothing and use supertype
                            (AExpr::Literal(_), AExpr::Literal(_)) => {}
                            // cast literal to right type
                            (AExpr::Literal(_), _) => {
                                st = type_false.clone();
                            }
                            // cast literal to left type
                            (_, AExpr::Literal(_)) => {
                                st = type_true.clone();
                            }
                            // do nothing
                            _ => {}
                        }

                        // only cast if the type is not already the super type.
                        // this can prevent an expensive flattening and subsequent aggregation
                        // in a groupby context. To be able to cast the groups need to be
                        // flattened
                        let new_node_truthy = if type_true != st {
                            expr_arena.add(AExpr::Cast {
                                expr: truthy_node,
                                data_type: st.clone(),
                                strict: false,
                            })
                        } else {
                            truthy_node
                        };

                        let new_node_falsy = if type_false != st {
                            expr_arena.add(AExpr::Cast {
                                expr: falsy_node,
                                data_type: st,
                                strict: false,
                            })
                        } else {
                            falsy_node
                        };

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
                        let mut st = get_supertype(&type_left, &type_right)
                            .expect("could not find supertype of binary expr");

                        match (left, right) {
                            // do nothing and use supertype
                            (AExpr::Literal(_), AExpr::Literal(_)) => {}
                            // cast literal to right type
                            (AExpr::Literal(_), _) => {
                                st = type_right.clone();
                            }
                            // cast literal to left type
                            (_, AExpr::Literal(_)) => {
                                st = type_left.clone();
                            }
                            // do nothing
                            _ => {}
                        }

                        let cat_str_arithmetic = (type_left == DataType::Categorical
                            && type_right == DataType::Utf8)
                            || (type_left == DataType::Utf8 && type_right == DataType::Categorical);
                        if cat_str_arithmetic {
                            st = DataType::Utf8
                        }

                        // only cast if the type is not already the super type.
                        // this can prevent an expensive flattening and subsequent aggregation
                        // in a groupby context. To be able to cast the groups need to be
                        // flattened
                        let new_node_left = if type_left != st {
                            expr_arena.add(AExpr::Cast {
                                expr: node_left,
                                data_type: st.clone(),
                                strict: false,
                            })
                        } else {
                            node_left
                        };
                        let new_node_right = if type_right != st {
                            expr_arena.add(AExpr::Cast {
                                expr: node_right,
                                data_type: st,
                                strict: false,
                            })
                        } else {
                            node_right
                        };

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

#[cfg(test)]
mod test {
    use crate::logical_plan::optimizer::stack_opt::{OptimizationRule, StackOptimizer};
    use crate::prelude::*;
    use crate::utils::expr_to_root_column_names;
    use polars_core::prelude::*;

    fn optimize_expr(expr: Expr, schema: Schema) -> Expr {
        // initialize arena's
        let mut expr_arena = Arena::with_capacity(64);
        let mut lp_arena = Arena::with_capacity(32);
        let schema = Arc::new(schema);

        // dummy input needed to put the schema
        let input = Box::new(LogicalPlan::Projection {
            expr: vec![],
            input: Box::new(Default::default()),
            schema: schema.clone(),
        });

        let lp = LogicalPlan::Projection {
            expr: vec![expr],
            input,
            schema,
        };

        let root = to_alp(lp, &mut expr_arena, &mut lp_arena);
        let mut rules: Vec<Box<dyn OptimizationRule>> = vec![Box::new(TypeCoercionRule {})];

        let opt = StackOptimizer {};
        let lp_top = opt.optimize_loop(&mut rules, &mut expr_arena, &mut lp_arena, root);
        if let LogicalPlan::Projection { mut expr, .. } =
            node_to_lp(lp_top, &mut expr_arena, &mut lp_arena)
        {
            expr.pop().unwrap()
        } else {
            unreachable!()
        }
    }

    #[test]
    fn test_categorical_utf8() {
        let schema = Schema::new(vec![Field::new("fruits", DataType::Categorical)]);

        let expr = col("fruits").eq(lit("somestr"));
        let out = optimize_expr(expr.clone(), schema.clone());
        // we test that the fruits column is not casted to utf8 for the comparison
        assert_eq!(out, expr);

        let expr = col("fruits") + (lit("somestr"));
        let out = optimize_expr(expr.clone(), schema);
        let expected = col("fruits").cast(DataType::Utf8) + lit("somestr");
        assert_eq!(out, expected);
    }
}
