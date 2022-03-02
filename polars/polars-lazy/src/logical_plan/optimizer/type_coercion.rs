use polars_core::prelude::*;
use polars_core::utils::get_supertype;

use crate::logical_plan::optimizer::stack_opt::OptimizationRule;
use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::is_scan;

pub struct TypeCoercionRule {}

/// determine if we use the supertype or not. For instance when we have a column Int64 and we compare with literal UInt32
/// it would be wasteful to cast the column instead of the literal.
fn use_supertype(
    mut st: DataType,
    left: &AExpr,
    right: &AExpr,
    type_left: &DataType,
    type_right: &DataType,
) -> DataType {
    // only interesting on numerical types
    // other types will always use the supertype.
    if type_left.is_numeric() && type_right.is_numeric() {
        match (left, right) {
            // don't let the literal f64 coerce the f32 column
            (AExpr::Literal(LiteralValue::Float64(_)), _) if matches!(type_right, DataType::Float32) => {
                st = DataType::Float32
            }
            (_, AExpr::Literal(LiteralValue::Float64(_))) if matches!(type_left, DataType::Float32) => {
                st = DataType::Float32
            }

            // do nothing and use supertype
            (AExpr::Literal(_), AExpr::Literal(_))
            // always make sure that we cast to floats if one of the operands is float
            // and the left type is integer
            |(AExpr::Literal(LiteralValue::Float32(_) | LiteralValue::Float64(_)), _)
            |(_, AExpr::Literal(LiteralValue::Float32(_) | LiteralValue::Float64(_)))
            => {}

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
    }
    st
}

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
                        let st = use_supertype(st, truthy, falsy, &type_true, &type_false);

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

                    #[allow(unused_mut, unused_assignments)]
                    let mut compare_cat_to_string = false;
                    #[cfg(feature = "dtype-categorical")]
                    {
                        compare_cat_to_string =
                            matches!(
                                op,
                                Operator::Eq
                                    | Operator::NotEq
                                    | Operator::Gt
                                    | Operator::Lt
                                    | Operator::GtEq
                                    | Operator::LtEq
                            ) && (matches!(type_left, DataType::Categorical(_))
                                && type_right == DataType::Utf8)
                                || (type_left == DataType::Utf8
                                    && matches!(type_right, DataType::Categorical(_)));
                    }

                    let datetime_arithmetic = matches!(op, Operator::Minus | Operator::Plus)
                        && matches!(
                            (&type_left, &type_right),
                            (DataType::Datetime(_, _), DataType::Duration(_))
                                | (DataType::Duration(_), DataType::Datetime(_, _))
                                | (DataType::Date, DataType::Duration(_))
                                | (DataType::Duration(_), DataType::Date)
                        );

                    if type_left == type_right || compare_cat_to_string || datetime_arithmetic {
                        None
                    } else {
                        let st = get_supertype(&type_left, &type_right)
                            .expect("could not find supertype of binary expr");

                        let mut st = use_supertype(st, left, right, &type_left, &type_right);

                        #[allow(unused_mut, unused_assignments)]
                        let mut cat_str_arithmetic = false;

                        #[cfg(feature = "dtype-categorical")]
                        {
                            cat_str_arithmetic = (matches!(type_left, DataType::Categorical(_))
                                && type_right == DataType::Utf8)
                                || (type_left == DataType::Utf8
                                    && matches!(type_right, DataType::Categorical(_)));
                        }

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
    use crate::logical_plan::optimizer::stack_opt::OptimizationRule;
    use crate::prelude::*;
    use crate::utils::test::optimize_expr;
    use polars_core::prelude::*;

    #[test]
    fn test_categorical_utf8() {
        let mut rules: Vec<Box<dyn OptimizationRule>> = vec![Box::new(TypeCoercionRule {})];
        let schema = Schema::from(vec![Field::new("fruits", DataType::Categorical(None))]);

        let expr = col("fruits").eq(lit("somestr"));
        let out = optimize_expr(expr.clone(), schema.clone(), &mut rules);
        // we test that the fruits column is not casted to utf8 for the comparison
        assert_eq!(out, expr);

        let expr = col("fruits") + (lit("somestr"));
        let out = optimize_expr(expr, schema, &mut rules);
        let expected = col("fruits").cast(DataType::Utf8) + lit("somestr");
        assert_eq!(out, expected);
    }
}
