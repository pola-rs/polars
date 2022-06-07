use crate::dsl::function_expr::FunctionExpr;
use polars_core::prelude::*;
use polars_core::utils::get_supertype;

use crate::logical_plan::optimizer::stack_opt::OptimizationRule;
use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::is_scan;

pub struct TypeCoercionRule {}

/// determine if we use the supertype or not. For instance when we have a column Int64 and we compare with literal UInt32
/// it would be wasteful to cast the column instead of the literal.
fn modify_supertype(
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

            // cast literal to right type if they fit in the range
            (AExpr::Literal(value), _) => {
                if let Some(lit_val) = value.to_anyvalue() {
                   if type_right.value_within_range(lit_val) {
                       st = type_right.clone();
                   }
                }
            }
            // cast literal to left type
            (_, AExpr::Literal(value)) => {

                if let Some(lit_val) = value.to_anyvalue() {
                    if type_left.value_within_range(lit_val) {
                        st = type_left.clone();
                    }
                }
            }
            // do nothing
            _ => {}
        }
    } else {
        use DataType::*;
        match (type_left, type_right, left, right) {
            // if the we compare a categorical to a literal string we want to cast the literal to categorical
            #[cfg(feature = "dtype-categorical")]
            (Categorical(_), Utf8, _, AExpr::Literal(_))
            | (Utf8, Categorical(_), AExpr::Literal(_), _) => {
                st = DataType::Categorical(None);
            }
            // when then expression literals can have a different list type.
            // so we cast the literal to the other hand side.
            (List(inner), List(other), _, AExpr::Literal(_))
            | (List(other), List(inner), AExpr::Literal(_), _)
                if inner != other =>
            {
                st = DataType::List(inner.clone())
            }
            // do nothing
            _ => {}
        }
    }
    st
}

fn get_input(lp_arena: &Arena<ALogicalPlan>, lp_node: Node) -> [Option<Node>; 2] {
    let plan = lp_arena.get(lp_node);
    let mut inputs = [None, None];

    // Used to get the schema of the input.
    if is_scan(plan) {
        inputs[0] = Some(lp_node);
    } else {
        plan.copy_inputs(&mut inputs);
    };
    inputs
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
                if let Some(input) = get_input(lp_arena, lp_node)[0] {
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
                        let st = modify_supertype(st, truthy, falsy, &type_true, &type_false);

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
                if let Some(input) = get_input(lp_arena, lp_node)[0] {
                    let input_schema = lp_arena.get(input).schema(lp_arena);

                    let left = expr_arena.get(node_left);
                    let right = expr_arena.get(node_right);

                    let type_left = left
                        .get_type(input_schema, Context::Default, expr_arena)
                        .ok()?;
                    let type_right = right
                        .get_type(input_schema, Context::Default, expr_arena)
                        .ok()?;

                    // don't coerce string with number comparisons. They must error
                    match (&type_left, &type_right, op) {
                        #[cfg(not(feature = "dtype-categorical"))]
                        (DataType::Utf8, dt, op) | (dt, DataType::Utf8, op)
                            if op.is_comparison() && dt.is_numeric() =>
                        {
                            return None
                        }
                        #[cfg(feature = "dtype-categorical")]
                        (DataType::Utf8 | DataType::Categorical(_), dt, op)
                        | (dt, DataType::Utf8 | DataType::Categorical(_), op)
                            if op.is_comparison() && dt.is_numeric() =>
                        {
                            return None
                        }
                        _ => {}
                    }

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

                    let list_arithmetic = op.is_arithmetic()
                        && matches!(
                            (&type_left, &type_right),
                            (DataType::List(_), _) | (_, DataType::List(_))
                        );

                    // Special path for list arithmetic
                    if list_arithmetic {
                        match (&type_left, &type_right) {
                            (DataType::List(inner), _) => {
                                return if type_right != **inner {
                                    let new_node_right = expr_arena.add(AExpr::Cast {
                                        expr: node_right,
                                        data_type: *inner.clone(),
                                        strict: false,
                                    });

                                    Some(AExpr::BinaryExpr {
                                        left: node_left,
                                        op,
                                        right: new_node_right,
                                    })
                                } else {
                                    None
                                };
                            }
                            (_, DataType::List(inner)) => {
                                return if type_left != **inner {
                                    let new_node_left = expr_arena.add(AExpr::Cast {
                                        expr: node_left,
                                        data_type: *inner.clone(),
                                        strict: false,
                                    });

                                    Some(AExpr::BinaryExpr {
                                        left: new_node_left,
                                        op,
                                        right: node_right,
                                    })
                                } else {
                                    None
                                };
                            }
                            _ => unreachable!(),
                        }
                    }

                    if type_left == type_right || compare_cat_to_string || datetime_arithmetic {
                        None
                    } else {
                        let st = get_supertype(&type_left, &type_right)
                            .expect("could not find supertype of binary expr");

                        let mut st = modify_supertype(st, left, right, &type_left, &type_right);

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
            #[cfg(feature = "is_in")]
            AExpr::Function {
                function: FunctionExpr::IsIn,
                ref input,
                options,
            } => {
                if let Some(input_node) = get_input(lp_arena, lp_node)[0] {
                    let input_schema = lp_arena.get(input_node).schema(lp_arena);
                    let left_node = input[0];
                    let other_node = input[1];
                    let left = expr_arena.get(left_node);
                    let other = expr_arena.get(other_node);

                    let type_left = left
                        .get_type(input_schema, Context::Default, expr_arena)
                        .ok()?;
                    let type_other = other
                        .get_type(input_schema, Context::Default, expr_arena)
                        .ok()?;

                    match (&type_left, type_other) {
                        (DataType::Categorical(Some(rev_map)), DataType::Utf8)
                            if rev_map.is_global() =>
                        {
                            let mut input = input.clone();

                            let casted_expr = AExpr::Cast {
                                expr: other_node,
                                data_type: DataType::Categorical(None),
                                // does not matter
                                strict: false,
                            };
                            let other_input = expr_arena.add(casted_expr);
                            input[1] = other_input;

                            Some(AExpr::Function {
                                function: FunctionExpr::IsIn,
                                input,
                                options,
                            })
                        }
                        _ => None,
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
