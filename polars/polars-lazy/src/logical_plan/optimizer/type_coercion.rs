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

fn get_schema(lp_arena: &Arena<ALogicalPlan>, lp_node: Node) -> Option<&SchemaRef> {
    get_input(lp_arena, lp_node)[0].map(|input| lp_arena.get(input).schema(lp_arena))
}

fn get_aexpr_and_type<'a>(
    expr_arena: &'a Arena<AExpr>,
    e: Node,
    input_schema: &Schema,
) -> Option<(&'a AExpr, DataType)> {
    let ae = expr_arena.get(e);
    Some((
        ae,
        ae.get_type(input_schema, Context::Default, expr_arena)
            .ok()?,
    ))
}

fn print_date_str_comparison_warning() {
    eprintln!("Warning: Comparing date/datetime/time column to string value, this will lead to string comparison and is unlikely what you want.\n\
    If this is intended, consider using an explicit cast to silence this warning.")
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
                let input_schema = get_schema(lp_arena, lp_node)?;
                let (truthy, type_true) =
                    get_aexpr_and_type(expr_arena, truthy_node, input_schema)?;
                let (falsy, type_false) = get_aexpr_and_type(expr_arena, falsy_node, input_schema)?;

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
            }
            AExpr::BinaryExpr {
                left: node_left,
                op,
                right: node_right,
            } => {
                let input_schema = get_schema(lp_arena, lp_node)?;
                let (left, type_left) = get_aexpr_and_type(expr_arena, node_left, input_schema)?;
                let (right, type_right) = get_aexpr_and_type(expr_arena, node_right, input_schema)?;

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
                    #[cfg(feature = "dtype-date")]
                    (DataType::Date, DataType::Utf8, op) if op.is_comparison() => {
                        print_date_str_comparison_warning()
                    }
                    #[cfg(feature = "dtype-datetime")]
                    (DataType::Datetime(_, _), DataType::Utf8, op) if op.is_comparison() => {
                        print_date_str_comparison_warning()
                    }
                    #[cfg(feature = "dtype-time")]
                    (DataType::Time, DataType::Utf8, op) if op.is_comparison() => {
                        print_date_str_comparison_warning()
                    }
                    _ => {}
                }

                #[allow(unused_mut, unused_assignments)]
                let mut compare_cat_to_string = false;
                #[cfg(feature = "dtype-categorical")]
                {
                    compare_cat_to_string = matches!(
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
            }
            #[cfg(feature = "is_in")]
            AExpr::Function {
                function: FunctionExpr::IsIn,
                ref input,
                options,
            } => {
                let input_schema = get_schema(lp_arena, lp_node)?;
                let other_node = input[1];
                let (_, type_left) = get_aexpr_and_type(expr_arena, input[0], input_schema)?;
                let (_, type_other) = get_aexpr_and_type(expr_arena, other_node, input_schema)?;

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
            }
            AExpr::Function {
                // only for `DataType::Unknown` as it still has to be set.
                function:
                    FunctionExpr::FillNull {
                        super_type: DataType::Unknown,
                    },
                ref input,
                options,
            } => {
                let input_schema = get_schema(lp_arena, lp_node)?;
                let other_node = input[1];
                let (left, type_left) = get_aexpr_and_type(expr_arena, input[0], input_schema)?;
                let (fill_value, type_fill_value) =
                    get_aexpr_and_type(expr_arena, other_node, input_schema)?;
                let super_type = get_supertype(&type_left, &type_fill_value).ok()?;
                let super_type =
                    modify_supertype(super_type, left, fill_value, &type_left, &type_fill_value);
                Some(AExpr::Function {
                    function: FunctionExpr::FillNull { super_type },
                    input: input.clone(),
                    options,
                })
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
