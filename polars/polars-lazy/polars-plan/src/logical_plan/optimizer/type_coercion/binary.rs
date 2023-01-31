use polars_utils::matches_any_order;

use super::*;

macro_rules! unpack {
    ($packed:expr) => {{
        match $packed {
            Some(payload) => payload,
            None => return Ok(None),
        }
    }};
}

#[allow(unused_variables)]
fn compares_cat_to_string(type_left: &DataType, type_right: &DataType, op: Operator) -> bool {
    #[cfg(feature = "dtype-categorical")]
    {
        op.is_comparison()
            && matches_any_order!(
                type_left,
                type_right,
                DataType::Utf8,
                DataType::Categorical(_)
            )
    }
    #[cfg(not(feature = "dtype-categorical"))]
    {
        false
    }
}

#[allow(unused_variables)]
fn is_datetime_arithmetic(type_left: &DataType, type_right: &DataType, op: Operator) -> bool {
    matches!(op, Operator::Minus | Operator::Plus)
        && matches_any_order!(
            &type_left,
            &type_right,
            DataType::Datetime(_, _) | DataType::Date,
            DataType::Duration(_)
        )
}

fn is_list_arithmetic(type_left: &DataType, type_right: &DataType, op: Operator) -> bool {
    op.is_arithmetic()
        && matches!(
            (&type_left, &type_right),
            (DataType::List(_), _) | (_, DataType::List(_))
        )
}

#[allow(unused_variables)]
fn is_cat_str_binary(type_left: &DataType, type_right: &DataType) -> bool {
    #[cfg(feature = "dtype-categorical")]
    {
        matches_any_order!(
            type_left,
            type_right,
            DataType::Utf8,
            DataType::Categorical(_)
        )
    }
    #[cfg(not(feature = "dtype-categorical"))]
    {
        false
    }
}

fn str_numeric_arithmetic(type_left: &DataType, type_right: &DataType) -> PolarsResult<()> {
    if type_left.is_numeric() && matches!(type_right, DataType::Utf8)
        || type_right.is_numeric() && matches!(type_left, DataType::Utf8)
    {
        Err(PolarsError::ComputeError(
            "Arithmetic on string and numeric not allowed. Try an explicit cast first.".into(),
        ))
    } else {
        Ok(())
    }
}

fn process_list_arithmetic(
    type_left: DataType,
    type_right: DataType,
    node_left: Node,
    node_right: Node,
    op: Operator,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Option<AExpr>> {
    match (&type_left, &type_right) {
        (DataType::List(inner), _) => {
            if type_right != **inner {
                let new_node_right = expr_arena.add(AExpr::Cast {
                    expr: node_right,
                    data_type: *inner.clone(),
                    strict: false,
                });

                Ok(Some(AExpr::BinaryExpr {
                    left: node_left,
                    op,
                    right: new_node_right,
                }))
            } else {
                Ok(None)
            }
        }
        (_, DataType::List(inner)) => {
            if type_left != **inner {
                let new_node_left = expr_arena.add(AExpr::Cast {
                    expr: node_left,
                    data_type: *inner.clone(),
                    strict: false,
                });

                Ok(Some(AExpr::BinaryExpr {
                    left: new_node_left,
                    op,
                    right: node_right,
                }))
            } else {
                Ok(None)
            }
        }
        _ => unreachable!(),
    }
}

pub(super) fn process_binary(
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &Arena<ALogicalPlan>,
    lp_node: Node,
    node_left: Node,
    op: Operator,
    node_right: Node,
) -> PolarsResult<Option<AExpr>> {
    let input_schema = get_schema(lp_arena, lp_node);
    let (left, type_left): (&AExpr, DataType) =
        unpack!(get_aexpr_and_type(expr_arena, node_left, &input_schema));
    let (right, type_right): (&AExpr, DataType) =
        unpack!(get_aexpr_and_type(expr_arena, node_right, &input_schema));
    unpack!(early_escape(&type_left, &type_right));

    use DataType::*;
    // don't coerce string with number comparisons. They must error
    match (&type_left, &type_right, op) {
        #[cfg(not(feature = "dtype-categorical"))]
        (DataType::Utf8, dt, op) | (dt, DataType::Utf8, op)
            if op.is_comparison() && dt.is_numeric() =>
        {
            return Ok(None)
        }
        #[cfg(feature = "dtype-categorical")]
        (Utf8 | Categorical(_), dt, op) | (dt, Utf8 | Categorical(_), op)
            if op.is_comparison() && dt.is_numeric() =>
        {
            return Ok(None)
        }
        #[cfg(feature = "dtype-date")]
        (Date, Utf8, op) if op.is_comparison() => err_date_str_compare()?,
        #[cfg(feature = "dtype-datetime")]
        (Datetime(_, _), Utf8, op) if op.is_comparison() => err_date_str_compare()?,
        #[cfg(feature = "dtype-time")]
        (Time, Utf8, op) if op.is_comparison() => err_date_str_compare()?,
        // structs can be arbitrarily nested, leave the complexity to the caller for now.
        #[cfg(feature = "dtype-struct")]
        (Struct(_), Struct(_), _op) => return Ok(None),
        _ => {}
    }
    let compare_cat_to_string = compares_cat_to_string(&type_left, &type_right, op);
    let datetime_arithmetic = is_datetime_arithmetic(&type_left, &type_right, op);
    let list_arithmetic = is_list_arithmetic(&type_left, &type_right, op);
    str_numeric_arithmetic(&type_left, &type_right)?;

    // Special path for list arithmetic
    if list_arithmetic {
        return process_list_arithmetic(
            type_left, type_right, node_left, node_right, op, expr_arena,
        );
    }

    // All early return paths
    if compare_cat_to_string
        || datetime_arithmetic
        || early_escape(&type_left, &type_right).is_none()
    {
        Ok(None)
    } else {
        // Coerce types:

        let st = unpack!(get_supertype(&type_left, &type_right));
        let mut st = modify_supertype(st, left, right, &type_left, &type_right);

        if is_cat_str_binary(&type_left, &type_right) {
            st = Utf8
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

        Ok(Some(AExpr::BinaryExpr {
            left: new_node_left,
            op,
            right: new_node_right,
        }))
    }
}
