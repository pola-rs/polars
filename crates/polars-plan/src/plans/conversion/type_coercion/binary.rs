#[cfg(feature = "dtype-categorical")]
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
                DataType::String | DataType::Unknown(UnknownKind::Str),
                DataType::Categorical(_, _) | DataType::Enum(_, _)
            )
    }
    #[cfg(not(feature = "dtype-categorical"))]
    {
        false
    }
}

#[allow(unused_variables)]
fn is_cat_str_binary(type_left: &DataType, type_right: &DataType) -> bool {
    #[cfg(feature = "dtype-categorical")]
    {
        matches_any_order!(
            type_left,
            type_right,
            DataType::String,
            DataType::Categorical(_, _) | DataType::Enum(_, _)
        )
    }
    #[cfg(not(feature = "dtype-categorical"))]
    {
        false
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
                    options: CastOptions::NonStrict,
                });

                Ok(Some(AExpr::BinaryExpr {
                    left: node_left,
                    op,
                    right: new_node_right,
                }))
            } else {
                Ok(None)
            }
        },
        (_, DataType::List(inner)) => {
            if type_left != **inner {
                let new_node_left = expr_arena.add(AExpr::Cast {
                    expr: node_left,
                    data_type: *inner.clone(),
                    options: CastOptions::NonStrict,
                });

                Ok(Some(AExpr::BinaryExpr {
                    left: new_node_left,
                    op,
                    right: node_right,
                }))
            } else {
                Ok(None)
            }
        },
        _ => unreachable!(),
    }
}

#[cfg(feature = "dtype-struct")]
// Ensure we don't cast to supertype
// otherwise we will fill a struct with null fields
fn process_struct_numeric_arithmetic(
    type_left: DataType,
    type_right: DataType,
    node_left: Node,
    node_right: Node,
    op: Operator,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Option<AExpr>> {
    match (&type_left, &type_right) {
        (DataType::Struct(fields), _) => {
            if let Some(first) = fields.first() {
                let new_node_right = expr_arena.add(AExpr::Cast {
                    expr: node_right,
                    data_type: DataType::Struct(vec![first.clone()]),
                    options: CastOptions::NonStrict,
                });
                Ok(Some(AExpr::BinaryExpr {
                    left: node_left,
                    op,
                    right: new_node_right,
                }))
            } else {
                Ok(None)
            }
        },
        (_, DataType::Struct(fields)) => {
            if let Some(first) = fields.first() {
                let new_node_left = expr_arena.add(AExpr::Cast {
                    expr: node_left,
                    data_type: DataType::Struct(vec![first.clone()]),
                    options: CastOptions::NonStrict,
                });

                Ok(Some(AExpr::BinaryExpr {
                    left: new_node_left,
                    op,
                    right: node_right,
                }))
            } else {
                Ok(None)
            }
        },
        _ => unreachable!(),
    }
}

#[cfg(any(
    feature = "dtype-date",
    feature = "dtype-datetime",
    feature = "dtype-time"
))]
fn err_date_str_compare() -> PolarsResult<()> {
    if cfg!(feature = "python") {
        polars_bail!(
            InvalidOperation:
            "cannot compare 'date/datetime/time' to a string value \
            (create native python {{ 'date', 'datetime', 'time' }} or compare to a temporal column)"
        );
    } else {
        polars_bail!(
            InvalidOperation: "cannot compare 'date/datetime/time' to a string value"
        );
    }
}

pub(super) fn process_binary(
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &Arena<IR>,
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

    match (&type_left, &type_right) {
        (Unknown(UnknownKind::Any), Unknown(UnknownKind::Any)) => return Ok(None),
        (
            Unknown(UnknownKind::Any),
            Unknown(UnknownKind::Int(_) | UnknownKind::Float | UnknownKind::Str),
        ) => {
            let right = unpack!(materialize(right));
            let right = expr_arena.add(right);

            return Ok(Some(AExpr::BinaryExpr {
                left: node_left,
                op,
                right,
            }));
        },
        (
            Unknown(UnknownKind::Int(_) | UnknownKind::Float | UnknownKind::Str),
            Unknown(UnknownKind::Any),
        ) => {
            let left = unpack!(materialize(left));
            let left = expr_arena.add(left);

            return Ok(Some(AExpr::BinaryExpr {
                left,
                op,
                right: node_right,
            }));
        },
        _ => {
            unpack!(early_escape(&type_left, &type_right));
        },
    }

    use DataType::*;
    // don't coerce string with number comparisons. They must error
    match (&type_left, &type_right, op) {
        #[cfg(not(feature = "dtype-categorical"))]
        (DataType::String, dt, op) | (dt, DataType::String, op)
            if op.is_comparison() && dt.is_numeric() =>
        {
            return Ok(None)
        },
        #[cfg(feature = "dtype-categorical")]
        (String | Unknown(UnknownKind::Str) | Categorical(_, _), dt, op)
        | (dt, Unknown(UnknownKind::Str) | String | Categorical(_, _), op)
            if op.is_comparison() && dt.is_numeric() =>
        {
            return Ok(None)
        },
        #[cfg(feature = "dtype-categorical")]
        (Unknown(UnknownKind::Str) | String | Enum(_, _), dt, op)
        | (dt, Unknown(UnknownKind::Str) | String | Enum(_, _), op)
            if op.is_comparison() && dt.is_numeric() =>
        {
            return Ok(None)
        },
        #[cfg(feature = "dtype-date")]
        (Date, String | Unknown(UnknownKind::Str), op)
        | (String | Unknown(UnknownKind::Str), Date, op)
            if op.is_comparison() =>
        {
            err_date_str_compare()?
        },
        #[cfg(feature = "dtype-datetime")]
        (Datetime(_, _), String | Unknown(UnknownKind::Str), op)
        | (String | Unknown(UnknownKind::Str), Datetime(_, _), op)
            if op.is_comparison() =>
        {
            err_date_str_compare()?
        },
        #[cfg(feature = "dtype-time")]
        (Time | Unknown(UnknownKind::Str), String, op) if op.is_comparison() => {
            err_date_str_compare()?
        },
        // structs can be arbitrarily nested, leave the complexity to the caller for now.
        #[cfg(feature = "dtype-struct")]
        (Struct(_), Struct(_), _op) => return Ok(None),
        _ => {},
    }

    if op.is_arithmetic() {
        match (&type_left, &type_right) {
            (Duration(_), Duration(_)) => return Ok(None),
            (Duration(_), r) if r.is_numeric() => return Ok(None),
            (String, a) | (a, String) if a.is_numeric() => {
                polars_bail!(InvalidOperation: "arithmetic on string and numeric not allowed, try an explicit cast first")
            },
            (List(_), _) | (_, List(_)) => {
                return process_list_arithmetic(
                    type_left, type_right, node_left, node_right, op, expr_arena,
                )
            },
            (Datetime(_, _), _)
            | (_, Datetime(_, _))
            | (Date, _)
            | (_, Date)
            | (Duration(_), _)
            | (_, Duration(_))
            | (Time, _)
            | (_, Time) => return Ok(None),
            #[cfg(feature = "dtype-struct")]
            (Struct(_), a) | (a, Struct(_)) if a.is_numeric() => {
                return process_struct_numeric_arithmetic(
                    type_left, type_right, node_left, node_right, op, expr_arena,
                )
            },
            _ => {},
        }
    } else if compares_cat_to_string(&type_left, &type_right, op) {
        return Ok(None);
    }

    // Coerce types:
    let st = unpack!(get_supertype(&type_left, &type_right));
    let mut st = modify_supertype(st, left, right, &type_left, &type_right);

    if is_cat_str_binary(&type_left, &type_right) {
        st = String
    }

    // only cast if the type is not already the super type.
    // this can prevent an expensive flattening and subsequent aggregation
    // in a group_by context. To be able to cast the groups need to be
    // flattened
    let new_node_left = if type_left != st {
        expr_arena.add(AExpr::Cast {
            expr: node_left,
            data_type: st.clone(),
            options: CastOptions::NonStrict,
        })
    } else {
        node_left
    };
    let new_node_right = if type_right != st {
        expr_arena.add(AExpr::Cast {
            expr: node_right,
            data_type: st,
            options: CastOptions::NonStrict,
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
