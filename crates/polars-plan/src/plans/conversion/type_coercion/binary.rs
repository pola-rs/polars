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
        op.is_comparison_or_bitwise()
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
                    dtype: DataType::Struct(vec![first.clone()]),
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
                    dtype: DataType::Struct(vec![first.clone()]),
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
    input_schema: &Schema,
    node_left: Node,
    op: Operator,
    node_right: Node,
) -> PolarsResult<Option<AExpr>> {
    let (left, type_left): (&AExpr, DataType) =
        unpack!(get_aexpr_and_type(expr_arena, node_left, input_schema));
    let (right, type_right): (&AExpr, DataType) =
        unpack!(get_aexpr_and_type(expr_arena, node_right, input_schema));

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
        _ => {},
    }

    if op.is_comparison()
        && let Some(rewrite) = (match (left, right) {
            (_, AExpr::Literal(lv)) => {
                if let LiteralValue::Scalar(s) = lv.clone().materialize() {
                    coerce_comparison_literal(node_left, &type_left, op, s.clone(), expr_arena)
                } else {
                    None
                }
            },
            (AExpr::Literal(lv), _) => {
                if let LiteralValue::Scalar(s) = lv.clone().materialize() {
                    coerce_comparison_literal(
                        node_right,
                        &type_right,
                        op.swap_operands(),
                        s,
                        expr_arena,
                    )
                    .map(|rewrite| match rewrite {
                        CmpLiteralRhsRewrite::ReplaceLit(new_lit) => {
                            CmpLiteralRhsRewrite::NewAExpr {
                                aexpr: AExpr::BinaryExpr {
                                    left: node_right,
                                    op: op.swap_operands(),
                                    right: expr_arena
                                        .add(AExpr::Literal(LiteralValue::Scalar(new_lit))),
                                },
                                output_constraint: None,
                            }
                        },
                        CmpLiteralRhsRewrite::NewAExpr { .. } => rewrite,
                    })
                } else {
                    None
                }
            },
            _ => None,
        })
    {
        return Ok(Some(match rewrite {
            CmpLiteralRhsRewrite::ReplaceLit(new_lit) => AExpr::BinaryExpr {
                left: node_left,
                op,
                right: expr_arena.add(AExpr::Literal(LiteralValue::Scalar(new_lit))),
            },
            CmpLiteralRhsRewrite::NewAExpr {
                aexpr,
                output_constraint: _,
            } => aexpr,
        }));
    }

    unpack!(early_escape(&type_left, &type_right));

    let left = expr_arena.get(node_left);
    let right = expr_arena.get(node_right);

    use DataType::*;
    // don't coerce string with number comparisons. They must error
    match (&type_left, &type_right, op) {
        #[cfg(not(feature = "dtype-categorical"))]
        (DataType::String, dt, op) | (dt, DataType::String, op)
            if op.is_comparison_or_bitwise() && dt.is_primitive_numeric() =>
        {
            return Ok(None);
        },
        #[cfg(feature = "dtype-categorical")]
        (String | Unknown(UnknownKind::Str) | Categorical(_, _), dt, op)
        | (dt, Unknown(UnknownKind::Str) | String | Categorical(_, _), op)
            if op.is_comparison_or_bitwise() && dt.is_primitive_numeric() =>
        {
            return Ok(None);
        },
        #[cfg(feature = "dtype-categorical")]
        (Unknown(UnknownKind::Str) | String | Enum(_, _), dt, op)
        | (dt, Unknown(UnknownKind::Str) | String | Enum(_, _), op)
            if op.is_comparison_or_bitwise() && dt.is_primitive_numeric() =>
        {
            return Ok(None);
        },
        #[cfg(feature = "dtype-date")]
        (Date, String | Unknown(UnknownKind::Str), op)
        | (String | Unknown(UnknownKind::Str), Date, op)
            if op.is_comparison_or_bitwise() =>
        {
            err_date_str_compare()?
        },
        #[cfg(feature = "dtype-datetime")]
        (Datetime(_, _), String | Unknown(UnknownKind::Str), op)
        | (String | Unknown(UnknownKind::Str), Datetime(_, _), op)
            if op.is_comparison_or_bitwise() =>
        {
            err_date_str_compare()?
        },
        #[cfg(feature = "dtype-time")]
        (Time | Unknown(UnknownKind::Str), String, op) if op.is_comparison_or_bitwise() => {
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
            (Duration(_), r) if r.is_primitive_numeric() => return Ok(None),
            (String, a) | (a, String) if a.is_primitive_numeric() => {
                polars_bail!(InvalidOperation: "arithmetic on string and numeric not allowed, try an explicit cast first")
            },
            (Datetime(_, _), _)
            | (_, Datetime(_, _))
            | (Date, _)
            | (_, Date)
            | (Duration(_), _)
            | (_, Duration(_))
            | (Time, _)
            | (_, Time)
            | (List(_), _)
            | (_, List(_)) => return Ok(None),
            #[cfg(feature = "dtype-array")]
            (Array(..), _) | (_, Array(..)) => return Ok(None),
            #[cfg(feature = "dtype-struct")]
            (Struct(_), a) | (a, Struct(_)) if a.is_primitive_numeric() => {
                return process_struct_numeric_arithmetic(
                    type_left, type_right, node_left, node_right, op, expr_arena,
                );
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

    // TODO! raise here?
    // We should at least never cast to Unknown.
    if matches!(st, DataType::Unknown(UnknownKind::Any)) {
        return Ok(None);
    }

    // Only cast if the type is not already the super type.
    // this can prevent an expensive flattening and subsequent aggregation
    // in a group_by context. To be able to cast the groups need to be
    // flattened
    let new_node_left = if type_left != st {
        expr_arena.add(AExpr::Cast {
            expr: node_left,
            dtype: st.clone(),
            options: CastOptions::NonStrict,
        })
    } else {
        node_left
    };
    let new_node_right = if type_right != st {
        expr_arena.add(AExpr::Cast {
            expr: node_right,
            dtype: st,
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

pub(super) enum CmpLiteralRhsRewrite {
    /// Replace the RHS literal of the original expr
    ReplaceLit(Scalar),
    /// Rewritten to a new expression.
    NewAExpr {
        aexpr: AExpr,
        output_constraint: Option<BoolValueAlways>,
    },
}

pub(super) enum BoolValueAlways {
    FalseOrNull,
    True,
    TrueOrNull,
}

/// Attempt to coerce a scalar RHS of a comparison to the dtype of the LHS.
/// This can avoid a cast insertion, which can prevent filters from being applied in scans.
pub(super) fn coerce_comparison_literal(
    ae_node_left: Node,
    dtype_lhs: &DataType,
    cmp_op: Operator,
    lit_rhs: Scalar,
    expr_arena: &mut Arena<AExpr>,
) -> Option<CmpLiteralRhsRewrite> {
    use CmpLiteralRhsRewrite::*;

    #[rustfmt::skip]
    fn matching_and_supported_dtype_class(dtype_lhs: &DataType, dtype_rhs: &DataType) -> bool {
        (dtype_lhs.is_integer() && dtype_rhs.is_integer())
        || (
            dtype_lhs.is_datetime()
            && (dtype_rhs.is_datetime() || dtype_rhs.is_date())
        )
        || (dtype_lhs.is_duration() && dtype_rhs.is_duration())
    }

    fn supertype_introduces_nulls_on_lhs(dtype_lhs: &DataType, supertype: &DataType) -> bool {
        dtype_lhs.is_integer()
            && dtype_lhs != supertype
            && get_numeric_upcast_supertype_lossless(dtype_lhs, supertype).as_ref()
                != Some(supertype)
    }

    let supertype = get_supertype(dtype_lhs, lit_rhs.dtype())?;

    macro_rules! finish_rewrite_cmp_with_null_literal {
        () => {{
            let ir_boolean_function = match cmp_op {
                Operator::EqValidity => IRBooleanFunction::IsNull,
                Operator::NotEqValidity => IRBooleanFunction::IsNotNull,
                _ => {
                    return Some(NewAExpr {
                        aexpr: repeat_opt_bool_ae(None, ae_node_left, expr_arena),
                        output_constraint: Some(BoolValueAlways::FalseOrNull),
                    });
                },
            };

            if supertype_introduces_nulls_on_lhs(dtype_lhs, &supertype) {
                return None;
            }

            let function = IRFunctionExpr::Boolean(ir_boolean_function);
            let options = function.function_options();

            return Some(NewAExpr {
                aexpr: AExpr::Function {
                    input: vec![ExprIR::from_node(ae_node_left, expr_arena)],
                    function,
                    options,
                },
                output_constraint: None,
            });
        }};
    }

    if lit_rhs.is_null() {
        finish_rewrite_cmp_with_null_literal!();
    }

    if dtype_lhs == lit_rhs.dtype() {
        return None;
    }

    if !matching_and_supported_dtype_class(dtype_lhs, lit_rhs.dtype()) {
        return None;
    }

    if !matching_and_supported_dtype_class(dtype_lhs, &supertype) {
        // Reject integer comparisons that compare as floats (e.g. Int64<>UInt64).
        // E.g. See below, the 2nd and 3rd compare as true due to (i64 as f64)<>(u64 as f64) comparison.
        // DataFrame(
        //     [
        //         Series("i64", 3 * [((1 << 63) - 1)], dtype=pl.Int64),
        //         Series("u64", [(1 << 63) - 1, 1 << 63, (1 << 63) + 1], dtype=pl.UInt64),
        //     ]
        // ).with_columns(eq=pl.col("i64") == pl.col("u64"))
        // shape: (3, 3)
        // ┌─────────────────────┬─────────────────────┬──────┐
        // │ i64                 ┆ u64                 ┆ eq   │
        // │ ---                 ┆ ---                 ┆ ---  │
        // │ i64                 ┆ u64                 ┆ bool │
        // ╞═════════════════════╪═════════════════════╪══════╡
        // │ 9223372036854775807 ┆ 9223372036854775807 ┆ true │
        // │ 9223372036854775807 ┆ 9223372036854775808 ┆ true │  <--- We would return `false` on these rows
        // │ 9223372036854775807 ┆ 9223372036854775809 ┆ true │  <----^
        // └─────────────────────┴─────────────────────┴──────┘
        return None;
    }

    if supertype_introduces_nulls_on_lhs(dtype_lhs, &supertype) {
        // Reject integer comparisons that cast to a supertype that introduces NULLs on the LHS.
        // E.g. See below, the 2nd and 3rd row became NULL due to u128->i128 cast:
        // DataFrame(
        //     [
        //         Series("u128", [(1 << 127) - 1, 1 << 127, (1 << 128) - 1], dtype=pl.UInt128),
        //         Series("i16", 3 * [-1], dtype=pl.Int16),
        //     ]
        // ).with_columns(eq=pl.col("u128") == pl.col("i16"))
        // shape: (3, 3)
        // ┌─────────────────────────────────┬─────┬───────┐
        // │ u128                            ┆ i16 ┆ eq    │
        // │ ---                             ┆ --- ┆ ---   │
        // │ u128                            ┆ i16 ┆ bool  │
        // ╞═════════════════════════════════╪═════╪═══════╡
        // │ 170141183460469231731687303715… ┆ -1  ┆ false │
        // │ 170141183460469231731687303715… ┆ -1  ┆ null  │  <--- We would return `false` on these rows
        // │ 340282366920938463463374607431… ┆ -1  ┆ null  │  <----^
        // └─────────────────────────────────┴─────┴───────┘
        return None;
    }

    let lit_rhs_casted: Option<Scalar> = lit_rhs
        .clone()
        .cast_with_options(dtype_lhs, CastOptions::NonStrict)
        .ok()
        .filter(|x| !x.is_null());

    if let Some(lit_rhs_casted) = lit_rhs_casted {
        // Datetime comparison upcast goes to the lower precision type, which causes flooring to
        // have an effect on Eq and LtEq comparisons. Here, we compute when instead casting to the
        // higher precision type, the value corresponding to the next multiple of the lower precision
        // source type, then subtract 1 from this value.
        //
        // E.g. Consider: Datetime[ns] 2026-01-01 00:00:00.000000001 == Datetime[us] 2026-01-01 00:00:00.000000,
        // which must compare equal.
        // [us]->[ns] cast becomes 2026-01-01 00:00:00.000000000, then we need to increase it
        // to 2026-01-01 00:00:00.000000999 to preserve the comparison behavior.
        let lit_casted_upper_equality_bound: Option<AnyValue<'static>> =
            match (lit_rhs.dtype(), lit_rhs_casted.as_any_value()) {
                #[cfg(feature = "dtype-datetime")]
                (
                    DataType::Datetime(TimeUnit::Milliseconds, _),
                    AnyValue::DatetimeOwned(phys_i64, tu, tz),
                ) => {
                    let unit_multiple: i64 = match tu {
                        TimeUnit::Milliseconds => 0,
                        TimeUnit::Microseconds => 1_000,
                        TimeUnit::Nanoseconds => 1_000_000,
                    };

                    (unit_multiple > 0).then_some(AnyValue::DatetimeOwned(
                        i64::saturating_add(phys_i64, unit_multiple.wrapping_sub(1)),
                        tu,
                        tz,
                    ))
                },
                #[cfg(feature = "dtype-datetime")]
                (
                    DataType::Datetime(TimeUnit::Microseconds, _),
                    AnyValue::DatetimeOwned(phys_i64, tu, tz),
                ) => {
                    let unit_multiple: i64 = match tu {
                        TimeUnit::Milliseconds => 0,
                        TimeUnit::Microseconds => 0,
                        TimeUnit::Nanoseconds => 1_000,
                    };

                    (unit_multiple > 0).then_some(AnyValue::DatetimeOwned(
                        i64::saturating_add(phys_i64, unit_multiple.wrapping_sub(1)),
                        tu,
                        tz,
                    ))
                },
                #[cfg(feature = "dtype-duration")]
                (DataType::Duration(TimeUnit::Milliseconds), AnyValue::Duration(phys_i64, tu)) => {
                    let unit_multiple: i64 = match tu {
                        TimeUnit::Milliseconds => 0,
                        TimeUnit::Microseconds => 1_000,
                        TimeUnit::Nanoseconds => 1_000_000,
                    };

                    (unit_multiple > 0).then_some(AnyValue::Duration(
                        i64::saturating_add(phys_i64, unit_multiple.wrapping_sub(1)),
                        tu,
                    ))
                },
                #[cfg(feature = "dtype-duration")]
                (DataType::Duration(TimeUnit::Microseconds), AnyValue::Duration(phys_i64, tu)) => {
                    let unit_multiple: i64 = match tu {
                        TimeUnit::Milliseconds => 0,
                        TimeUnit::Microseconds => 0,
                        TimeUnit::Nanoseconds => 1_000,
                    };

                    (unit_multiple > 0).then_some(AnyValue::Duration(
                        i64::saturating_add(phys_i64, unit_multiple.wrapping_sub(1)),
                        tu,
                    ))
                },
                _ => None,
            };

        if dtype_lhs.is_temporal() {
            match cmp_op {
                Operator::LtEq => {
                    let upper_lit =
                        Scalar::new(dtype_lhs.clone(), lit_casted_upper_equality_bound?);
                    return Some(ReplaceLit(upper_lit));
                },
                Operator::Eq => {
                    // E.g.
                    // In: datetime[ms] == 2026-01-01 (Date)
                    // Out: (datetime[ms]).is_between(2026-01-01 00:00:00.000, 2026-01-01 23:59:59.999, closed='both')
                    return {
                        #[cfg(feature = "is_between")]
                        {
                            let upper_lit =
                                Scalar::new(dtype_lhs.clone(), lit_casted_upper_equality_bound?);
                            let node_high =
                                expr_arena.add(AExpr::Literal(LiteralValue::Scalar(upper_lit)));

                            let node_low = expr_arena
                                .add(AExpr::Literal(LiteralValue::Scalar(lit_rhs_casted)));

                            let function = IRFunctionExpr::Boolean(IRBooleanFunction::IsBetween {
                                closed: ClosedInterval::Both,
                            });
                            let options = function.function_options();

                            Some(NewAExpr {
                                aexpr: AExpr::Function {
                                    input: vec![
                                        ExprIR::from_node(ae_node_left, expr_arena),
                                        ExprIR::from_node(node_low, expr_arena),
                                        ExprIR::from_node(node_high, expr_arena),
                                    ],
                                    function,
                                    options,
                                },
                                output_constraint: None,
                            })
                        }
                        #[cfg(not(feature = "is_between"))]
                        {
                            None
                        }
                    };
                },
                _ => {},
            }
        }

        return Some(ReplaceLit(lit_rhs_casted));
    }

    let nonnull_rows_value: bool = if lit_rhs
        .clone()
        .cast_with_options(&supertype, CastOptions::NonStrict)
        .ok()?
        .is_null()
    {
        finish_rewrite_cmp_with_null_literal!()
    } else {
        // Since the literal is out of range it's either below min or above max. Determining the side
        // can be done by comparing against any in-range value (we pick 0).
        let rhs_phys_lt0 = || {
            let lit_rhs = lit_rhs.clone().to_physical();

            if lit_rhs.dtype().is_unsigned_integer() {
                return Some(false);
            }

            let AnyValue::Int128(x) = lit_rhs
                .cast_with_options(&DataType::Int128, CastOptions::NonStrict)
                .ok()
                .filter(|x| !x.is_null())?
                .as_any_value()
            else {
                unreachable!()
            };

            Some(x < 0)
        };

        match cmp_op {
            Operator::Eq => false,
            Operator::NotEq => true,
            Operator::Lt | Operator::LtEq => !rhs_phys_lt0()?,
            Operator::Gt | Operator::GtEq => rhs_phys_lt0()?,
            Operator::EqValidity => {
                return Some(NewAExpr {
                    aexpr: repeat_opt_bool_ae(Some(false), ae_node_left, expr_arena),
                    output_constraint: Some(BoolValueAlways::FalseOrNull),
                });
            },
            Operator::NotEqValidity => {
                return Some(NewAExpr {
                    aexpr: repeat_opt_bool_ae(Some(true), ae_node_left, expr_arena),
                    output_constraint: Some(BoolValueAlways::True),
                });
            },
            _ => return None,
        }
    };

    // Set all non-null rows to either true / false.
    // when(left.is_not_null()).then(nonnull_rows_value)

    let left_not_null = {
        let function = IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull);
        let options = function.function_options();
        expr_arena.add(AExpr::Function {
            input: vec![ExprIR::from_node(ae_node_left, expr_arena)],
            function,
            options,
        })
    };

    Some(NewAExpr {
        aexpr: AExpr::Ternary {
            predicate: left_not_null,
            truthy: expr_arena.add(AExpr::Literal(LiteralValue::Scalar(Scalar::new(
                DataType::Boolean,
                AnyValue::Boolean(nonnull_rows_value),
            )))),
            falsy: expr_arena.add(AExpr::Literal(LiteralValue::Scalar(Scalar::new(
                DataType::Boolean,
                AnyValue::Null,
            )))),
        },
        output_constraint: Some(if nonnull_rows_value {
            BoolValueAlways::TrueOrNull
        } else {
            BoolValueAlways::FalseOrNull
        }),
    })
}

fn repeat_opt_bool_ae(value: Option<bool>, len_of: Node, expr_arena: &mut Arena<AExpr>) -> AExpr {
    let value_scalar_lit = AExpr::Literal(LiteralValue::Scalar(value.map_or_else(
        || Scalar::null(DataType::Boolean),
        |v| Scalar::new(DataType::Boolean, AnyValue::Boolean(v)),
    )));

    if is_scalar_ae(len_of, expr_arena) {
        return value_scalar_lit;
    }

    let function = IRFunctionExpr::Repeat;
    let options = function.function_options();

    AExpr::Function {
        input: vec![
            ExprIR::from_node(expr_arena.add(value_scalar_lit), expr_arena),
            ExprIR::from_node(
                expr_arena.add(AExpr::Agg(IRAggExpr::Count {
                    input: len_of,
                    include_nulls: true,
                })),
                expr_arena,
            ),
        ],
        function,
        options,
    }
}
