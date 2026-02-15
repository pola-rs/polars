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

/// Extract the underlying integer value from a temporal scalar.
#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
fn extract_temporal_i64(scalar: &Scalar) -> Option<i64> {
    match scalar.value() {
        #[cfg(feature = "dtype-date")]
        AnyValue::Date(v) => Some(i64::from(*v)),
        #[cfg(feature = "dtype-datetime")]
        AnyValue::Datetime(v, _, _) | AnyValue::DatetimeOwned(v, _, _) => Some(*v),
        #[cfg(feature = "dtype-duration")]
        AnyValue::Duration(v, _) => Some(*v),
        _ => None,
    }
}

/// Build a temporal scalar from its physical integer representation by using
/// Polars' cast machinery instead of direct constructor assumptions.
#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
fn build_temporal_scalar_from_repr(dtype: &DataType, val: i64) -> Option<Scalar> {
    let repr_scalar = match dtype {
        #[cfg(feature = "dtype-date")]
        DataType::Date => Scalar::from(i32::try_from(val).ok()?),
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(_, _) => Scalar::from(val),
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(_) => Scalar::from(val),
        _ => return None,
    };

    let out = repr_scalar
        .cast_with_options(dtype, CastOptions::NonStrict)
        .ok()?;
    if out.is_null() { None } else { Some(out) }
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
#[derive(Copy, Clone)]
enum TemporalOutOfBounds {
    BelowMin,
    AboveMax,
}

/// Build the minimum/maximum value scalar for a temporal dtype.
#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
fn build_temporal_bound_scalar(dtype: &DataType, bound: TemporalOutOfBounds) -> Option<Scalar> {
    let val = match (dtype, bound) {
        #[cfg(feature = "dtype-date")]
        (DataType::Date, TemporalOutOfBounds::BelowMin) => i64::from(i32::MIN),
        #[cfg(feature = "dtype-date")]
        (DataType::Date, TemporalOutOfBounds::AboveMax) => i64::from(i32::MAX),
        #[cfg(feature = "dtype-datetime")]
        (DataType::Datetime(_, _), TemporalOutOfBounds::BelowMin) => i64::MIN,
        #[cfg(feature = "dtype-datetime")]
        (DataType::Datetime(_, _), TemporalOutOfBounds::AboveMax) => i64::MAX,
        #[cfg(feature = "dtype-duration")]
        (DataType::Duration(_), TemporalOutOfBounds::BelowMin) => i64::MIN,
        #[cfg(feature = "dtype-duration")]
        (DataType::Duration(_), TemporalOutOfBounds::AboveMax) => i64::MAX,
        _ => return None,
    };
    build_temporal_scalar_from_repr(dtype, val)
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
enum TemporalScalarCast {
    Casted { scalar: Scalar, lossless: bool },
    OutOfBounds(TemporalOutOfBounds),
}

/// Classify a failed temporal cast as underflow/overflow, if possible.
#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
fn classify_temporal_out_of_bounds(
    scalar: &Scalar,
    target: &DataType,
) -> Option<TemporalOutOfBounds> {
    let source_dtype = scalar.dtype();
    let source_val = extract_temporal_i64(scalar)?;

    let min_target = build_temporal_bound_scalar(target, TemporalOutOfBounds::BelowMin)?;
    let max_target = build_temporal_bound_scalar(target, TemporalOutOfBounds::AboveMax)?;

    let min_in_source = min_target
        .cast_with_options(source_dtype, CastOptions::NonStrict)
        .ok()?;
    let max_in_source = max_target
        .cast_with_options(source_dtype, CastOptions::NonStrict)
        .ok()?;

    if min_in_source.is_null() || max_in_source.is_null() {
        return None;
    }

    let min_source_val = extract_temporal_i64(&min_in_source)?;
    let max_source_val = extract_temporal_i64(&max_in_source)?;

    if source_val < min_source_val {
        Some(TemporalOutOfBounds::BelowMin)
    } else if source_val > max_source_val {
        Some(TemporalOutOfBounds::AboveMax)
    } else {
        None
    }
}

/// Try to cast a temporal scalar to a different temporal type.
/// Returns cast info or `None` if the conversion isn't supported.
/// Uses Polars' own `Scalar::cast_with_options` for correctness, then verifies via round-trip.
#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
fn try_cast_temporal_scalar(scalar: &Scalar, target: &DataType) -> Option<TemporalScalarCast> {
    use DataType::*;

    if scalar.is_null() {
        return Some(TemporalScalarCast::Casted {
            scalar: Scalar::null(target.clone()),
            lossless: true,
        });
    }

    // Only handle specific temporal mismatches where pure value conversion is safe.
    // Timezone-aware Date<->Datetime conversions are excluded because they require
    // offset adjustments that depend on timezone rules (DST, etc.).
    // Datetime<->Datetime with equal timezone is safe here: this path only changes
    // time unit on the same epoch timeline and does not perform timezone replacement.
    let is_supported = match (scalar.dtype(), target) {
        #[cfg(all(feature = "dtype-date", feature = "dtype-datetime"))]
        (Datetime(_, tz), Date) | (Date, Datetime(_, tz)) => tz.is_none(),
        #[cfg(feature = "dtype-datetime")]
        (Datetime(_, tz_a), Datetime(_, tz_b)) => tz_a == tz_b,
        #[cfg(feature = "dtype-duration")]
        (Duration(_), Duration(_)) => true,
        _ => false,
    };

    if !is_supported {
        return None;
    }

    // Use Polars' own casting for the conversion.
    let casted = scalar
        .clone()
        .cast_with_options(target, CastOptions::NonStrict)
        .ok()?;

    // Cast produced null (overflow/underflow): classify comparison direction.
    if casted.is_null() {
        return classify_temporal_out_of_bounds(scalar, target)
            .map(TemporalScalarCast::OutOfBounds);
    }

    // Check lossless via round-trip: cast to target and back, compare underlying i64 values.
    let round_tripped = casted
        .clone()
        .cast_with_options(scalar.dtype(), CastOptions::NonStrict)
        .ok()?;

    let original_val = extract_temporal_i64(scalar)?;
    let rt_val = extract_temporal_i64(&round_tripped)?;
    let lossless = original_val == rt_val;

    if lossless {
        return Some(TemporalScalarCast::Casted {
            scalar: casted,
            lossless: true,
        });
    }

    // Lossy: ensure we return the floor value (largest target value <= original).
    // If round-tripped > original, cast landed on the first representable tick above
    // the original (e.g. -1ns -> 0us via truncation toward zero). Moving down by one
    // target tick yields the floor. One tick is sufficient: unit downcasts are integer
    // scale changes, so any upward rounding error is strictly less than one target tick.
    if rt_val > original_val {
        let casted_val = extract_temporal_i64(&casted)?;
        let floor_val = casted_val.checked_sub(1)?;
        let floor_scalar = build_temporal_scalar_from_repr(target, floor_val)?;
        Some(TemporalScalarCast::Casted {
            scalar: floor_scalar,
            lossless: false,
        })
    } else {
        Some(TemporalScalarCast::Casted {
            scalar: casted,
            lossless: false,
        })
    }
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
fn restore_operand_order(op: Operator, lit_on_left: bool) -> Operator {
    if lit_on_left { op.swap_operands() } else { op }
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
fn adjust_lossy_temporal_op(op: Operator) -> Option<Operator> {
    Some(match op {
        Operator::Lt => Operator::LtEq,
        Operator::LtEq => Operator::LtEq,
        Operator::Gt => Operator::Gt,
        Operator::GtEq => Operator::Gt,
        _ => return None,
    })
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
fn out_of_bounds_temporal_op(op: Operator, bound: TemporalOutOfBounds) -> Option<Operator> {
    // Out-of-bounds is not a floor/ceiling rewrite: there is no representable converted literal.
    // This function assumes normalized form `x op l` (column on left), where `l` is out-of-range.
    // We choose `op'` so `x op l` is equivalent to `x op' B` for all representable x.
    //
    // If l > MAX (AboveMax), B = MAX:
    //   x <  l  <=> x <= MAX  (always true for non-null x)   => Lt    -> LtEq
    //   x <= l  <=> x <= MAX  (always true for non-null x)   => LtEq  -> LtEq
    //   x >  l  <=> x >  MAX  (always false for non-null x)  => Gt    -> Gt
    //   x >= l  <=> x >  MAX  (always false for non-null x)  => GtEq  -> Gt
    //   x == l  <=> x >  MAX  (always false for non-null x)  => Eq    -> Gt
    //   x != l  <=> x <= MAX  (always true for non-null x)   => NotEq -> LtEq
    //
    // If l < MIN (BelowMin), B = MIN:
    //   x <  l  <=> x <  MIN  (always false for non-null x)  => Lt    -> Lt
    //   x <= l  <=> x <  MIN  (always false for non-null x)  => LtEq  -> Lt
    //   x >  l  <=> x >= MIN  (always true for non-null x)   => Gt    -> GtEq
    //   x >= l  <=> x >= MIN  (always true for non-null x)   => GtEq  -> GtEq
    //   x == l  <=> x <  MIN  (always false for non-null x)  => Eq    -> Lt
    //   x != l  <=> x >= MIN  (always true for non-null x)   => NotEq -> GtEq
    //
    // Keeping this as a real comparison (instead of constant true/false) preserves null semantics.
    use Operator::*;
    Some(match (bound, op) {
        (TemporalOutOfBounds::AboveMax, Lt) => LtEq,
        (TemporalOutOfBounds::AboveMax, LtEq) => LtEq,
        (TemporalOutOfBounds::AboveMax, Gt) => Gt,
        (TemporalOutOfBounds::AboveMax, GtEq) => Gt,
        (TemporalOutOfBounds::AboveMax, Eq) => Gt,
        (TemporalOutOfBounds::AboveMax, NotEq) => LtEq,
        (TemporalOutOfBounds::BelowMin, Lt) => Lt,
        (TemporalOutOfBounds::BelowMin, LtEq) => Lt,
        (TemporalOutOfBounds::BelowMin, Gt) => GtEq,
        (TemporalOutOfBounds::BelowMin, GtEq) => GtEq,
        (TemporalOutOfBounds::BelowMin, Eq) => Lt,
        (TemporalOutOfBounds::BelowMin, NotEq) => GtEq,
        _ => return None,
    })
}

/// Try to resolve temporal type mismatches at plan time by casting the literal value,
/// avoiding Cast nodes that block I/O-layer predicate pushdown (e.g. Parquet row group pruning).
#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
fn try_cast_temporal_lit_to_column_type(
    expr_arena: &mut Arena<AExpr>,
    node_left: Node,
    op: Operator,
    node_right: Node,
    type_left: &DataType,
    type_right: &DataType,
) -> Option<AExpr> {
    use DataType::*;

    // Check if this is a temporal type mismatch we can handle.
    let is_temporal_mismatch = match (type_left, type_right) {
        #[cfg(all(feature = "dtype-date", feature = "dtype-datetime"))]
        (Date, Datetime(_, tz)) | (Datetime(_, tz), Date) => tz.is_none(),
        #[cfg(feature = "dtype-datetime")]
        (Datetime(tu_a, tz_a), Datetime(tu_b, tz_b)) => tu_a != tu_b && tz_a == tz_b,
        #[cfg(feature = "dtype-duration")]
        (Duration(tu_a), Duration(tu_b)) => tu_a != tu_b,
        _ => false,
    };

    if !is_temporal_mismatch {
        return None;
    }

    // Determine which side is the scalar literal and which is the column.
    let left_scalar = match expr_arena.get(node_left) {
        AExpr::Literal(LiteralValue::Scalar(s)) => Some(s),
        _ => None,
    };
    let right_scalar = match expr_arena.get(node_right) {
        AExpr::Literal(LiteralValue::Scalar(s)) => Some(s),
        _ => None,
    };

    let (scalar, col_node, col_type, lit_on_left) = match (left_scalar, right_scalar) {
        (Some(scalar), None) => (scalar.clone(), node_right, type_right, true),
        (None, Some(scalar)) => (scalar.clone(), node_left, type_left, false),
        // Both literals or neither — don't optimize.
        _ => return None,
    };

    // Normalize to col-on-left perspective for operator adjustment.
    let normalized_op = restore_operand_order(op, lit_on_left);

    let (new_scalar, adjusted_op) = match try_cast_temporal_scalar(&scalar, col_type)? {
        TemporalScalarCast::Casted {
            scalar: new_scalar,
            lossless: true,
        } => (new_scalar, normalized_op),
        TemporalScalarCast::Casted {
            scalar: new_scalar,
            lossless: false,
        } => {
            // Lossy: use the floor-value semantics.
            //   col <  lit -> col <= floor(lit)
            //   col <= lit -> col <= floor(lit)
            //   col >  lit -> col >  floor(lit)
            //   col >= lit -> col >  floor(lit)
            let adjusted_op = adjust_lossy_temporal_op(normalized_op)?;
            (new_scalar, adjusted_op)
        },
        TemporalScalarCast::OutOfBounds(bound) => {
            // Out-of-range is handled separately from lossy-floor:
            // there is no representable floor/ceiling in target dtype. We classify
            // BelowMin/AboveMax, then compare against target MIN/MAX with an adjusted op.
            // This yields the same truth values for non-null rows and preserves null behavior.
            let bound_scalar = match bound {
                TemporalOutOfBounds::AboveMax => {
                    build_temporal_bound_scalar(col_type, TemporalOutOfBounds::AboveMax)?
                },
                TemporalOutOfBounds::BelowMin => {
                    build_temporal_bound_scalar(col_type, TemporalOutOfBounds::BelowMin)?
                },
            };
            let adjusted_op = out_of_bounds_temporal_op(normalized_op, bound)?;
            (bound_scalar, adjusted_op)
        },
    };

    let new_lit_node = expr_arena.add(AExpr::Literal(LiteralValue::Scalar(new_scalar)));
    let final_op = restore_operand_order(adjusted_op, lit_on_left);

    Some(if lit_on_left {
        AExpr::BinaryExpr {
            left: new_lit_node,
            op: final_op,
            right: col_node,
        }
    } else {
        AExpr::BinaryExpr {
            left: col_node,
            op: final_op,
            right: new_lit_node,
        }
    })
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
        _ => {
            unpack!(early_escape(&type_left, &type_right));
        },
    }

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

    // Try to resolve temporal type mismatches at plan time by casting the literal,
    // avoiding Cast nodes that block predicate pushdown to I/O.
    #[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
    if op.is_comparison() {
        if let Some(result) = try_cast_temporal_lit_to_column_type(
            expr_arena,
            node_left,
            op,
            node_right,
            &type_left,
            &type_right,
        ) {
            return Ok(Some(result));
        }
    }

    // Coerce types:
    let st = unpack!(get_supertype(&type_left, &type_right));
    // Re-obtain references after potential mutable borrow above.
    let left = expr_arena.get(node_left);
    let right = expr_arena.get(node_right);
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

#[cfg(test)]
#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
mod tests {
    use polars_core::prelude::TimeUnit;

    use super::*;

    #[test]
    #[cfg(feature = "dtype-date")]
    fn temporal_extract_i64_handles_date_and_non_temporal() {
        let date_scalar = Scalar::new_date(7);
        assert_eq!(extract_temporal_i64(&date_scalar), Some(7));

        let int_scalar = Scalar::from(7i64);
        assert_eq!(extract_temporal_i64(&int_scalar), None);
    }

    #[test]
    #[cfg(feature = "dtype-date")]
    fn temporal_build_scalar_from_repr_handles_date_and_invalid_dtype() {
        let out = build_temporal_scalar_from_repr(&DataType::Date, 5);
        assert!(out.is_some());
        assert_eq!(out.unwrap().dtype(), &DataType::Date);

        let invalid = build_temporal_scalar_from_repr(&DataType::Int64, 5);
        assert!(invalid.is_none());
    }

    #[test]
    #[cfg(feature = "dtype-date")]
    fn temporal_build_bound_scalar_handles_date_and_invalid_dtype() {
        assert!(
            build_temporal_bound_scalar(&DataType::Date, TemporalOutOfBounds::BelowMin).is_some()
        );
        assert!(
            build_temporal_bound_scalar(&DataType::Date, TemporalOutOfBounds::AboveMax).is_some()
        );
        assert!(
            build_temporal_bound_scalar(&DataType::Int64, TemporalOutOfBounds::AboveMax).is_none()
        );
    }

    #[test]
    #[cfg(all(feature = "dtype-date", feature = "dtype-datetime"))]
    fn classify_temporal_out_of_bounds_none_paths() {
        // Datetime bound cast to Date is null => classifier returns None through the early-null path.
        let date_scalar = Scalar::new_date(123);
        assert!(
            classify_temporal_out_of_bounds(
                &date_scalar,
                &DataType::Datetime(TimeUnit::Microseconds, None)
            )
            .is_none()
        );

        // In-range scalar should also classify as None.
        assert!(classify_temporal_out_of_bounds(&date_scalar, &DataType::Date).is_none());
    }

    #[test]
    #[cfg(feature = "dtype-duration")]
    fn try_cast_temporal_scalar_null_and_unsupported_paths() {
        let null_duration = Scalar::null(DataType::Duration(TimeUnit::Microseconds));
        let out =
            try_cast_temporal_scalar(&null_duration, &DataType::Duration(TimeUnit::Nanoseconds));
        assert!(matches!(
            out,
            Some(TemporalScalarCast::Casted { lossless: true, .. })
        ));

        let non_temporal = Scalar::from(1i64);
        assert!(
            try_cast_temporal_scalar(&non_temporal, &DataType::Duration(TimeUnit::Nanoseconds))
                .is_none()
        );
    }

    #[test]
    fn temporal_op_helpers_reject_unsupported_ops() {
        assert_eq!(adjust_lossy_temporal_op(Operator::Eq), None);
        assert_eq!(
            out_of_bounds_temporal_op(Operator::Plus, TemporalOutOfBounds::AboveMax),
            None
        );
    }
}
