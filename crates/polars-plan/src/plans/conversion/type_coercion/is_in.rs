use super::*;

#[derive(Debug)]
pub(super) enum IsInTypeCoercionResult {
    SuperType(DataType, DataType),
    SelfCast { dtype: DataType, strict: bool },
    OtherCast { dtype: DataType, strict: bool },
    Implode,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_is_in(
    input: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
    is_contains: bool,
    op: &'static str,
    flat_idx: usize,
    nested_idx: usize,
) -> PolarsResult<Option<IsInTypeCoercionResult>> {
    let (_, type_left) = unpack!(get_aexpr_and_type(
        expr_arena,
        input[flat_idx].node(),
        input_schema
    ));
    let (_, type_other) = unpack!(get_aexpr_and_type(
        expr_arena,
        input[nested_idx].node(),
        input_schema
    ));

    let left_nl = type_left.nesting_level();
    let right_nl = type_other.nesting_level();

    // @HACK. This needs to happen until 3.0 because we support `pl.col.a.is_in(pl.col.a)`.
    if !is_contains && left_nl == right_nl {
        polars_warn!(
            Deprecation,
            "`is_in` with a collection of the same datatype is ambiguous and deprecated.
Please use `implode` to return to previous behavior.

See https://github.com/pola-rs/polars/issues/22149 for more information."
        );
        return Ok(Some(IsInTypeCoercionResult::Implode));
    }

    let wrap_other = |resolved_inner_type: DataType| match &type_other {
        DataType::List(_) => DataType::List(Box::new(resolved_inner_type)),
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, width) => DataType::Array(Box::new(resolved_inner_type), *width),
        _ => unreachable!(),
    };

    let type_left_materialized = type_left.clone().materialize_unknown(false)?;
    let Some(type_other_inner) = type_other.inner_dtype() else {
        panic!();
        polars_bail!(InvalidOperation: "'{op:?}' cannot check for {type_left:?} values in {type_other:?} data.\n\
        Hint: container dtype ({type_other:?}) must be nested");
    };

    let casted_inner_expr = match (&type_left_materialized, type_other_inner) {
        // Types are equal, do nothing
        (dtml, dto) if dtml == dto => return Ok(None),

        // All-null can represent anything (and/or empty list), so cast to target dtype
        (DataType::Null, _) => IsInTypeCoercionResult::SelfCast {
            dtype: type_other_inner.clone(),
            strict: false,
        },
        (_, DataType::Null) => IsInTypeCoercionResult::OtherCast {
            dtype: wrap_other(type_left_materialized),
            strict: false,
        },

        #[cfg(feature = "dtype-categorical")]
        (DataType::Enum(_, _), DataType::String) => IsInTypeCoercionResult::OtherCast {
            dtype: wrap_other(type_left_materialized),
            strict: true,
        },
        #[cfg(feature = "dtype-categorical")]
        (DataType::String, DataType::Enum(_, _)) => IsInTypeCoercionResult::SelfCast {
            dtype: type_other_inner.clone(),
            strict: true,
        },
        #[cfg(feature = "dtype-categorical")]
        (DataType::String, DataType::Categorical(_, _)) => IsInTypeCoercionResult::SelfCast {
            dtype: type_other_inner.clone(),
            strict: false,
        },
        #[cfg(feature = "dtype-categorical")]
        (DataType::Categorical(_, _), DataType::String) => IsInTypeCoercionResult::OtherCast {
            dtype: wrap_other(type_left_materialized),
            strict: false,
        },

        #[cfg(feature = "dtype-decimal")]
        (DataType::Decimal(_, _), dt) if dt.is_primitive_numeric() => {
            IsInTypeCoercionResult::OtherCast {
                dtype: wrap_other(type_left_materialized),
                strict: false,
            }
        },
        #[cfg(feature = "dtype-decimal")]
        (DataType::Decimal(_, _), _) | (_, DataType::Decimal(_, _)) => {
            polars_bail!(InvalidOperation: "'{op}' cannot check for {type_left:?} values in {type_other:?} data")
        },
        // can't check for more granular time_unit in less-granular time_unit data,
        // or we'll cast away valid/necessary precision (eg: nanosecs to millisecs)
        (DataType::Datetime(lhs_unit, _), DataType::Datetime(rhs_unit, _)) => {
            if lhs_unit <= rhs_unit {
                return Ok(None);
            } else {
                polars_bail!(InvalidOperation: "'{op}' cannot check for {rhs_unit:?} precision values in {lhs_unit:?} Datetime data")
            }
        },
        (DataType::Duration(lhs_unit), DataType::Duration(rhs_unit)) => {
            if lhs_unit <= rhs_unit {
                return Ok(None);
            } else {
                polars_bail!(InvalidOperation: "'{op}' cannot check for {rhs_unit:?} precision values in {lhs_unit:?} Duration data")
            }
        },

        // Don't attempt to cast between obviously mismatched types. Only allow
        // to cast to a supertype if the cast is lossless.
        (dtml, dto) => {
            if (dtml.is_primitive_numeric() && dto.is_primitive_numeric()) || dtml.is_null() {
                if let Some(super_type) = get_numeric_upcast_supertype_lossless(dtml, dto) {
                    return Ok(Some(IsInTypeCoercionResult::SuperType(
                        super_type.clone(),
                        wrap_other(super_type),
                    )));
                } else {
                    // We disabled lossless coercion of the operands in 2.0.
                    let lossy_supertype = try_get_supertype(dtml, dto)?;
                    polars_bail!(InvalidOperation: "'{op}' cannot check for {type_left:?} values in {type_other:?} data.\n\
                        Hint: Before version 2.0, Polars would perform this check by lossily coercing the operands to {lossy_supertype:?}. \
                        However, since Polars 2.0, for is_in() it is required to explicitly cast (one of) the operands to a compatible type.")
                }
            }
            polars_bail!(InvalidOperation: "'{op}' cannot check for {type_left:?} values in {type_other:?} data")
        },
    };
    Ok(Some(casted_inner_expr))
}
