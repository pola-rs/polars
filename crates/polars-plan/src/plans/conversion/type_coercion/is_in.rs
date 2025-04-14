use super::*;

pub(super) enum IsInTypeCoercionResult {
    SuperType(DataType, DataType),
    OtherCast(DataType),
    Implode,
}

pub(super) fn resolve_is_in(
    input: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    lp_arena: &Arena<IR>,
    lp_node: Node,
) -> PolarsResult<Option<IsInTypeCoercionResult>> {
    let input_schema = get_schema(lp_arena, lp_node);
    let other_e = &input[1];
    let (_, type_left) = unpack!(get_aexpr_and_type(
        expr_arena,
        input[0].node(),
        &input_schema
    ));
    let (_, type_other) = unpack!(get_aexpr_and_type(
        expr_arena,
        other_e.node(),
        &input_schema
    ));

    let left_nl = type_left.nesting_level();
    let right_nl = type_other.nesting_level();

    // @HACK. This needs to happen until 2.0 because we support `pl.col.a.is_in(pl.col.a)`.
    if left_nl == right_nl {
        polars_warn!(
            Deprecation,
            "`is_in` with a collection of the same datatype is ambiguous and deprecated.
Please use `implode` to return to previous behavior.

See https://github.com/pola-rs/polars/issues/22149 for more information."
        );
        return Ok(Some(IsInTypeCoercionResult::Implode));
    }

    if left_nl + 1 != right_nl {
        polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} values in {:?} data", &type_other, &type_left);
    }

    let type_other_inner = type_other.inner_dtype().unwrap();

    unpack!(early_escape(&type_left, type_other_inner));

    let cast_type = match &type_other {
        DataType::List(_) => DataType::List(Box::new(type_left.clone())),
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, width) => DataType::Array(Box::new(type_left.clone()), *width),
        _ => unreachable!(),
    };

    let casted_expr = match (&type_left, type_other_inner) {
        // types are equal, do nothing
        (a, b) if a == b => return Ok(None),
        // all-null can represent anything (and/or empty list), so cast to target dtype
        (_, DataType::Null) => IsInTypeCoercionResult::OtherCast(cast_type),
        #[cfg(feature = "dtype-categorical")]
        (DataType::Enum(_, _), DataType::String) => IsInTypeCoercionResult::OtherCast(cast_type),

        // @NOTE: Local Categorical coercion has to happen in the kernel, which makes it streaming
        // incompatible.
        #[cfg(feature = "dtype-categorical")]
        (DataType::Categorical(Some(rm), ordering), DataType::String) if rm.is_global() => {
            IsInTypeCoercionResult::OtherCast(match &type_other {
                DataType::List(_) => {
                    DataType::List(Box::new(DataType::Categorical(None, *ordering)))
                },
                #[cfg(feature = "dtype-array")]
                DataType::Array(_, width) => {
                    DataType::Array(Box::new(DataType::Categorical(None, *ordering)), *width)
                },
                _ => unreachable!(),
            })
        },

        #[cfg(feature = "dtype-categorical")]
        (DataType::Categorical(_, _), DataType::String) => return Ok(None),
        #[cfg(feature = "dtype-categorical")]
        (DataType::String, DataType::Categorical(_, _) | DataType::Enum(_, _)) => return Ok(None),
        #[cfg(feature = "dtype-decimal")]
        (DataType::Decimal(_, _), dt) if dt.is_primitive_numeric() => {
            IsInTypeCoercionResult::OtherCast(cast_type)
        },
        #[cfg(feature = "dtype-decimal")]
        (DataType::Decimal(_, _), _) | (_, DataType::Decimal(_, _)) => {
            polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} values in {:?} data", &type_other, &type_left)
        },
        // can't check for more granular time_unit in less-granular time_unit data,
        // or we'll cast away valid/necessary precision (eg: nanosecs to millisecs)
        (DataType::Datetime(lhs_unit, _), DataType::Datetime(rhs_unit, _)) => {
            if lhs_unit <= rhs_unit {
                return Ok(None);
            } else {
                polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} precision values in {:?} Datetime data", &rhs_unit, &lhs_unit)
            }
        },
        (DataType::Duration(lhs_unit), DataType::Duration(rhs_unit)) => {
            if lhs_unit <= rhs_unit {
                return Ok(None);
            } else {
                polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} precision values in {:?} Duration data", &rhs_unit, &lhs_unit)
            }
        },
        (_, DataType::List(other_inner)) => {
            if other_inner.as_ref() == &type_left
                || (type_left == DataType::Null)
                || (other_inner.as_ref() == &DataType::Null)
                || (other_inner.as_ref().is_primitive_numeric() && type_left.is_primitive_numeric())
            {
                return Ok(None);
            }
            polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} values in {:?} data", &type_left, &type_other)
        },
        #[cfg(feature = "dtype-array")]
        (_, DataType::Array(other_inner, _)) => {
            if other_inner.as_ref() == &type_left
                || (type_left == DataType::Null)
                || (other_inner.as_ref() == &DataType::Null)
                || (other_inner.as_ref().is_primitive_numeric() && type_left.is_primitive_numeric())
            {
                return Ok(None);
            }
            polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} values in {:?} data", &type_left, &type_other)
        },
        #[cfg(feature = "dtype-struct")]
        (DataType::Struct(_), _) | (_, DataType::Struct(_)) => return Ok(None),

        // don't attempt to cast between obviously mismatched types, but
        // allow integer/float comparison (will use their supertypes).
        (a, b) => {
            use DataType as T;
            if a != b
                && a.to_physical().is_primitive_numeric()
                && b.to_physical().is_primitive_numeric()
            {
                // @TAG: 2.0
                // @HACK: `is_in` does supertype casting between primitive numerics, which
                // honestly makes very little sense. To stay backwards compatible we keep this,
                // but please 2.0 remove this.

                // Since these two also also physically numerical but aren't allowed, we exclude
                // them here.
                let mut is_allowed_type = true;
                #[cfg(feature = "dtype-categorical")]
                {
                    is_allowed_type &= !matches!(a, T::Categorical(..) | T::Enum(..));
                }
                #[cfg(feature = "dtype-decimal")]
                {
                    is_allowed_type &= !matches!(a, T::Decimal(..));
                }

                if is_allowed_type {
                    let super_type =
                        polars_core::utils::try_get_supertype(&type_left, type_other_inner)?;
                    let other_type = match &type_other {
                        DataType::List(_) => DataType::List(Box::new(super_type.clone())),
                        #[cfg(feature = "dtype-array")]
                        DataType::Array(_, width) => {
                            DataType::Array(Box::new(super_type.clone()), *width)
                        },
                        _ => unreachable!(),
                    };

                    return Ok(Some(IsInTypeCoercionResult::SuperType(
                        super_type, other_type,
                    )));
                }
            }

            if (a.is_primitive_numeric() && b.is_primitive_numeric()) || (a == &DataType::Null) {
                return Ok(None);
            }
            polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} values in {:?} data", &type_other, &type_left)
        },
    };
    Ok(Some(casted_expr))
}
