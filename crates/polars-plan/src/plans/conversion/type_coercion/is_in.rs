use super::*;

#[derive(Debug)]
pub(super) enum IsInTypeCoercionResult {
    SuperType(DataType, DataType),
    SelfCast { dtype: DataType, strict: bool },
    OtherCast { dtype: DataType, strict: bool },
    Implode,
}

impl IsInTypeCoercionResult {
    fn map_self<F: Fn(DataType) -> DataType>(self, f: F) -> Self {
        use IsInTypeCoercionResult::*;
        match self {
            SuperType(dt1, dt2) => SuperType(f(dt1), dt2),
            SelfCast { dtype, strict } => SelfCast {
                dtype: f(dtype),
                strict,
            },
            x @ OtherCast { .. } => x,
            Implode => Implode,
        }
    }

    fn map_other<F: Fn(DataType) -> DataType>(self, f: F) -> Self {
        use IsInTypeCoercionResult::*;
        match self {
            SuperType(dt1, dt2) => SuperType(dt1, f(dt2)),
            x @ SelfCast { .. } => x,
            OtherCast { dtype, strict } => OtherCast {
                dtype: f(dtype),
                strict,
            },
            Implode => Implode,
        }
    }
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

    // dbg!(&type_left);
    // dbg!(&type_other);

    let wrap_other = |resolved_type_other: DataType| match &type_other {
        DataType::List(_) => DataType::List(Box::new(resolved_type_other)),
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, width) => DataType::Array(Box::new(resolved_type_other), *width),
        _ => unreachable!(),
    };

    let type_left_materialized = type_left.clone().materialize_unknown(false)?;
    let Some(type_other_inner) = type_other.inner_dtype() else {
        polars_bail!(InvalidOperation: "'{op:?}' cannot check for {type_left:?} values in {type_other:?} data.\n\
        Hint: container dtype ({type_other:?}) must be nested");
    };

    Ok((resolve_is_in_inner(
        &type_left_materialized,
        &type_other_inner,
        &type_left,
        &type_other,
        op,
    )?
    .map(|r| r.map_other(wrap_other))))
}

fn resolve_is_in_inner(
    type_left: &DataType,
    type_other_inner: &DataType,
    top_level_left_type: &DataType,
    top_level_nested_type: &DataType,
    op: &'static str,
) -> PolarsResult<Option<IsInTypeCoercionResult>> {
    // dbg!(type_left);
    // dbg!(type_other_inner);

    let casted_inner_expr = match (type_left, type_other_inner) {
        // Types are equal, do nothing
        (a, b) if a == b => return Ok(None),

        // All-null can represent anything (and/or empty list), so cast to target dtype
        (DataType::Null, _) => IsInTypeCoercionResult::SelfCast {
            dtype: type_other_inner.clone(),
            strict: false,
        },
        (_, DataType::Null) => IsInTypeCoercionResult::OtherCast {
            dtype: type_left.clone(),
            strict: false,
        },

        #[cfg(feature = "dtype-categorical")]
        (DataType::Enum(_, _), DataType::String) => IsInTypeCoercionResult::OtherCast {
            dtype: type_left.clone(),
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
            dtype: type_left.clone(),
            strict: false,
        },

        #[cfg(feature = "dtype-decimal")]
        (DataType::Decimal(_, _), dt) if dt.is_primitive_numeric() => {
            IsInTypeCoercionResult::OtherCast {
                dtype: type_left.clone(),
                strict: false,
            }
        },
        #[cfg(feature = "dtype-decimal")]
        (DataType::Decimal(_, _), _) | (_, DataType::Decimal(_, _)) => {
            polars_bail!(InvalidOperation: "'{op}' cannot check for {top_level_left_type:?} values in {top_level_nested_type:?} data")
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
        (a, b) => {
            if (a.is_primitive_numeric() && b.is_primitive_numeric()) || (a == &DataType::Null) {
                if let Some(super_type) =
                    get_numeric_upcast_supertype_lossless(&type_left, type_other_inner)
                {
                    return Ok(Some(IsInTypeCoercionResult::SuperType(
                        super_type.clone(),
                        super_type,
                    )));
                } else {
                    // We disabled lossless coercion of the operands in 2.0.
                    let lossy_supertype = try_get_supertype(&type_left, type_other_inner)?;
                    polars_bail!(InvalidOperation: "'{op}' cannot check for {top_level_left_type:?} values in {top_level_nested_type:?} data.\n\
                        Hint: Before version 2.0, Polars would perform this check by lossily coercing the operands to {lossy_supertype:?}. \
                        However, since Polars 2.0 the is_in() it is required to explicitly cast (one of) the operands to a compatible type.")
                }
            }
            polars_bail!(InvalidOperation: "'{op}' cannot check for {top_level_left_type:?} values in {top_level_nested_type:?} data")
        },
    };
    Ok(Some(casted_inner_expr))
}
