use polars_core::utils::try_get_supertype;

use super::*;

#[derive(Debug)]
pub(super) enum IsInTypeCoercionResult {
    SuperType(DataType, DataType),
    SelfCast {
        dtype: DataType,
        strict: bool,
    },
    OtherCast {
        dtype: DataType,
        strict: bool,
    },
    Implode,
    /// Cast the needle to `dtype` as it is evaluated; a needle the cast cannot represent exactly
    /// matches nothing.
    GuardedSelfCast {
        dtype: DataType,
    },
}

/// Cast the needle onto the element dtype, guarding the cast unless it is always exact.
fn needle_to_element(needle: &DataType, element: &DataType) -> IsInTypeCoercionResult {
    if needle.is_primitive_numeric()
        && get_numeric_upcast_supertype_lossless(needle, element).as_ref() == Some(element)
    {
        IsInTypeCoercionResult::SelfCast {
            dtype: element.clone(),
            strict: false,
        }
    } else {
        IsInTypeCoercionResult::GuardedSelfCast {
            dtype: element.clone(),
        }
    }
}

/// The needle dtype a membership function casts to as it runs, if coercion chose one.
pub(super) fn needle_cast(function: &IRFunctionExpr) -> Option<&DataType> {
    match function {
        #[cfg(feature = "is_in")]
        IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { needle_cast, .. })
        | IRFunctionExpr::ListExpr(IRListFunction::Contains { needle_cast, .. }) => {
            needle_cast.as_ref()
        },
        #[cfg(all(feature = "is_in", feature = "dtype-array"))]
        IRFunctionExpr::ArrayExpr(IRArrayFunction::Contains { needle_cast, .. }) => {
            needle_cast.as_ref()
        },
        #[cfg(feature = "dtype-map")]
        IRFunctionExpr::MapExpr(
            IRMapFunction::Get { needle_cast } | IRMapFunction::ContainsKey { needle_cast },
        ) => needle_cast.as_ref(),
        _ => None,
    }
}

pub(super) fn needle_cast_mut(function: &mut IRFunctionExpr) -> &mut Option<DataType> {
    match function {
        #[cfg(feature = "is_in")]
        IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { needle_cast, .. })
        | IRFunctionExpr::ListExpr(IRListFunction::Contains { needle_cast, .. }) => needle_cast,
        #[cfg(all(feature = "is_in", feature = "dtype-array"))]
        IRFunctionExpr::ArrayExpr(IRArrayFunction::Contains { needle_cast, .. }) => needle_cast,
        #[cfg(feature = "dtype-map")]
        IRFunctionExpr::MapExpr(
            IRMapFunction::Get { needle_cast } | IRMapFunction::ContainsKey { needle_cast },
        ) => needle_cast,
        _ => unreachable!("not a membership function"),
    }
}

pub(super) fn is_map_lookup(function: &IRFunctionExpr) -> bool {
    #[cfg(feature = "dtype-map")]
    if matches!(function, IRFunctionExpr::MapExpr(_)) {
        return true;
    }
    let _ = function;
    false
}

/// Where a membership function keeps its operands.
///
/// The flat operand is compared against the elements of the nested one; which input is which
/// depends on the function.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum MembershipForm {
    /// `flat.is_in(nested)`.
    IsIn,
    /// `nested.contains(flat)`, including Map key lookup.
    Contains,
}

impl MembershipForm {
    /// Input index of the operand compared against the container's elements.
    pub(super) fn flat(self) -> usize {
        (self == Self::Contains) as usize
    }

    /// Input index of the container.
    pub(super) fn nested(self) -> usize {
        (self == Self::IsIn) as usize
    }

    pub(super) fn is_contains(self) -> bool {
        self == Self::Contains
    }
}

/// Resolve Map lookup coercion without changing stored keys.
///
/// Cast only the needle. A cast that is exact, or that yields null for a key no Map can hold,
/// is fine. A cast that could round the needle onto a different key is not.
#[cfg(feature = "dtype-map")]
pub(super) fn resolve_map_key(
    input: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
    op: &'static str,
) -> PolarsResult<Option<IsInTypeCoercionResult>> {
    let (_, needle) = unpack!(get_aexpr_and_type(
        expr_arena,
        input[1].node(),
        input_schema
    ));
    let (_, map) = unpack!(get_aexpr_and_type(
        expr_arena,
        input[0].node(),
        input_schema
    ));
    let DataType::Map(key, _) = &map else {
        // Output field resolution rejects non-Map inputs.
        return Ok(None);
    };

    if let Some(result) = resolve_temporal_map_key(&needle, key, op)? {
        return Ok(Some(result));
    }

    Ok(Some(
        match resolve_is_in(
            input,
            expr_arena,
            input_schema,
            MembershipForm::Contains,
            op,
            Some(key.as_ref()),
        )? {
            None => return Ok(None),
            // The needle alone moves, so a strict cast still raises as it would for `is_in`.
            Some(
                result @ (IsInTypeCoercionResult::SelfCast { .. }
                | IsInTypeCoercionResult::GuardedSelfCast { .. }),
            ) => result,
            // Only the needle has to widen to reach the stored key type.
            Some(IsInTypeCoercionResult::SuperType(supertype, _)) if supertype == **key => {
                IsInTypeCoercionResult::SelfCast {
                    dtype: supertype,
                    strict: false,
                }
            },
            Some(
                IsInTypeCoercionResult::SuperType(_, _) | IsInTypeCoercionResult::OtherCast { .. },
            ) => {
                polars_bail!(
                InvalidOperation:
                "'{op}' cannot look up a `{needle}` key in a Map with `{key}` keys\n\
                Hint: cast the key to `{key}` first.",
                )
            },
            Some(IsInTypeCoercionResult::Implode) => {
                unreachable!("a key lookup resolves as a `contains`")
            },
        },
    ))
}

/// Cast the needle to the Map's temporal unit, guarding against overflow and rounding.
#[cfg(feature = "dtype-map")]
fn resolve_temporal_map_key(
    needle: &DataType,
    key: &DataType,
    op: &'static str,
) -> PolarsResult<Option<IsInTypeCoercionResult>> {
    let (needle_unit, key_unit, target) = match (needle, key) {
        (DataType::Datetime(needle_unit, needle_tz), DataType::Datetime(key_unit, key_tz)) => {
            // Comparing across zones is a `SchemaMismatch` in the kernel. Name it here instead.
            polars_ensure!(
                needle_tz == key_tz,
                InvalidOperation:
                "'{op}' cannot look up a `{needle}` key in a Map with `{key}` keys, as the time \
                zones differ\nHint: convert the key to `{key}` first.",
            );
            (
                needle_unit,
                key_unit,
                DataType::Datetime(*key_unit, key_tz.clone()),
            )
        },
        (DataType::Duration(needle_unit), DataType::Duration(key_unit)) => {
            (needle_unit, key_unit, DataType::Duration(*key_unit))
        },
        _ => return Ok(None),
    };
    if needle_unit == key_unit {
        return Ok(None);
    }
    Ok(Some(IsInTypeCoercionResult::GuardedSelfCast {
        dtype: target,
    }))
}

/// Resolve the cast that makes a membership function's operands comparable.
///
/// `element_dtype` names the container's elements for containers that have no
/// [`DataType::inner_dtype`], such as a Map searched by its keys.
pub(super) fn resolve_is_in(
    input: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
    form: MembershipForm,
    op: &'static str,
    element_dtype: Option<&DataType>,
) -> PolarsResult<Option<IsInTypeCoercionResult>> {
    let (_, type_left) = unpack!(get_aexpr_and_type(
        expr_arena,
        input[form.flat()].node(),
        input_schema
    ));
    let (_, type_other) = unpack!(get_aexpr_and_type(
        expr_arena,
        input[form.nested()].node(),
        input_schema
    ));

    let left_nl = type_left.nesting_level();
    let right_nl = type_other.nesting_level();

    // @HACK. This needs to happen until 3.0 because we support `pl.col.a.is_in(pl.col.a)`.
    if !form.is_contains() && left_nl == right_nl {
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
        // A Map lookup reaches this through the lossless upcast of its key.
        #[cfg(feature = "dtype-map")]
        DataType::Map(_, value) => DataType::Map(Box::new(resolved_inner_type), value.clone()),
        _ => unreachable!(),
    };

    let type_left_materialized = type_left.clone().materialize_unknown(false)?;
    let Some(type_other_inner) = element_dtype.or_else(|| type_other.inner_dtype()) else {
        polars_bail!(InvalidOperation: "'{op}' cannot check for {type_left_materialized:?} values in {type_other:?} data.\n\
        Hint: container dtype ({type_other:?}) must be nested{}", map_hint(form, &type_other, None));
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

        // An integer cast is exact or out of range, and the guard makes out of range a miss.
        (dtml, dto) if dtml.is_integer() && dto.is_integer() => needle_to_element(dtml, dto),

        #[cfg(feature = "dtype-decimal")]
        (DataType::Decimal(_, _), _) | (_, DataType::Decimal(_, _)) => {
            polars_bail!(InvalidOperation: "'{op}' cannot check for {type_left_materialized:?} values in {type_other:?} data")
        },
        // A unit cast can overflow or round, so it is guarded.
        (DataType::Datetime(needle_unit, needle_tz), DataType::Datetime(other_unit, _)) => {
            // Equal units but unequal dtypes means the time zones differ; the kernel compares
            // the physical values, so instants match across zones.
            if needle_unit == other_unit {
                return Ok(None);
            }
            IsInTypeCoercionResult::GuardedSelfCast {
                dtype: DataType::Datetime(*other_unit, needle_tz.clone()),
            }
        },
        (DataType::Duration(_), DataType::Duration(other_unit)) => {
            IsInTypeCoercionResult::GuardedSelfCast {
                dtype: DataType::Duration(*other_unit),
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
                    polars_bail!(InvalidOperation: "'{op}' cannot check for {type_left_materialized:?} values in {type_other:?} data.\n\
                        Hint: Before version 2.0, Polars would perform this check by lossily coercing the operands to {lossy_supertype:?}. \
                        However, since Polars 2.0, for '{op}' it is required to explicitly cast (one of) the operands to a compatible type.")
                }
            }
            polars_bail!(
                InvalidOperation: "'{op}' cannot check for {type_left_materialized:?} values in {type_other:?} data{}",
                map_hint(form, &type_other, Some(dto)),
            )
        },
    };
    Ok(Some(casted_inner_expr))
}

/// Point an `is_in` on a Map at the Map namespace.
fn map_hint(
    form: MembershipForm,
    container: &DataType,
    element: Option<&DataType>,
) -> &'static str {
    #[cfg(feature = "dtype-map")]
    if !form.is_contains()
        && (matches!(container, DataType::Map(..)) || matches!(element, Some(DataType::Map(..))))
    {
        return "\nHint: use `map.contains_key` to search a Map by its keys.";
    }
    let _ = (form, container, element);
    ""
}
