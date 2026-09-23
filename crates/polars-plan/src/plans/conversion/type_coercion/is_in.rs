use polars_core::utils::try_get_supertype;

use super::*;

#[derive(Debug)]
pub(super) enum IsInTypeCoercionResult {
    /// Cast the needle to `dtype`, which is exact for every value.
    SelfCast { dtype: DataType },
    /// Only for a container whose elements are all null, so the cast is always valid.
    OtherCast { dtype: DataType },
    #[cfg(feature = "is_in")]
    Implode,
    /// Cast the needle to `dtype` as it is evaluated; a needle the cast cannot represent exactly
    /// matches nothing.
    GuardedSelfCast { dtype: DataType },
}

/// Cast a numeric needle onto the element dtype, guarding the cast unless it always is exact.
fn numeric_needle_cast(needle: &DataType, element: &DataType) -> IsInTypeCoercionResult {
    if get_numeric_upcast_supertype_lossless(needle, element).as_ref() == Some(element) {
        IsInTypeCoercionResult::SelfCast {
            dtype: element.clone(),
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

    #[cfg(feature = "is_in")]
    pub(super) fn is_contains(self) -> bool {
        self == Self::Contains
    }
}

/// Which kind of container a membership function searches.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Container {
    /// A List or Array, searched by the `is_in` kernel.
    Sequence,
    /// A Map, searched by its keys.
    #[cfg(feature = "dtype-map")]
    Map,
}

/// Resolve Map lookup coercion without changing stored keys.
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
    let needle = needle.materialize_unknown(false)?;
    resolve_needle(&needle, key, Container::Map, op, &map)
}

/// Resolve the cast that makes a membership function's operands comparable.
#[cfg(feature = "is_in")]
pub(super) fn resolve_is_in(
    input: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
    form: MembershipForm,
    op: &'static str,
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

    let type_left_materialized = type_left.materialize_unknown(false)?;
    let Some(type_other_inner) = type_other.inner_dtype() else {
        polars_bail!(InvalidOperation: "'{op}' cannot check for {type_left_materialized:?} values in {type_other:?} data.\n\
        Hint: container dtype ({type_other:?}) must be nested{}", map_hint(form, &type_other, None));
    };

    resolve_needle(
        &type_left_materialized,
        type_other_inner,
        Container::Sequence,
        op,
        &type_other,
    )
    .map_err(
        |err| match map_hint(form, &type_other, Some(type_other_inner)) {
            "" => err,
            hint => err.wrap_msg(|msg| format!("{msg}{hint}")),
        },
    )
}

/// Choose the cast for a needle compared against elements of dtype `element`.
///
/// Only the needle is cast, so answering never depends on elements other than those compared.
/// Whether a pair is allowed depends on the dtypes alone.
fn resolve_needle(
    needle: &DataType,
    element: &DataType,
    container: Container,
    op: &'static str,
    container_dtype: &DataType,
) -> PolarsResult<Option<IsInTypeCoercionResult>> {
    use IsInTypeCoercionResult as R;

    let fail = |hint: &str| -> PolarsError {
        match container {
            Container::Sequence => polars_err!(
                InvalidOperation: "'{op}' cannot check for {needle:?} values in {container_dtype:?} data{hint}",
            ),
            #[cfg(feature = "dtype-map")]
            Container::Map => polars_err!(
                InvalidOperation: "'{op}' cannot look up a {needle:?} key in a Map with {element:?} keys{hint}",
            ),
        }
    };
    let self_cast = |dtype: &DataType| R::SelfCast {
        dtype: dtype.clone(),
    };

    let result = match (needle, element) {
        (n, e) if n == e => return Ok(None),

        // A null needle can take any element dtype; it is absent from a Map.
        (DataType::Null, e) => self_cast(e),
        // Elements that can only be null take the needle's dtype without looking at the data.
        (n, e) if container == Container::Sequence && is_null_shape_of(e, n) => R::OtherCast {
            dtype: with_inner(container_dtype, n.clone()),
        },

        (n, e) if (n.is_integer() && e.is_integer()) || (n.is_float() && e.is_float()) => {
            numeric_needle_cast(n, e)
        },
        (n, e) if n.is_primitive_numeric() && e.is_primitive_numeric() => {
            return Err(match container {
                Container::Sequence => {
                    // We disabled lossy coercion of the operands in 2.0.
                    let lossy_supertype = try_get_supertype(n, e)?;
                    fail(&format!(
                        ".\nHint: Before version 2.0, Polars would perform this check by lossily \
                        coercing the operands to {lossy_supertype:?}. However, since Polars 2.0, \
                        for '{op}' it is required to explicitly cast (one of) the operands to a \
                        compatible type."
                    ))
                },
                #[cfg(feature = "dtype-map")]
                Container::Map => fail(&format!("\nHint: cast the key to {e:?} first.")),
            });
        },

        // The kernel compares decimals of any precision and scale exactly.
        #[cfg(feature = "dtype-decimal")]
        (DataType::Decimal(_, _), DataType::Decimal(_, _)) => match container {
            Container::Sequence => return Ok(None),
            #[cfg(feature = "dtype-map")]
            Container::Map => R::GuardedSelfCast {
                dtype: element.clone(),
            },
        },

        (DataType::Datetime(needle_unit, needle_tz), DataType::Datetime(unit, tz)) => {
            if needle_tz.is_some() != tz.is_some() {
                return Err(fail(
                    ", as one is time-zone-aware and the other is not\n\
                    Hint: use `dt.replace_time_zone` to give both sides a time zone, or neither.",
                ));
            }
            // Instants compare across zones. The kernel compares physical values, and a Map key
            // must match the stored dtype exactly; the cast keeps the instant either way.
            let dtype = match container {
                Container::Sequence if needle_unit == unit => return Ok(None),
                Container::Sequence => DataType::Datetime(*unit, needle_tz.clone()),
                #[cfg(feature = "dtype-map")]
                Container::Map => element.clone(),
            };
            if needle_unit == unit {
                self_cast(&dtype)
            } else {
                R::GuardedSelfCast { dtype }
            }
        },
        (DataType::Duration(_), DataType::Duration(_)) => R::GuardedSelfCast {
            dtype: element.clone(),
        },

        // The kernel looks strings up by category, so an unknown label is absent.
        #[cfg(feature = "dtype-categorical")]
        (DataType::String, DataType::Enum(_, _) | DataType::Categorical(_, _))
        | (DataType::Enum(_, _) | DataType::Categorical(_, _), DataType::String) => match container
        {
            Container::Sequence => return Ok(None),
            #[cfg(feature = "dtype-map")]
            Container::Map if element.is_string() => self_cast(element),
            // An unknown label becomes a null key, which no Map holds.
            #[cfg(feature = "dtype-map")]
            Container::Map => R::GuardedSelfCast {
                dtype: element.clone(),
            },
        },

        _ => return Err(fail("")),
    };
    Ok(Some(result))
}

/// Whether `element` can only hold nulls where `needle` holds values, e.g. `List(Null)` for a
/// `List(Int64)` needle.
fn is_null_shape_of(element: &DataType, needle: &DataType) -> bool {
    match (element, needle) {
        (DataType::Null, _) => true,
        (DataType::List(e), DataType::List(n)) => is_null_shape_of(e, n),
        #[cfg(feature = "dtype-array")]
        (DataType::Array(e, ew), DataType::Array(n, nw)) => ew == nw && is_null_shape_of(e, n),
        #[cfg(feature = "dtype-struct")]
        (DataType::Struct(ef), DataType::Struct(nf)) => {
            ef.len() == nf.len()
                && ef
                    .iter()
                    .zip(nf)
                    .all(|(e, n)| e.name() == n.name() && is_null_shape_of(e.dtype(), n.dtype()))
        },
        _ => false,
    }
}

/// `container` with its elements replaced by `inner`.
fn with_inner(container: &DataType, inner: DataType) -> DataType {
    match container {
        DataType::List(_) => DataType::List(Box::new(inner)),
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, width) => DataType::Array(Box::new(inner), *width),
        _ => unreachable!("a List or Array container"),
    }
}

/// Point an `is_in` on a Map at the Map namespace.
#[cfg(feature = "is_in")]
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
