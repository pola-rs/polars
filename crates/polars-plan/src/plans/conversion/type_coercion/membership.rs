use polars_core::utils::try_get_supertype;

use super::*;

#[derive(Debug)]
pub(super) enum IsInTypeCoercionResult {
    /// Cast the needle to `dtype`, which is exact for every value.
    CastNeedle { dtype: DataType },
    /// Cast a container whose element dtype has only Null leaves.
    CastContainer { dtype: DataType },
    #[cfg(feature = "is_in")]
    Implode,
    /// Cast the needle to `dtype`; inexact values match nothing.
    GuardedNeedleCast { dtype: DataType },
}

/// Cast a numeric needle onto the element dtype, guarding the cast unless it always is exact.
fn numeric_needle_cast(needle: &DataType, element: &DataType) -> IsInTypeCoercionResult {
    if get_numeric_upcast_supertype_lossless(needle, element).as_ref() == Some(element) {
        IsInTypeCoercionResult::CastNeedle {
            dtype: element.clone(),
        }
    } else {
        IsInTypeCoercionResult::GuardedNeedleCast {
            dtype: element.clone(),
        }
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

/// Resolve membership compatibility and coercion from operand dtypes alone.
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
    let cast_needle = |dtype: &DataType| R::CastNeedle {
        dtype: dtype.clone(),
    };
    let guarded = |dtype: &DataType| R::GuardedNeedleCast {
        dtype: dtype.clone(),
    };

    let result = match (needle, element) {
        (n, e) if n == e => return Ok(None),

        // A null needle can take any element dtype; it is absent from a Map.
        (DataType::Null, e) => cast_needle(e),
        // Elements that can only be null take the needle's dtype without looking at the data.
        (n, e) if container == Container::Sequence && is_null_shape_of(e, n) => R::CastContainer {
            dtype: with_inner(container_dtype, n.clone()),
        },
        // Map keys are never null, but a nested key can hold only null leaves. The key equality
        // kernel compares those with the needle's values as they are.
        #[cfg(feature = "dtype-map")]
        (n, e) if container == Container::Map && !e.is_null() && is_null_shape_of(e, n) => {
            return Ok(None);
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
            Container::Map => guarded(element),
        },
        #[cfg(feature = "dtype-decimal")]
        (n, DataType::Decimal(_, _)) if n.is_integer() => match container {
            // Integers are exact decimals at scale 0; only 128-bit values can fail the cast.
            Container::Sequence => guarded(&DataType::Decimal(
                polars_compute::decimal::DEC128_MAX_PREC,
                0,
            )),
            #[cfg(feature = "dtype-map")]
            Container::Map => guarded(element),
        },
        // A decimal with a fractional part, or out of the integer's range, matches nothing.
        #[cfg(feature = "dtype-decimal")]
        (DataType::Decimal(_, _), e) if e.is_integer() => guarded(element),

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
                cast_needle(&dtype)
            } else {
                guarded(&dtype)
            }
        },
        (DataType::Duration(_), DataType::Duration(_)) => guarded(element),

        // The kernel looks strings up by category, so an unknown label is absent.
        #[cfg(feature = "dtype-categorical")]
        (DataType::String, DataType::Enum(_, _) | DataType::Categorical(_, _))
        | (DataType::Enum(_, _) | DataType::Categorical(_, _), DataType::String) => match container
        {
            Container::Sequence => return Ok(None),
            #[cfg(feature = "dtype-map")]
            Container::Map if element.is_string() => cast_needle(element),
            // An unknown label becomes a null key, which no Map holds.
            #[cfg(feature = "dtype-map")]
            Container::Map => guarded(element),
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

/// Apply membership or Map lookup casts chosen by `resolve`.
///
/// `expr_node` must be an [`AExpr::Function`] whose operands are laid out as `form` says.
#[cfg(any(feature = "is_in", feature = "dtype-map"))]
pub(super) fn coerce_is_in(
    expr_node: Node,
    expr_arena: &mut Arena<AExpr>,
    schema: &Schema,
    form: MembershipForm,
    resolve: impl FnOnce(&[ExprIR], &Arena<AExpr>) -> PolarsResult<Option<IsInTypeCoercionResult>>,
) -> PolarsResult<Option<AExpr>> {
    let AExpr::Function {
        function,
        input,
        options,
    } = expr_arena.get(expr_node)
    else {
        unreachable!("caller matched a function expression")
    };
    let options = *options;
    let (flat, nested) = (form.flat(), form.nested());

    // The needle is only cast when the function runs, so its dtype still differs.
    if function.membership_needle_cast().is_some() {
        return Ok(None);
    }

    let result = resolve(input, expr_arena)?;

    #[cfg(feature = "is_in")]
    if let Some(haystack) =
        cast_literal_haystack(function, input, form, result.as_ref(), expr_arena, schema)?
    {
        let (function, mut input) = (function.clone(), input.to_vec());
        input[nested].set_node(expr_arena.add(AExpr::Literal(haystack.into())));
        return Ok(Some(AExpr::Function {
            function,
            input,
            options,
        }));
    }

    let Some(result) = result else {
        return Ok(None);
    };

    let mut function = function.clone();
    let mut input = input.to_vec();
    use IsInTypeCoercionResult as R;
    let (operand, dtype) = match result {
        R::CastNeedle { dtype } => (flat, dtype),
        R::CastContainer { dtype } => (nested, dtype),
        R::GuardedNeedleCast { dtype } => {
            // A literal needle that casts exactly is substituted now. Any other needle is cast
            // and checked as the function runs, so that it is evaluated once.
            match exact_literal_needle(&input[flat], &dtype, expr_arena, schema)? {
                Some(needle) => {
                    input[flat].set_node(expr_arena.add(AExpr::Literal(needle.into())));
                    input[flat].set_dtype(dtype);
                },
                None => *function.membership_needle_cast_mut().unwrap() = Some(dtype),
            }
            return Ok(Some(AExpr::Function {
                function,
                input,
                options,
            }));
        },
        #[cfg(feature = "is_in")]
        R::Implode => {
            assert!(!form.is_contains());
            let other_input = expr_arena.add(AExpr::Agg(IRAggExpr::Implode {
                input: input[nested].node(),
                maintain_order: true,
            }));
            input[nested].set_node(other_input);
            return Ok(Some(AExpr::Function {
                function,
                input,
                options,
            }));
        },
    };
    let (_, from) = unpack!(get_aexpr_and_type(
        expr_arena,
        input[operand].node(),
        schema
    ));
    cast_expr_ir(
        &mut input[operand],
        &from,
        &dtype,
        expr_arena,
        CastOptions::NonStrict,
    )?;

    Ok(Some(AExpr::Function {
        function,
        input,
        options,
    }))
}

/// A scalar literal needle cast to `dtype`, if the cast is exact.
fn exact_literal_needle(
    needle: &ExprIR,
    dtype: &DataType,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<Option<Scalar>> {
    let AExpr::Literal(lv) = expr_arena.get(needle.node()) else {
        return Ok(None);
    };
    let scalar = match lv {
        LiteralValue::Dyn(dyn_value) => {
            let from = needle
                .dtype(schema, expr_arena)?
                .clone()
                .materialize_unknown(false)?;
            dyn_value
                .clone()
                .try_materialize_to_dtype(&from, CastOptions::Strict)?
        },
        LiteralValue::Scalar(scalar) => scalar.clone(),
        _ => return Ok(None),
    };
    let (casted, inexact) = scalar
        .into_series(PlSmallStr::EMPTY)
        ._cast_reporting_inexact(dtype)?;
    if inexact.is_some() {
        return Ok(None);
    }
    Ok(Some(Scalar::new(
        dtype.clone(),
        casted.get(0)?.into_static(),
    )))
}

/// Cast eligible literal haystacks to the needle dtype to preserve pushdown.
/// Inexact elements cannot match: drop them from Lists, or skip the rewrite
/// for Arrays.
#[cfg(feature = "is_in")]
fn cast_literal_haystack(
    function: &IRFunctionExpr,
    input: &[ExprIR],
    form: MembershipForm,
    result: Option<&IsInTypeCoercionResult>,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<Option<Scalar>> {
    use IsInTypeCoercionResult as R;

    let (flat, nested) = (form.flat(), form.nested());
    if is_map_lookup(function)
        || !matches!(
            result,
            None | Some(R::CastNeedle { .. } | R::GuardedNeedleCast { .. })
        )
        || matches!(expr_arena.get(input[flat].node()), AExpr::Literal(_))
    {
        return Ok(None);
    }
    let AExpr::Literal(LiteralValue::Scalar(haystack)) = expr_arena.get(input[nested].node())
    else {
        return Ok(None);
    };
    let needle = input[flat].dtype(schema, expr_arena)?;
    // Casting strings to a Categorical would add them to its categories.
    if !needle.is_known() || needle.contains_categoricals() {
        return Ok(None);
    }
    let (elements, width) = match haystack.value() {
        AnyValue::List(s) => (s, None::<usize>),
        #[cfg(feature = "dtype-array")]
        AnyValue::Array(s, width) => (s, Some(*width)),
        _ => return Ok(None),
    };
    if elements.dtype() == needle {
        return Ok(None);
    }
    let Ok((casted, inexact)) = elements._cast_reporting_inexact(needle) else {
        return Ok(None);
    };
    // Some casts keep the nesting, e.g. a Struct to `Null` gives a Struct of nulls. A haystack
    // of another dtype would be rewritten again on every pass.
    if casted.dtype() != needle {
        return Ok(None);
    }
    let casted = match (inexact, width) {
        (None, _) => casted,
        (Some(inexact), None) => casted.filter(&!&inexact)?,
        // Dropping elements would change an Array's width.
        (Some(_), Some(_)) => return Ok(None),
    };
    let value = match width {
        None => AnyValue::List(casted),
        #[cfg(feature = "dtype-array")]
        Some(width) => AnyValue::Array(casted, width),
        #[cfg(not(feature = "dtype-array"))]
        Some(_) => unreachable!(),
    };

    Ok(Some(Scalar::new(
        with_inner(haystack.dtype(), needle.clone()),
        value.into_static(),
    )))
}
