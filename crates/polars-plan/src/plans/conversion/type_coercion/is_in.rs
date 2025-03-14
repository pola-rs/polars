use super::*;

pub(super) fn resolve_is_in(
    input: &[ExprIR],
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &Arena<IR>,
    lp_node: Node,
) -> PolarsResult<Option<AExpr>> {
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

    // The level of r should be one level deeper.
    // as we check if col(a) is in a container.
    //
    // So if we check if `Series("A": [1, 2, 3])` `is_in` `[1, 2]`
    //
    // This is typed as `List<T>` where B is a scalar and can be broadcasted
    //
    // A    B
    // 1    [[1, 2]]
    // 2
    // 3
    //
    // The valid states are a scalar list (above), or a list of equal elements as A
    //
    // A    B
    // 1    [[1, 2]]
    // 2    [[1]]
    // 3    [[5, 3]]

    let ae_builder = AExprBuilder::new(other_e.node(), expr_arena);
    use DataType as D;
    let ae_builder = match (&type_left, &type_other) {
        (_, D::List(other_inner)) => {
            if other_inner.as_ref() == &type_left
                || (type_left == D::Null)
                || (other_inner.as_ref() == &D::Null)
                || (other_inner.as_ref().is_primitive_numeric() && type_left.is_primitive_numeric())
            {
                return Ok(None);
            }
            polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} values in {:?} data", &type_left, &type_other)
        },
        // types are equal, cast b
        //(a, b) if a == b => other_e.clone(),
        // all-null can represent anything (and/or empty list), so cast to target dtype
        (_, D::Null) => ae_builder.cast(type_left, CastOptions::NonStrict),
        (D::Decimal(_, _), dt) if dt.is_primitive_numeric() => {
            ae_builder.cast(type_left, CastOptions::NonStrict)
        },
        #[cfg(feature = "dtype-categorical")]
        (D::Categorical(_, _) | D::Enum(_, _), D::String) => ae_builder,
        #[cfg(feature = "dtype-categorical")]
        (D::String, D::Categorical(_, _) | D::Enum(_, _)) => ae_builder,
        #[cfg(feature = "dtype-decimal")]
        (D::Decimal(_, _), _) | (_, D::Decimal(_, _)) => {
            polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} values in {:?} data", &type_other, &type_left)
        },
        // can't check for more granular time_unit in less-granular time_unit data,
        // or we'll cast away valid/necessary precision (eg: nanosecs to millisecs)
        (D::Datetime(lhs_unit, _), D::Datetime(rhs_unit, _)) => {
            if lhs_unit <= rhs_unit {
                ae_builder
            } else {
                polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} precision values in {:?} Datetime data", &rhs_unit, &lhs_unit)
            }
        },
        (D::Duration(lhs_unit), D::Duration(rhs_unit)) => {
            if lhs_unit <= rhs_unit {
                ae_builder
            } else {
                polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} precision values in {:?} Duration data", &rhs_unit, &lhs_unit)
            }
        },
        #[cfg(feature = "dtype-array")]
        (_, D::Array(other_inner, _)) => {
            if other_inner.as_ref() == &type_left
                || (type_left == D::Null)
                || (other_inner.as_ref() == &D::Null)
                || (other_inner.as_ref().is_primitive_numeric() && type_left.is_primitive_numeric())
            {
                ae_builder
            } else {
                polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} values in {:?} data", &type_left, &type_other)
            }
        },
        #[cfg(feature = "dtype-struct")]
        (D::Struct(_), _) | (_, D::Struct(_)) => return Ok(None),

        // don't attempt to cast between obviously mismatched types, but
        // allow integer/float comparison (will use their supertypes).
        (a, b) => {
            if (a.is_primitive_numeric() && b.is_primitive_numeric()) || (a == &D::Null) {
                ae_builder
            } else {
                polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} values in {:?} data", &type_other, &type_left)
            }
        },
    };

    Ok(Some(ae_builder.implode().build_ae()))
}
