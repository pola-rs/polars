mod binary;

use std::borrow::Cow;

use arrow::temporal_conversions::{time_unit_multiple, SECONDS_IN_DAY};
use binary::process_binary;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_core::utils::{get_supertype, get_supertype_with_options, materialize_dyn_int};
use polars_utils::idx_vec::UnitVec;
use polars_utils::itertools::Itertools;
use polars_utils::{format_list, unitvec};

use super::*;

pub struct TypeCoercionRule {}

macro_rules! unpack {
    ($packed:expr) => {
        match $packed {
            Some(payload) => payload,
            None => return Ok(None),
        }
    };
}

/// determine if we use the supertype or not. For instance when we have a column Int64 and we compare with literal UInt32
/// it would be wasteful to cast the column instead of the literal.
fn modify_supertype(
    mut st: DataType,
    left: &AExpr,
    right: &AExpr,
    type_left: &DataType,
    type_right: &DataType,
) -> DataType {
    // TODO! This must be removed and dealt properly with dynamic str.
    use DataType::*;
    match (type_left, type_right, left, right) {
        // if the we compare a categorical to a literal string we want to cast the literal to categorical
        #[cfg(feature = "dtype-categorical")]
        (Categorical(_, ordering), String | Unknown(UnknownKind::Str), _, AExpr::Literal(_))
        | (String | Unknown(UnknownKind::Str), Categorical(_, ordering), AExpr::Literal(_), _) => {
            st = Categorical(None, *ordering)
        },
        #[cfg(feature = "dtype-categorical")]
        (dt @ Enum(_, _), String | Unknown(UnknownKind::Str), _, AExpr::Literal(_))
        | (String | Unknown(UnknownKind::Str), dt @ Enum(_, _), AExpr::Literal(_), _) => {
            st = dt.clone()
        },
        // when then expression literals can have a different list type.
        // so we cast the literal to the other hand side.
        (List(inner), List(other), _, AExpr::Literal(_))
        | (List(other), List(inner), AExpr::Literal(_), _)
            if inner != other =>
        {
            st = match &**inner {
                #[cfg(feature = "dtype-categorical")]
                Categorical(_, ordering) => List(Box::new(Categorical(None, *ordering))),
                _ => List(inner.clone()),
            };
        },
        // do nothing
        _ => {},
    }
    st
}

fn get_input(lp_arena: &Arena<IR>, lp_node: Node) -> UnitVec<Node> {
    let plan = lp_arena.get(lp_node);
    let mut inputs: UnitVec<Node> = unitvec!();

    // Used to get the schema of the input.
    if is_scan(plan) {
        inputs.push(lp_node);
    } else {
        plan.copy_inputs(&mut inputs);
    };
    inputs
}

fn get_schema(lp_arena: &Arena<IR>, lp_node: Node) -> Cow<'_, SchemaRef> {
    let inputs = get_input(lp_arena, lp_node);
    if inputs.is_empty() {
        // Files don't have an input, so we must take their schema.
        Cow::Borrowed(lp_arena.get(lp_node).scan_schema())
    } else {
        let input = inputs[0];
        lp_arena.get(input).schema(lp_arena)
    }
}

fn get_aexpr_and_type<'a>(
    expr_arena: &'a Arena<AExpr>,
    e: Node,
    input_schema: &Schema,
) -> Option<(&'a AExpr, DataType)> {
    let ae = expr_arena.get(e);
    Some((
        ae,
        ae.get_type(input_schema, Context::Default, expr_arena)
            .ok()?,
    ))
}

fn materialize(aexpr: &AExpr) -> Option<AExpr> {
    match aexpr {
        AExpr::Literal(lv) => Some(AExpr::Literal(lv.clone().materialize())),
        _ => None,
    }
}

impl OptimizationRule for TypeCoercionRule {
    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        lp_arena: &Arena<IR>,
        lp_node: Node,
    ) -> PolarsResult<Option<AExpr>> {
        let expr = expr_arena.get(expr_node);
        let out = match *expr {
            AExpr::Cast {
                expr,
                ref data_type,
                options,
            } => {
                let input = expr_arena.get(expr);

                inline_or_prune_cast(
                    input,
                    data_type,
                    options.strict(),
                    lp_node,
                    lp_arena,
                    expr_arena,
                )?
            },
            AExpr::Ternary {
                truthy: truthy_node,
                falsy: falsy_node,
                predicate,
            } => {
                let input_schema = get_schema(lp_arena, lp_node);
                let (truthy, type_true) =
                    unpack!(get_aexpr_and_type(expr_arena, truthy_node, &input_schema));
                let (falsy, type_false) =
                    unpack!(get_aexpr_and_type(expr_arena, falsy_node, &input_schema));

                if type_true == type_false {
                    return Ok(None);
                }
                let st = unpack!(get_supertype(&type_true, &type_false));
                let st = modify_supertype(st, truthy, falsy, &type_true, &type_false);

                // only cast if the type is not already the super type.
                // this can prevent an expensive flattening and subsequent aggregation
                // in a group_by context. To be able to cast the groups need to be
                // flattened
                let new_node_truthy = if type_true != st {
                    expr_arena.add(AExpr::Cast {
                        expr: truthy_node,
                        data_type: st.clone(),
                        options: CastOptions::Strict,
                    })
                } else {
                    truthy_node
                };

                let new_node_falsy = if type_false != st {
                    expr_arena.add(AExpr::Cast {
                        expr: falsy_node,
                        data_type: st,
                        options: CastOptions::Strict,
                    })
                } else {
                    falsy_node
                };

                Some(AExpr::Ternary {
                    truthy: new_node_truthy,
                    falsy: new_node_falsy,
                    predicate,
                })
            },
            AExpr::BinaryExpr {
                left: node_left,
                op,
                right: node_right,
            } => return process_binary(expr_arena, lp_arena, lp_node, node_left, op, node_right),
            #[cfg(feature = "is_in")]
            AExpr::Function {
                function: FunctionExpr::Boolean(BooleanFunction::IsIn),
                ref input,
                options,
            } => {
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

                unpack!(early_escape(&type_left, &type_other));

                let casted_expr = match (&type_left, &type_other) {
                    // types are equal, do nothing
                    (a, b) if a == b => return Ok(None),
                    // all-null can represent anything (and/or empty list), so cast to target dtype
                    (_, DataType::Null) => AExpr::Cast {
                        expr: other_e.node(),
                        data_type: type_left,
                        options: CastOptions::NonStrict,
                    },
                    #[cfg(feature = "dtype-categorical")]
                    (DataType::Categorical(_, _) | DataType::Enum(_, _), DataType::String) => {
                        return Ok(None)
                    },
                    #[cfg(feature = "dtype-categorical")]
                    (DataType::String, DataType::Categorical(_, _) | DataType::Enum(_, _)) => {
                        return Ok(None)
                    },
                    #[cfg(feature = "dtype-decimal")]
                    (DataType::Decimal(_, _), dt) if dt.is_numeric() => AExpr::Cast {
                        expr: other_e.node(),
                        data_type: type_left,
                        options: CastOptions::NonStrict,
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
                            || (other_inner.as_ref().is_numeric() && type_left.is_numeric())
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
                            || (other_inner.as_ref().is_numeric() && type_left.is_numeric())
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
                        if (a.is_numeric() && b.is_numeric()) || (a == &DataType::Null) {
                            return Ok(None);
                        }
                        polars_bail!(InvalidOperation: "'is_in' cannot check for {:?} values in {:?} data", &type_other, &type_left)
                    },
                };
                let mut input = input.clone();
                let other_input = expr_arena.add(casted_expr);
                input[1].set_node(other_input);

                Some(AExpr::Function {
                    function: FunctionExpr::Boolean(BooleanFunction::IsIn),
                    input,
                    options,
                })
            },
            // shift and fill should only cast left and fill value to super type.
            AExpr::Function {
                function: FunctionExpr::ShiftAndFill,
                ref input,
                options,
            } => {
                let mut input = input.clone();

                let input_schema = get_schema(lp_arena, lp_node);
                let left_node = input[0].node();
                let fill_value_node = input[2].node();
                let (left, type_left) =
                    unpack!(get_aexpr_and_type(expr_arena, left_node, &input_schema));
                let (fill_value, type_fill_value) = unpack!(get_aexpr_and_type(
                    expr_arena,
                    fill_value_node,
                    &input_schema
                ));

                unpack!(early_escape(&type_left, &type_fill_value));

                let super_type = unpack!(get_supertype(&type_left, &type_fill_value));
                let super_type =
                    modify_supertype(super_type, left, fill_value, &type_left, &type_fill_value);

                let new_node_left = if type_left != super_type {
                    expr_arena.add(AExpr::Cast {
                        expr: left_node,
                        data_type: super_type.clone(),
                        options: CastOptions::NonStrict,
                    })
                } else {
                    left_node
                };

                let new_node_fill_value = if type_fill_value != super_type {
                    expr_arena.add(AExpr::Cast {
                        expr: fill_value_node,
                        data_type: super_type.clone(),
                        options: CastOptions::NonStrict,
                    })
                } else {
                    fill_value_node
                };

                input[0].set_node(new_node_left);
                input[2].set_node(new_node_fill_value);

                Some(AExpr::Function {
                    function: FunctionExpr::ShiftAndFill,
                    input,
                    options,
                })
            },
            // generic type coercion of any function.
            AExpr::Function {
                // only for `DataType::Unknown` as it still has to be set.
                ref function,
                ref input,
                mut options,
            } if options.cast_to_supertypes.is_some() => {
                let input_schema = get_schema(lp_arena, lp_node);
                let mut dtypes = Vec::with_capacity(input.len());
                for e in input {
                    let (_, dtype) =
                        unpack!(get_aexpr_and_type(expr_arena, e.node(), &input_schema));
                    // Ignore Unknown in the inputs.
                    // We will raise if we cannot find the supertype later.
                    match dtype {
                        DataType::Unknown(UnknownKind::Any) => {
                            options.cast_to_supertypes = None;
                            return Ok(None);
                        },
                        _ => dtypes.push(dtype),
                    }
                }

                if dtypes.iter().all_equal() {
                    options.cast_to_supertypes = None;
                    return Ok(None);
                }

                // TODO! use args_to_supertype.
                let self_e = input[0].clone();
                let (self_ae, type_self) =
                    unpack!(get_aexpr_and_type(expr_arena, self_e.node(), &input_schema));

                let mut super_type = type_self.clone();
                for other in &input[1..] {
                    let (other, type_other) =
                        unpack!(get_aexpr_and_type(expr_arena, other.node(), &input_schema));

                    let Some(new_st) = get_supertype_with_options(
                        &super_type,
                        &type_other,
                        options.cast_to_supertypes.unwrap(),
                    ) else {
                        raise_supertype(function, input, &input_schema, expr_arena)?;
                        unreachable!()
                    };
                    if input.len() == 2 {
                        // modify_supertype is a bit more conservative of casting columns
                        // to literals
                        super_type =
                            modify_supertype(new_st, self_ae, other, &type_self, &type_other)
                    } else {
                        // when dealing with more than 1 argument, we simply find the supertypes
                        super_type = new_st
                    }
                }

                if matches!(super_type, DataType::Unknown(UnknownKind::Any)) {
                    raise_supertype(function, input, &input_schema, expr_arena)?;
                    unreachable!()
                }

                let function = function.clone();
                let input = input.clone();

                match super_type {
                    DataType::Unknown(UnknownKind::Float) => super_type = DataType::Float64,
                    DataType::Unknown(UnknownKind::Int(v)) => {
                        super_type = materialize_dyn_int(v).dtype()
                    },
                    _ => {},
                }

                let input = input
                    .into_iter()
                    .zip(dtypes)
                    .map(|(mut e, dtype)| {
                        match super_type {
                            #[cfg(feature = "dtype-categorical")]
                            DataType::Categorical(_, _) if dtype.is_string() => {
                                // pass
                            },
                            _ => {
                                if dtype != super_type {
                                    let n = expr_arena.add(AExpr::Cast {
                                        expr: e.node(),
                                        data_type: super_type.clone(),
                                        options: CastOptions::NonStrict,
                                    });
                                    e.set_node(n);
                                }
                            },
                        }
                        e
                    })
                    .collect::<Vec<_>>();

                // Ensure we don't go through this on next iteration.
                options.cast_to_supertypes = None;
                Some(AExpr::Function {
                    function,
                    input,
                    options,
                })
            },
            AExpr::Slice { offset, length, .. } => {
                let input_schema = get_schema(lp_arena, lp_node);
                let (_, offset_dtype) =
                    unpack!(get_aexpr_and_type(expr_arena, offset, &input_schema));
                polars_ensure!(offset_dtype.is_integer(), InvalidOperation: "offset must be integral for slice expression, not {}", offset_dtype);
                let (_, length_dtype) =
                    unpack!(get_aexpr_and_type(expr_arena, length, &input_schema));
                polars_ensure!(length_dtype.is_integer() || length_dtype.is_null(), InvalidOperation: "length must be integral for slice expression, not {}", length_dtype);
                None
            },
            _ => None,
        };
        Ok(out)
    }
}

fn inline_or_prune_cast(
    aexpr: &AExpr,
    dtype: &DataType,
    strict: bool,
    lp_node: Node,
    lp_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<Option<AExpr>> {
    if !dtype.is_known() {
        return Ok(None);
    }
    let lv = match (aexpr, dtype) {
        // PRUNE
        (
            AExpr::BinaryExpr {
                op: Operator::LogicalOr | Operator::LogicalAnd,
                ..
            },
            _,
        ) => {
            if let Some(schema) = lp_arena.get(lp_node).input_schema(lp_arena) {
                let field = aexpr.to_field(&schema, Context::Default, expr_arena)?;
                if field.dtype == *dtype {
                    return Ok(Some(aexpr.clone()));
                }
            }
            return Ok(None);
        },
        // INLINE
        (AExpr::Literal(lv), _) => match lv {
            LiteralValue::Series(s) => {
                let s = if strict {
                    s.strict_cast(dtype)
                } else {
                    s.cast(dtype)
                }?;
                LiteralValue::Series(SpecialEq::new(s))
            },
            LiteralValue::StrCat(s) => {
                let av = AnyValue::String(s).strict_cast(dtype);
                return Ok(av.map(|av| AExpr::Literal(av.try_into().unwrap())));
            },
            // We generate casted literal datetimes, so ensure we cast upon conversion
            // to create simpler expr trees.
            #[cfg(feature = "dtype-datetime")]
            LiteralValue::DateTime(ts, tu, None) if dtype.is_date() => {
                let from_size = time_unit_multiple(tu.to_arrow()) * SECONDS_IN_DAY;
                LiteralValue::Date((*ts / from_size) as i32)
            },
            lv @ (LiteralValue::Int(_) | LiteralValue::Float(_)) => {
                let av = lv.to_any_value().ok_or_else(|| polars_err!(InvalidOperation: "literal value: {:?} too large for Polars", lv))?;
                let av = av.strict_cast(dtype);
                return Ok(av.map(|av| AExpr::Literal(av.try_into().unwrap())));
            },
            LiteralValue::Null => match dtype {
                DataType::Unknown(UnknownKind::Float | UnknownKind::Int(_) | UnknownKind::Str) => {
                    return Ok(Some(AExpr::Literal(LiteralValue::Null)))
                },
                _ => return Ok(None),
            },
            _ => {
                let Some(av) = lv.to_any_value() else {
                    return Ok(None);
                };
                if dtype == &av.dtype() {
                    return Ok(Some(aexpr.clone()));
                }
                match (av, dtype) {
                    // casting null always remains null
                    (AnyValue::Null, _) => return Ok(None),
                    // series cast should do this one
                    #[cfg(feature = "dtype-datetime")]
                    (AnyValue::Datetime(_, _, _), DataType::Datetime(_, _)) => return Ok(None),
                    #[cfg(feature = "dtype-duration")]
                    (AnyValue::Duration(_, _), _) => return Ok(None),
                    #[cfg(feature = "dtype-categorical")]
                    (AnyValue::Categorical(_, _, _), _) | (_, DataType::Categorical(_, _)) => {
                        return Ok(None)
                    },
                    #[cfg(feature = "dtype-categorical")]
                    (AnyValue::Enum(_, _, _), _) | (_, DataType::Enum(_, _)) => return Ok(None),
                    #[cfg(feature = "dtype-struct")]
                    (_, DataType::Struct(_)) => return Ok(None),
                    (av, _) => {
                        let out = {
                            match av.strict_cast(dtype) {
                                Some(out) => out,
                                None => return Ok(None),
                            }
                        };
                        out.try_into()?
                    },
                }
            },
        },
        _ => return Ok(None),
    };
    Ok(Some(AExpr::Literal(lv)))
}

fn early_escape(type_self: &DataType, type_other: &DataType) -> Option<()> {
    match (type_self, type_other) {
        (lhs, rhs) if lhs == rhs => None,
        _ => Some(()),
    }
}

fn raise_supertype(
    function: &FunctionExpr,
    inputs: &[ExprIR],
    input_schema: &Schema,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<()> {
    let dtypes = inputs
        .iter()
        .map(|e| {
            let ae = expr_arena.get(e.node());
            ae.to_dtype(input_schema, Context::Default, expr_arena)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let st = dtypes
        .iter()
        .cloned()
        .map(Some)
        .reduce(|a, b| get_supertype(&a?, &b?))
        .expect("always at least 2 inputs");
    // We could get a supertype with the default options, so the input types are not allowed for this
    // specific operation.
    if st.is_some() {
        polars_bail!(InvalidOperation: "got invalid or ambiguous dtypes: '{}' in expression '{}'\
                        \n\nConsider explicitly casting your input types to resolve potential ambiguity.", format_list!(&dtypes), function);
    } else {
        polars_bail!(InvalidOperation: "could not determine supertype of: {} in expression '{}'\
                        \n\nIt might also be the case that the type combination isn't allowed in this specific operation.", format_list!(&dtypes), function);
    }
}

#[cfg(test)]
#[cfg(feature = "dtype-categorical")]
mod test {
    use polars_core::prelude::*;

    use super::*;

    #[test]
    fn test_categorical_string() {
        let mut expr_arena = Arena::new();
        let mut lp_arena = Arena::new();
        let optimizer = StackOptimizer {};
        let rules: &mut [Box<dyn OptimizationRule>] = &mut [Box::new(TypeCoercionRule {})];

        let df = DataFrame::new(Vec::from([Series::new_empty(
            "fruits",
            &DataType::Categorical(None, Default::default()),
        )]))
        .unwrap();

        let expr_in = vec![col("fruits").eq(lit("somestr"))];
        let lp = DslBuilder::from_existing_df(df.clone())
            .project(expr_in.clone(), Default::default())
            .build();

        let mut lp_top =
            to_alp(lp, &mut expr_arena, &mut lp_arena, &mut OptFlags::default()).unwrap();
        lp_top = optimizer
            .optimize_loop(rules, &mut expr_arena, &mut lp_arena, lp_top)
            .unwrap();
        let lp = node_to_lp(lp_top, &expr_arena, &mut lp_arena);

        // we test that the fruits column is not cast to string for the comparison
        if let DslPlan::Select { expr, .. } = lp {
            assert_eq!(expr, expr_in);
        };

        let expr_in = vec![col("fruits") + (lit("somestr"))];
        let lp = DslBuilder::from_existing_df(df)
            .project(expr_in, Default::default())
            .build();
        let mut lp_top =
            to_alp(lp, &mut expr_arena, &mut lp_arena, &mut OptFlags::default()).unwrap();
        lp_top = optimizer
            .optimize_loop(rules, &mut expr_arena, &mut lp_arena, lp_top)
            .unwrap();
        let lp = node_to_lp(lp_top, &expr_arena, &mut lp_arena);

        // we test that the fruits column is cast to string for the addition
        let expected = vec![col("fruits").cast(DataType::String) + lit("somestr")];
        if let DslPlan::Select { expr, .. } = lp {
            assert_eq!(expr, expected);
        };
    }
}
