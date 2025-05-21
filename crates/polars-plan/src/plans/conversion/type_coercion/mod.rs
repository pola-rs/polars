mod binary;
mod functions;
#[cfg(feature = "is_in")]
mod is_in;

use binary::process_binary;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_core::utils::{get_supertype, get_supertype_with_options, materialize_dyn_int};
use polars_utils::format_list;
use polars_utils::itertools::Itertools;

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
pub(super) use unpack;

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
            ref ae @ AExpr::Cast { .. } => {
                let AExpr::Cast {
                    expr: input_expr,
                    dtype,
                    options,
                } = ae.clone()
                else {
                    unreachable!()
                };

                let input = expr_arena.get(input_expr).clone();

                if let Some(schema) = lp_arena.get(lp_node).input_schema(lp_arena) {
                    if let CastOptions::Strict = options {
                        let cast_from = expr_arena
                            .get(input_expr)
                            .to_field(&schema, Context::Default, expr_arena)?
                            .dtype;
                        let cast_to = &dtype;

                        let v = CastColumnsPolicy {
                            integer_upcast: true,
                            float_upcast: true,
                            float_downcast: true,
                            datetime_nanoseconds_downcast: true,
                            datetime_microseconds_downcast: true,
                            datetime_convert_timezone: true,
                            missing_struct_fields: MissingColumnsPolicy::Insert,
                            extra_struct_fields: ExtraColumnsPolicy::Ignore,
                        }
                        .should_cast_column("", cast_to, &cast_from);

                        #[expect(clippy::single_match)]
                        match v {
                            // No casting needed
                            // TODO: Enable after release 1.30.0
                            // Ok(false) => {
                            //     return Ok(Some(expr_arena.get(input_expr).clone()));
                            // },
                            Ok(true | false) => {
                                let options = if cast_from.is_primitive_numeric()
                                    && cast_to.is_primitive_numeric()
                                {
                                    CastOptions::Overflowing
                                } else {
                                    CastOptions::NonStrict
                                };

                                let dtype = cast_to.clone();

                                expr_arena.replace(
                                    expr_node,
                                    AExpr::Cast {
                                        expr: input_expr,
                                        dtype,
                                        options,
                                    },
                                );
                            },

                            Err(_) => {},
                        }
                    }
                }

                inline_or_prune_cast(&input, &dtype, options, lp_node, lp_arena, expr_arena)?
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
                        dtype: st.clone(),
                        options: CastOptions::Strict,
                    })
                } else {
                    truthy_node
                };

                let new_node_falsy = if type_false != st {
                    expr_arena.add(AExpr::Cast {
                        expr: falsy_node,
                        dtype: st,
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
                ref function,
                ref input,
                options,
            } if {
                let mut matches = matches!(
                    function,
                    FunctionExpr::Boolean(BooleanFunction::IsIn { .. })
                        | FunctionExpr::ListExpr(ListFunction::Contains { .. })
                );
                #[cfg(feature = "dtype-array")]
                {
                    matches |= matches!(
                        function,
                        FunctionExpr::ArrayExpr(ArrayFunction::Contains { .. })
                    );
                }
                matches
            } =>
            {
                let (op, flat, nested, is_contains) = match function {
                    FunctionExpr::Boolean(BooleanFunction::IsIn { .. }) => ("is_in", 0, 1, false),
                    FunctionExpr::ListExpr(ListFunction::Contains { .. }) => {
                        ("list.contains", 1, 0, true)
                    },
                    #[cfg(feature = "dtype-array")]
                    FunctionExpr::ArrayExpr(ArrayFunction::Contains { .. }) => {
                        ("arr.contains", 1, 0, true)
                    },
                    _ => unreachable!(),
                };

                let Some(result) = is_in::resolve_is_in(
                    input,
                    expr_arena,
                    lp_arena,
                    lp_node,
                    is_contains,
                    op,
                    flat,
                    nested,
                )?
                else {
                    return Ok(None);
                };

                let function = function.clone();
                let mut input = input.to_vec();
                use self::is_in::IsInTypeCoercionResult;
                match result {
                    IsInTypeCoercionResult::SuperType(flat_type, nested_type) => {
                        let input_schema = get_schema(lp_arena, lp_node);
                        let (_, type_left) = unpack!(get_aexpr_and_type(
                            expr_arena,
                            input[flat].node(),
                            &input_schema
                        ));
                        let (_, type_other) = unpack!(get_aexpr_and_type(
                            expr_arena,
                            input[nested].node(),
                            &input_schema
                        ));
                        cast_expr_ir(
                            &mut input[flat],
                            &type_left,
                            &flat_type,
                            expr_arena,
                            CastOptions::NonStrict,
                        )?;
                        cast_expr_ir(
                            &mut input[nested],
                            &type_other,
                            &nested_type,
                            expr_arena,
                            CastOptions::NonStrict,
                        )?;
                    },
                    IsInTypeCoercionResult::SelfCast { dtype, strict } => {
                        let input_schema = get_schema(lp_arena, lp_node);
                        let (_, type_self) = unpack!(get_aexpr_and_type(
                            expr_arena,
                            input[flat].node(),
                            &input_schema
                        ));
                        let options = if strict {
                            CastOptions::Strict
                        } else {
                            CastOptions::NonStrict
                        };
                        cast_expr_ir(&mut input[flat], &type_self, &dtype, expr_arena, options)?;
                    },
                    IsInTypeCoercionResult::OtherCast { dtype, strict } => {
                        let input_schema = get_schema(lp_arena, lp_node);
                        let (_, type_other) = unpack!(get_aexpr_and_type(
                            expr_arena,
                            input[nested].node(),
                            &input_schema
                        ));
                        let options = if strict {
                            CastOptions::Strict
                        } else {
                            CastOptions::NonStrict
                        };
                        cast_expr_ir(&mut input[nested], &type_other, &dtype, expr_arena, options)?;
                    },
                    IsInTypeCoercionResult::Implode => {
                        assert!(!is_contains);
                        let other_input =
                            expr_arena.add(AExpr::Agg(IRAggExpr::Implode(input[1].node())));
                        input[1].set_node(other_input);
                    },
                }

                Some(AExpr::Function {
                    function,
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

                let mut input = input.clone();
                let new_node_left = if type_left != super_type {
                    expr_arena.add(AExpr::Cast {
                        expr: left_node,
                        dtype: super_type.clone(),
                        options: CastOptions::NonStrict,
                    })
                } else {
                    left_node
                };

                let new_node_fill_value = if type_fill_value != super_type {
                    expr_arena.add(AExpr::Cast {
                        expr: fill_value_node,
                        dtype: super_type.clone(),
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
            } if options.cast_options.is_some() => {
                let casting_rules = options.cast_options.unwrap();
                let input_schema = get_schema(lp_arena, lp_node);

                let function = function.clone();
                let mut input = input.clone();

                if let Some(dtypes) =
                    functions::get_function_dtypes(&input, expr_arena, &input_schema, &function)?
                {
                    let self_e = input[0].clone();
                    let (self_ae, type_self) =
                        unpack!(get_aexpr_and_type(expr_arena, self_e.node(), &input_schema));
                    let mut super_type = type_self.clone();
                    match casting_rules {
                        CastingRules::Supertype(super_type_opts) => {
                            for other in &input[1..] {
                                let (other, type_other) = unpack!(get_aexpr_and_type(
                                    expr_arena,
                                    other.node(),
                                    &input_schema
                                ));

                                let Some(new_st) = get_supertype_with_options(
                                    &super_type,
                                    &type_other,
                                    super_type_opts,
                                ) else {
                                    raise_supertype(&function, &input, &input_schema, expr_arena)?;
                                    unreachable!()
                                };
                                if input.len() == 2 {
                                    // modify_supertype is a bit more conservative of casting columns
                                    // to literals
                                    super_type = modify_supertype(
                                        new_st,
                                        self_ae,
                                        other,
                                        &type_self,
                                        &type_other,
                                    )
                                } else {
                                    // when dealing with more than 1 argument, we simply find the supertypes
                                    super_type = new_st
                                }
                            }
                        },
                        CastingRules::FirstArgLossless => {
                            if super_type.is_integer() {
                                for other in &input[1..] {
                                    let other =
                                        other.dtype(&input_schema, Context::Default, expr_arena)?;
                                    if other.is_float() {
                                        polars_bail!(InvalidOperation: "cannot cast lossless between {} and {}", super_type, other)
                                    }
                                }
                            }
                        },
                    }

                    if matches!(super_type, DataType::Unknown(UnknownKind::Any)) {
                        raise_supertype(&function, &input, &input_schema, expr_arena)?;
                        unreachable!()
                    }

                    match super_type {
                        DataType::Unknown(UnknownKind::Float) => super_type = DataType::Float64,
                        DataType::Unknown(UnknownKind::Int(v)) => {
                            super_type = materialize_dyn_int(v).dtype()
                        },
                        _ => {},
                    }

                    for (e, dtype) in input.iter_mut().zip(dtypes) {
                        match super_type {
                            #[cfg(feature = "dtype-categorical")]
                            DataType::Categorical(_, _) if dtype.is_string() => {
                                // pass
                            },
                            _ => cast_expr_ir(
                                e,
                                &dtype,
                                &super_type,
                                expr_arena,
                                CastOptions::NonStrict,
                            )?,
                        }
                    }
                }

                // Ensure we don't go through this on next iteration.
                options.cast_options = None;
                Some(AExpr::Function {
                    function,
                    input,
                    options,
                })
            },
            #[cfg(all(feature = "temporal", feature = "dtype-duration"))]
            AExpr::Function {
                function: ref function @ FunctionExpr::TemporalExpr(TemporalFunction::Duration(_)),
                ref input,
                options,
            } => {
                let input_schema = get_schema(lp_arena, lp_node);

                for (i, expr) in input.iter().enumerate() {
                    let (_, dtype) =
                        unpack!(get_aexpr_and_type(expr_arena, expr.node(), &input_schema));

                    if !matches!(dtype, DataType::Int64) {
                        let function = function.clone();
                        let mut input = input.to_vec();
                        cast_expr_ir(
                            &mut input[i],
                            &dtype,
                            &DataType::Int64,
                            expr_arena,
                            CastOptions::NonStrict,
                        )?;
                        for expr in &mut input[i + 1..] {
                            let (_, dtype) =
                                unpack!(get_aexpr_and_type(expr_arena, expr.node(), &input_schema));
                            cast_expr_ir(
                                expr,
                                &dtype,
                                &DataType::Int64,
                                expr_arena,
                                CastOptions::Strict,
                            )?;
                        }

                        return Ok(Some(AExpr::Function {
                            function,
                            input,
                            options,
                        }));
                    }
                }

                None
            },
            #[cfg(feature = "list_gather")]
            AExpr::Function {
                function: ref function @ FunctionExpr::ListExpr(ListFunction::Gather(_)),
                ref input,
                options,
            } => {
                let input_schema = get_schema(lp_arena, lp_node);
                let (_, type_left) = unpack!(get_aexpr_and_type(
                    expr_arena,
                    input[0].node(),
                    &input_schema
                ));
                let (_, type_other) = unpack!(get_aexpr_and_type(
                    expr_arena,
                    input[1].node(),
                    &input_schema
                ));

                let DataType::List(inner_dtype) = &type_other else {
                    // @HACK. This needs to happen until 2.0 because we support
                    // `pl.col.a.list.gather(0)` and `pl.col.a.list.gather(pl.col.b)` where `b` is
                    // an integer.
                    let function = function.clone();
                    let mut input = input.clone();

                    polars_warn!(
                        Deprecation,
                        "`list.gather` with a flat datatype is deprecated.
Please use `implode` to return to previous behavior.

See https://github.com/pola-rs/polars/issues/22149 for more information."
                    );

                    let other_input =
                        expr_arena.add(AExpr::Agg(IRAggExpr::Implode(input[1].node())));
                    input[1].set_node(other_input);

                    return Ok(Some(AExpr::Function {
                        function,
                        input,
                        options,
                    }));
                };

                polars_ensure!(
                    inner_dtype.is_integer(),
                    op = "list.gather",
                    type_left,
                    type_other
                );
                None
            },
            #[cfg(all(feature = "strings", feature = "find_many"))]
            AExpr::Function {
                function:
                    ref function @ FunctionExpr::StringExpr(
                        StringFunction::ContainsAny { .. }
                        | StringFunction::FindMany { .. }
                        | StringFunction::ExtractMany { .. },
                    ),
                ref input,
                options,
            } => {
                let input_schema = get_schema(lp_arena, lp_node);
                let (_, type_left) = unpack!(get_aexpr_and_type(
                    expr_arena,
                    input[0].node(),
                    &input_schema
                ));
                let (_, type_other) = unpack!(get_aexpr_and_type(
                    expr_arena,
                    input[1].node(),
                    &input_schema
                ));

                let DataType::List(inner_dtype) = &type_other else {
                    // @HACK. This needs to happen until 2.0 because we support
                    // `pl.col.a.str.contains_any(pl.col.b)` where `b` is a string.
                    let function = function.clone();
                    let mut input = input.clone();

                    polars_warn!(
                        Deprecation,
                        "`{function}` with a flat string datatype is deprecated.
Please use `implode` to return to previous behavior.
See https://github.com/pola-rs/polars/issues/22149 for more information."
                    );

                    let other_input =
                        expr_arena.add(AExpr::Agg(IRAggExpr::Implode(input[1].node())));
                    input[1].set_node(other_input);

                    return Ok(Some(AExpr::Function {
                        function,
                        input,
                        options,
                    }));
                };

                polars_ensure!(
                    type_left.is_string() && inner_dtype.is_string(),
                    op = format!("{function}"),
                    type_left,
                    type_other
                );
                None
            },
            #[cfg(all(feature = "strings", feature = "find_many"))]
            AExpr::Function {
                function:
                    ref function @ FunctionExpr::StringExpr(StringFunction::ReplaceMany { .. }),
                ref input,
                options,
            } => {
                let input_schema = get_schema(lp_arena, lp_node);
                let (_, type_left) = unpack!(get_aexpr_and_type(
                    expr_arena,
                    input[0].node(),
                    &input_schema
                ));
                let (_, type_patterns) = unpack!(get_aexpr_and_type(
                    expr_arena,
                    input[1].node(),
                    &input_schema
                ));
                let (_, type_replace_with) = unpack!(get_aexpr_and_type(
                    expr_arena,
                    input[2].node(),
                    &input_schema
                ));

                let (
                    DataType::List(type_patterns_inner_dtype),
                    DataType::List(type_replace_with_inner_dtype),
                ) = (&type_patterns, &type_replace_with)
                else {
                    // @HACK. This needs to happen until 2.0 because we support
                    // `pl.col.a.str.replace_with(pl.col.b, ..)` where `b` is a string.
                    let function = function.clone();
                    let mut input = input.clone();

                    polars_warn!(
                        Deprecation,
                        "`str.replace_many` with a flat string datatype is deprecated.
please use `implode` to return to previous behavior.
See https://github.com/pola-rs/polars/issues/22149 for more information."
                    );

                    if !type_patterns.is_list() {
                        let other_input =
                            expr_arena.add(AExpr::Agg(IRAggExpr::Implode(input[1].node())));
                        input[1].set_node(other_input);
                    }
                    if !type_replace_with.is_list() {
                        let other_input =
                            expr_arena.add(AExpr::Agg(IRAggExpr::Implode(input[2].node())));
                        input[2].set_node(other_input);
                    }

                    return Ok(Some(AExpr::Function {
                        function,
                        input,
                        options,
                    }));
                };

                polars_ensure!(
                    type_left.is_string()
                        && type_patterns_inner_dtype.is_string()
                        && type_replace_with_inner_dtype.is_string(),
                    op = "str.replace_many",
                    type_left,
                    type_patterns,
                    type_replace_with
                );
                None
            },
            #[cfg(feature = "replace")]
            AExpr::Function {
                function:
                    ref function @ (FunctionExpr::Replace | FunctionExpr::ReplaceStrict { .. }),
                ref input,
                options,
            } => {
                let input_schema = get_schema(lp_arena, lp_node);
                let (_, type_old) = unpack!(get_aexpr_and_type(
                    expr_arena,
                    input[1].node(),
                    &input_schema
                ));
                let (_, type_new) = unpack!(get_aexpr_and_type(
                    expr_arena,
                    input[2].node(),
                    &input_schema
                ));

                let (DataType::List(_), DataType::List(_)) = (&type_old, &type_new) else {
                    let function = function.clone();
                    let mut input = input.clone();

                    if !type_old.is_list() {
                        let other_input =
                            expr_arena.add(AExpr::Agg(IRAggExpr::Implode(input[1].node())));
                        input[1].set_node(other_input);
                    }
                    if !type_new.is_list() {
                        let other_input =
                            expr_arena.add(AExpr::Agg(IRAggExpr::Implode(input[2].node())));
                        input[2].set_node(other_input);
                    }

                    return Ok(Some(AExpr::Function {
                        function,
                        input,
                        options,
                    }));
                };

                None
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
    options: CastOptions,
    lp_node: Node,
    lp_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<Option<AExpr>> {
    if !dtype.is_known() {
        return Ok(None);
    }

    let out = match aexpr {
        // PRUNE
        AExpr::BinaryExpr { op, .. } => {
            use Operator::*;

            match op {
                LogicalOr | LogicalAnd => {
                    if let Some(schema) = lp_arena.get(lp_node).input_schema(lp_arena) {
                        let field = aexpr.to_field(&schema, Context::Default, expr_arena)?;
                        if field.dtype == *dtype {
                            return Ok(Some(aexpr.clone()));
                        }
                    }

                    None
                },
                Eq | EqValidity | NotEq | NotEqValidity | Lt | LtEq | Gt | GtEq => {
                    if dtype.is_bool() {
                        Some(aexpr.clone())
                    } else {
                        None
                    }
                },
                _ => None,
            }
        },
        // INLINE
        AExpr::Literal(lv) => try_inline_literal_cast(lv, dtype, options)?.map(AExpr::Literal),
        _ => None,
    };

    Ok(out)
}

fn try_inline_literal_cast(
    lv: &LiteralValue,
    dtype: &DataType,
    options: CastOptions,
) -> PolarsResult<Option<LiteralValue>> {
    let lv = match lv {
        LiteralValue::Series(s) => {
            let s = s.cast_with_options(dtype, options)?;
            LiteralValue::Series(SpecialEq::new(s))
        },
        LiteralValue::Dyn(dyn_value) => dyn_value.clone().try_materialize_to_dtype(dtype)?.into(),
        lv if lv.is_null() => match dtype {
            DataType::Unknown(UnknownKind::Float | UnknownKind::Int(_) | UnknownKind::Str) => {
                LiteralValue::untyped_null()
            },
            _ => return Ok(None),
        },
        LiteralValue::Scalar(sc) => sc.clone().cast_with_options(dtype, options)?.into(),
        lv => {
            let Some(av) = lv.to_any_value() else {
                return Ok(None);
            };
            if dtype == &av.dtype() {
                return Ok(Some(lv.clone()));
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
                    return Ok(None);
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
                    out.into()
                },
            }
        },
    };

    Ok(Some(lv))
}

fn cast_expr_ir(
    e: &mut ExprIR,
    from_dtype: &DataType,
    to_dtype: &DataType,
    expr_arena: &mut Arena<AExpr>,
    options: CastOptions,
) -> PolarsResult<()> {
    if from_dtype == to_dtype {
        return Ok(());
    }

    check_cast(from_dtype, to_dtype)?;

    if let AExpr::Literal(lv) = expr_arena.get(e.node()) {
        if let Some(literal) = try_inline_literal_cast(lv, to_dtype, options)? {
            e.set_node(expr_arena.add(AExpr::Literal(literal)));
            e.set_dtype(to_dtype.clone());
            return Ok(());
        }
    }

    e.set_node(expr_arena.add(AExpr::Cast {
        expr: e.node(),
        dtype: to_dtype.clone(),
        options: CastOptions::Strict,
    }));
    e.set_dtype(to_dtype.clone());

    Ok(())
}

fn check_cast(from: &DataType, to: &DataType) -> PolarsResult<()> {
    polars_ensure!(
        from.can_cast_to(to) != Some(false),
        InvalidOperation: "casting from {from:?} to {to:?} not supported"
    );
    Ok(())
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
        .map(|e| e.dtype(input_schema, Context::Default, expr_arena).cloned())
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

        let df = DataFrame::new(Vec::from([Column::new_empty(
            PlSmallStr::from_static("fruits"),
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
