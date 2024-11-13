use recursive::recursive;

use super::*;

fn float_type(field: &mut Field) {
    let should_coerce = match &field.dtype {
        DataType::Float32 => false,
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(..) => true,
        DataType::Boolean => true,
        dt => dt.is_numeric(),
    };
    if should_coerce {
        field.coerce(DataType::Float64);
    }
}

impl AExpr {
    pub fn to_dtype(
        &self,
        schema: &Schema,
        ctx: Context,
        arena: &Arena<AExpr>,
    ) -> PolarsResult<DataType> {
        self.to_field(schema, ctx, arena).map(|f| f.dtype)
    }

    /// Get Field result of the expression. The schema is the input data.
    pub fn to_field(
        &self,
        schema: &Schema,
        ctx: Context,
        arena: &Arena<AExpr>,
    ) -> PolarsResult<Field> {
        // Indicates whether we should auto-implode the result. This is initialized to true if we are
        // in an aggregation context, so functions that return scalars should explicitly set this
        // to false in `to_field_impl`.
        let mut agg_list = matches!(ctx, Context::Aggregation);
        let mut field = self.to_field_impl(schema, ctx, arena, &mut agg_list)?;

        if agg_list {
            field.coerce(field.dtype().clone().implode());
        }

        Ok(field)
    }

    /// Get Field result of the expression. The schema is the input data.
    ///
    /// This is taken as `&mut bool` as for some expressions this is determined by the upper node
    /// (e.g. `alias`, `cast`).
    #[recursive]
    pub fn to_field_impl(
        &self,
        schema: &Schema,
        ctx: Context,
        arena: &Arena<AExpr>,
        agg_list: &mut bool,
    ) -> PolarsResult<Field> {
        use AExpr::*;
        use DataType::*;
        match self {
            Len => {
                *agg_list = false;
                Ok(Field::new(PlSmallStr::from_static(LEN), IDX_DTYPE))
            },
            Window {
                function, options, ..
            } => {
                if let WindowType::Over(WindowMapping::Join) = options {
                    // expr.over(..), defaults to agg-list unless explicitly unset
                    // by the `to_field_impl` of the `expr`
                    *agg_list = true;
                }

                let e = arena.get(*function);
                e.to_field_impl(schema, ctx, arena, agg_list)
            },
            Explode(expr) => {
                // `Explode` is a "flatten" operation, which is not the same as returning a scalar.
                // Namely, it should be auto-imploded in the aggregation context, so we don't update
                // the `agg_list` state here.
                let field = arena
                    .get(*expr)
                    .to_field_impl(schema, ctx, arena, &mut false)?;

                if let List(inner) = field.dtype() {
                    Ok(Field::new(field.name().clone(), *inner.clone()))
                } else {
                    Ok(field)
                }
            },
            Alias(expr, name) => Ok(Field::new(
                name.clone(),
                arena
                    .get(*expr)
                    .to_field_impl(schema, ctx, arena, agg_list)?
                    .dtype,
            )),
            Column(name) => schema
                .get_field(name)
                .ok_or_else(|| PolarsError::ColumnNotFound(name.to_string().into())),
            Literal(sv) => {
                *agg_list = false;
                Ok(match sv {
                    LiteralValue::Series(s) => s.field().into_owned(),
                    _ => Field::new(sv.output_name().clone(), sv.get_datatype()),
                })
            },
            BinaryExpr { left, right, op } => {
                use DataType::*;

                let field = match op {
                    Operator::Lt
                    | Operator::Gt
                    | Operator::Eq
                    | Operator::NotEq
                    | Operator::LogicalAnd
                    | Operator::LtEq
                    | Operator::GtEq
                    | Operator::NotEqValidity
                    | Operator::EqValidity
                    | Operator::LogicalOr => {
                        let out_field;
                        let out_name = {
                            out_field = arena
                                .get(*left)
                                .to_field_impl(schema, ctx, arena, agg_list)?;
                            out_field.name()
                        };
                        Field::new(out_name.clone(), Boolean)
                    },
                    Operator::TrueDivide => {
                        return get_truediv_field(*left, *right, arena, ctx, schema, agg_list)
                    },
                    _ => {
                        return get_arithmetic_field(
                            *left, *right, arena, *op, ctx, schema, agg_list,
                        )
                    },
                };

                Ok(field)
            },
            Sort { expr, .. } => arena.get(*expr).to_field_impl(schema, ctx, arena, agg_list),
            Gather {
                expr,
                returns_scalar,
                ..
            } => {
                if *returns_scalar {
                    *agg_list = false;
                }
                arena
                    .get(*expr)
                    .to_field_impl(schema, ctx, arena, &mut false)
            },
            SortBy { expr, .. } => arena.get(*expr).to_field_impl(schema, ctx, arena, agg_list),
            Filter { input, .. } => arena
                .get(*input)
                .to_field_impl(schema, ctx, arena, agg_list),
            Agg(agg) => {
                use IRAggExpr::*;
                match agg {
                    Max { input: expr, .. }
                    | Min { input: expr, .. }
                    | First(expr)
                    | Last(expr) => {
                        *agg_list = false;
                        arena
                            .get(*expr)
                            .to_field_impl(schema, ctx, arena, &mut false)
                    },
                    Sum(expr) => {
                        *agg_list = false;
                        let mut field = arena
                            .get(*expr)
                            .to_field_impl(schema, ctx, arena, &mut false)?;
                        let dt = match field.dtype() {
                            Boolean => Some(IDX_DTYPE),
                            UInt8 | Int8 | Int16 | UInt16 => Some(Int64),
                            _ => None,
                        };
                        if let Some(dt) = dt {
                            field.coerce(dt);
                        }
                        Ok(field)
                    },
                    Median(expr) => {
                        *agg_list = false;
                        let mut field = arena
                            .get(*expr)
                            .to_field_impl(schema, ctx, arena, &mut false)?;
                        match field.dtype {
                            Date => field.coerce(Datetime(TimeUnit::Milliseconds, None)),
                            _ => float_type(&mut field),
                        }
                        Ok(field)
                    },
                    Mean(expr) => {
                        *agg_list = false;
                        let mut field = arena
                            .get(*expr)
                            .to_field_impl(schema, ctx, arena, &mut false)?;
                        match field.dtype {
                            Date => field.coerce(Datetime(TimeUnit::Milliseconds, None)),
                            _ => float_type(&mut field),
                        }
                        Ok(field)
                    },
                    Implode(expr) => {
                        let mut field = arena
                            .get(*expr)
                            .to_field_impl(schema, ctx, arena, &mut false)?;
                        field.coerce(DataType::List(field.dtype().clone().into()));
                        Ok(field)
                    },
                    Std(expr, _) => {
                        *agg_list = false;
                        let mut field = arena
                            .get(*expr)
                            .to_field_impl(schema, ctx, arena, &mut false)?;
                        float_type(&mut field);
                        Ok(field)
                    },
                    Var(expr, _) => {
                        *agg_list = false;
                        let mut field = arena
                            .get(*expr)
                            .to_field_impl(schema, ctx, arena, &mut false)?;
                        float_type(&mut field);
                        Ok(field)
                    },
                    NUnique(expr) => {
                        *agg_list = false;
                        let mut field = arena
                            .get(*expr)
                            .to_field_impl(schema, ctx, arena, &mut false)?;
                        field.coerce(IDX_DTYPE);
                        Ok(field)
                    },
                    Count(expr, _) => {
                        *agg_list = false;
                        let mut field = arena
                            .get(*expr)
                            .to_field_impl(schema, ctx, arena, &mut false)?;
                        field.coerce(IDX_DTYPE);
                        Ok(field)
                    },
                    AggGroups(expr) => {
                        *agg_list = true;
                        let mut field = arena
                            .get(*expr)
                            .to_field_impl(schema, ctx, arena, &mut false)?;
                        field.coerce(List(IDX_DTYPE.into()));
                        Ok(field)
                    },
                    Quantile { expr, .. } => {
                        *agg_list = false;
                        let mut field = arena
                            .get(*expr)
                            .to_field_impl(schema, ctx, arena, &mut false)?;
                        float_type(&mut field);
                        Ok(field)
                    },
                    #[cfg(feature = "bitwise")]
                    Bitwise(expr, _) => {
                        *agg_list = false;
                        let field = arena
                            .get(*expr)
                            .to_field_impl(schema, ctx, arena, &mut false)?;
                        // @Q? Do we need to coerce here?
                        Ok(field)
                    },
                }
            },
            Cast { expr, dtype, .. } => {
                let field = arena
                    .get(*expr)
                    .to_field_impl(schema, ctx, arena, agg_list)?;
                Ok(Field::new(field.name().clone(), dtype.clone()))
            },
            Ternary { truthy, falsy, .. } => {
                let mut agg_list_truthy = *agg_list;
                let mut agg_list_falsy = *agg_list;

                // During aggregation:
                // left: col(foo):              list<T>         nesting: 1
                // right; col(foo).first():     T               nesting: 0
                // col(foo) + col(foo).first() will have nesting 1 as we still maintain the groups list.
                let mut truthy =
                    arena
                        .get(*truthy)
                        .to_field_impl(schema, ctx, arena, &mut agg_list_truthy)?;
                let falsy =
                    arena
                        .get(*falsy)
                        .to_field_impl(schema, ctx, arena, &mut agg_list_falsy)?;

                let st = if let DataType::Null = *truthy.dtype() {
                    falsy.dtype().clone()
                } else {
                    try_get_supertype(truthy.dtype(), falsy.dtype())?
                };

                *agg_list = agg_list_truthy | agg_list_falsy;

                truthy.coerce(st);
                Ok(truthy)
            },
            AnonymousFunction {
                output_type,
                input,
                options,
                ..
            } => {
                let fields = func_args_to_fields(input, ctx, schema, arena, agg_list)?;
                polars_ensure!(!fields.is_empty(), ComputeError: "expression: '{}' didn't get any inputs", options.fmt_str);
                let out = output_type.get_field(schema, ctx, &fields)?;

                if options.flags.contains(FunctionFlags::RETURNS_SCALAR) {
                    *agg_list = false;
                } else if matches!(ctx, Context::Aggregation) {
                    *agg_list = true;
                }

                Ok(out)
            },
            Function {
                function,
                input,
                options,
            } => {
                let fields = func_args_to_fields(input, ctx, schema, arena, agg_list)?;
                polars_ensure!(!fields.is_empty(), ComputeError: "expression: '{}' didn't get any inputs", function);
                let out = function.get_field(schema, ctx, &fields)?;

                if options.flags.contains(FunctionFlags::RETURNS_SCALAR) {
                    *agg_list = false;
                } else if matches!(ctx, Context::Aggregation) {
                    *agg_list = true;
                }

                Ok(out)
            },
            Slice { input, .. } => arena
                .get(*input)
                .to_field_impl(schema, ctx, arena, agg_list),
        }
    }
}

fn func_args_to_fields(
    input: &[ExprIR],
    ctx: Context,
    schema: &Schema,
    arena: &Arena<AExpr>,
    agg_list: &mut bool,
) -> PolarsResult<Vec<Field>> {
    input
        .iter()
        .enumerate()
        // Default context because `col()` would return a list in aggregation context
        .map(|(i, e)| {
            let tmp = &mut false;

            arena
                .get(e.node())
                .to_field_impl(
                    schema,
                    ctx,
                    arena,
                    if i == 0 {
                        // Only mutate first agg_list as that is the dtype of the function.
                        agg_list
                    } else {
                        tmp
                    },
                )
                .map(|mut field| {
                    field.name = e.output_name().clone();
                    field
                })
        })
        .collect()
}

fn get_arithmetic_field(
    left: Node,
    right: Node,
    arena: &Arena<AExpr>,
    op: Operator,
    ctx: Context,
    schema: &Schema,
    agg_list: &mut bool,
) -> PolarsResult<Field> {
    use DataType::*;
    let left_ae = arena.get(left);
    let right_ae = arena.get(right);

    // don't traverse tree until strictly needed. Can have terrible performance.
    // # 3210

    // take the left field as a whole.
    // don't take dtype and name separate as that splits the tree every node
    // leading to quadratic behavior. # 4736
    //
    // further right_type is only determined when needed.
    let mut left_field = left_ae.to_field_impl(schema, ctx, arena, agg_list)?;

    let super_type = match op {
        Operator::Minus => {
            let right_type = right_ae.to_field_impl(schema, ctx, arena, agg_list)?.dtype;
            match (&left_field.dtype, &right_type) {
                #[cfg(feature = "dtype-struct")]
                (Struct(_), Struct(_)) => {
                    return Ok(left_field);
                },
                (Duration(_), Datetime(_, _))
                | (Datetime(_, _), Duration(_))
                | (Duration(_), Date)
                | (Date, Duration(_))
                | (Duration(_), Time)
                | (Time, Duration(_)) => try_get_supertype(left_field.dtype(), &right_type)?,
                (Datetime(tu, _), Date) | (Date, Datetime(tu, _)) => Duration(*tu),
                // T - T != T if T is a datetime / date
                (Datetime(tul, _), Datetime(tur, _)) => Duration(get_time_units(tul, tur)),
                (_, Datetime(_, _)) | (Datetime(_, _), _) => {
                    polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                },
                (Date, Date) => Duration(TimeUnit::Milliseconds),
                (_, Date) | (Date, _) => {
                    polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                },
                (Duration(tul), Duration(tur)) => Duration(get_time_units(tul, tur)),
                (_, Duration(_)) | (Duration(_), _) => {
                    polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                },
                (_, Time) | (Time, _) => {
                    polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                },
                (l @ List(a), r @ List(b))
                    if ![a, b]
                        .into_iter()
                        .all(|x| x.is_numeric() || x.is_bool() || x.is_null()) =>
                {
                    polars_bail!(
                        InvalidOperation:
                        "cannot {} two list columns with non-numeric inner types: (left: {}, right: {})",
                        "sub", l, r,
                    )
                },
                (list_dtype @ List(_), other_dtype) | (other_dtype, list_dtype @ List(_)) => {
                    // FIXME: This should not use `try_get_supertype()`! It should instead recursively use the enclosing match block.
                    // Otherwise we will silently permit addition operations between logical types (see above).
                    // This currently doesn't cause any problems because the list arithmetic implementation checks and raises errors
                    // if the leaf types aren't numeric, but it means we don't raise an error until execution and the DSL schema
                    // may be incorrect.
                    list_dtype.cast_leaf(try_get_supertype(
                        list_dtype.leaf_dtype(),
                        other_dtype.leaf_dtype(),
                    )?)
                },
                (left, right) => try_get_supertype(left, right)?,
            }
        },
        Operator::Plus => {
            let right_type = right_ae.to_field_impl(schema, ctx, arena, agg_list)?.dtype;
            match (&left_field.dtype, &right_type) {
                (Duration(_), Datetime(_, _))
                | (Datetime(_, _), Duration(_))
                | (Duration(_), Date)
                | (Date, Duration(_))
                | (Duration(_), Time)
                | (Time, Duration(_)) => try_get_supertype(left_field.dtype(), &right_type)?,
                (_, Datetime(_, _))
                | (Datetime(_, _), _)
                | (_, Date)
                | (Date, _)
                | (Time, _)
                | (_, Time) => {
                    polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                },
                (Duration(tul), Duration(tur)) => Duration(get_time_units(tul, tur)),
                (_, Duration(_)) | (Duration(_), _) => {
                    polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                },
                (Boolean, Boolean) => IDX_DTYPE,
                (l @ List(a), r @ List(b))
                    if ![a, b]
                        .into_iter()
                        .all(|x| x.is_numeric() || x.is_bool() || x.is_null()) =>
                {
                    polars_bail!(
                        InvalidOperation:
                        "cannot {} two list columns with non-numeric inner types: (left: {}, right: {})",
                        "add", l, r,
                    )
                },
                (list_dtype @ List(_), other_dtype) | (other_dtype, list_dtype @ List(_)) => {
                    list_dtype.cast_leaf(try_get_supertype(
                        list_dtype.leaf_dtype(),
                        other_dtype.leaf_dtype(),
                    )?)
                },
                (left, right) => try_get_supertype(left, right)?,
            }
        },
        _ => {
            let right_type = right_ae.to_field_impl(schema, ctx, arena, agg_list)?.dtype;

            match (&left_field.dtype, &right_type) {
                #[cfg(feature = "dtype-struct")]
                (Struct(_), Struct(_)) => {
                    return Ok(left_field);
                },
                (Datetime(_, _), _)
                | (_, Datetime(_, _))
                | (Time, _)
                | (_, Time)
                | (Date, _)
                | (_, Date) => {
                    polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                },
                (Duration(_), Duration(_)) => {
                    // True divide handled somewhere else
                    polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                },
                (l, Duration(_)) if l.is_numeric() => match op {
                    Operator::Multiply => {
                        left_field.coerce(right_type);
                        return Ok(left_field);
                    },
                    _ => {
                        polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                    },
                },
                (l @ List(a), r @ List(b))
                    if ![a, b]
                        .into_iter()
                        .all(|x| x.is_numeric() || x.is_bool() || x.is_null()) =>
                {
                    polars_bail!(
                        InvalidOperation:
                        "cannot {} two list columns with non-numeric inner types: (left: {}, right: {})",
                        op, l, r,
                    )
                },
                // List<->primitive operations can be done directly after casting the to the primitive
                // supertype for the primitive values on both sides.
                (list_dtype @ List(_), other_dtype) | (other_dtype, list_dtype @ List(_)) => {
                    let dtype = list_dtype.cast_leaf(try_get_supertype(
                        list_dtype.leaf_dtype(),
                        other_dtype.leaf_dtype(),
                    )?);
                    left_field.coerce(dtype);
                    return Ok(left_field);
                },
                _ => {
                    // Avoid needlessly type casting numeric columns during arithmetic
                    // with literals.
                    if (left_field.dtype.is_integer() && right_type.is_integer())
                        || (left_field.dtype.is_float() && right_type.is_float())
                    {
                        match (left_ae, right_ae) {
                            (AExpr::Literal(_), AExpr::Literal(_)) => {},
                            (AExpr::Literal(_), _) => {
                                // literal will be coerced to match right type
                                left_field.coerce(right_type);
                                return Ok(left_field);
                            },
                            (_, AExpr::Literal(_)) => {
                                // literal will be coerced to match right type
                                return Ok(left_field);
                            },
                            _ => {},
                        }
                    }
                },
            }

            try_get_supertype(&left_field.dtype, &right_type)?
        },
    };

    left_field.coerce(super_type);
    Ok(left_field)
}

fn get_truediv_field(
    left: Node,
    right: Node,
    arena: &Arena<AExpr>,
    ctx: Context,
    schema: &Schema,
    agg_list: &mut bool,
) -> PolarsResult<Field> {
    let mut left_field = arena
        .get(left)
        .to_field_impl(schema, ctx, arena, agg_list)?;
    let right_field = arena
        .get(right)
        .to_field_impl(schema, ctx, arena, agg_list)?;
    use DataType::*;

    // TODO: Re-investigate this. A lot of "_" is being used on the RHS match because this code
    // originally (mostly) only looked at the LHS dtype.
    let out_type = match (left_field.dtype(), right_field.dtype()) {
        (l @ List(a), r @ List(b))
            if ![a, b]
                .into_iter()
                .all(|x| x.is_numeric() || x.is_bool() || x.is_null()) =>
        {
            polars_bail!(
                InvalidOperation:
                "cannot {} two list columns with non-numeric inner types: (left: {}, right: {})",
                "div", l, r,
            )
        },
        (list_dtype @ List(_), other_dtype) | (other_dtype, list_dtype @ List(_)) => {
            list_dtype.cast_leaf(match (list_dtype.leaf_dtype(), other_dtype.leaf_dtype()) {
                (Float32, Float32) => Float32,
                (Float32, Float64) | (Float64, Float32) => Float64,
                // FIXME: We should properly recurse on the enclosing match block here.
                (dt, _) => dt.clone(),
            })
        },
        (Float32, _) => Float32,
        (dt, _) if dt.is_numeric() => Float64,
        #[cfg(feature = "dtype-duration")]
        (Duration(_), Duration(_)) => Float64,
        #[cfg(feature = "dtype-duration")]
        (Duration(_), dt) if dt.is_numeric() => return Ok(left_field),
        #[cfg(feature = "dtype-duration")]
        (Duration(_), dt) => {
            polars_bail!(InvalidOperation: "true division of {} with {} is not allowed", left_field.dtype(), dt)
        },
        #[cfg(feature = "dtype-datetime")]
        (Datetime(_, _), _) => {
            polars_bail!(InvalidOperation: "division of 'Datetime' datatype is not allowed")
        },
        #[cfg(feature = "dtype-time")]
        (Time, _) => polars_bail!(InvalidOperation: "division of 'Time' datatype is not allowed"),
        #[cfg(feature = "dtype-date")]
        (Date, _) => polars_bail!(InvalidOperation: "division of 'Date' datatype is not allowed"),
        // we don't know what to do here, best return the dtype
        (dt, _) => dt.clone(),
    };

    left_field.coerce(out_type);
    Ok(left_field)
}
