use recursive::recursive;

use super::*;

fn float_type(field: &mut Field) {
    if (field.dtype.is_numeric() || field.dtype == DataType::Boolean)
        && field.dtype != DataType::Float32
    {
        field.coerce(DataType::Float64)
    }
}

impl AExpr {
    pub fn to_dtype(
        &self,
        schema: &Schema,
        ctxt: Context,
        arena: &Arena<AExpr>,
    ) -> PolarsResult<DataType> {
        self.to_field(schema, ctxt, arena).map(|f| f.dtype)
    }

    /// Get Field result of the expression. The schema is the input data.
    pub fn to_field(
        &self,
        schema: &Schema,
        ctxt: Context,
        arena: &Arena<AExpr>,
    ) -> PolarsResult<Field> {
        // During aggregation a column that isn't aggregated gets an extra nesting level
        //      col(foo: i64) -> list[i64]
        // But not if we do an aggregation:
        //      col(foo: i64).sum() -> i64
        // The `nested` keeps track of the nesting we need to add.
        let mut nested = matches!(ctxt, Context::Aggregation) as u8;
        let mut field = self.to_field_impl(schema, arena, &mut nested)?;

        if nested >= 1 {
            field.coerce(field.data_type().clone().implode());
        }
        Ok(field)
    }

    /// Get Field result of the expression. The schema is the input data.
    #[recursive]
    pub fn to_field_impl(
        &self,
        schema: &Schema,
        arena: &Arena<AExpr>,
        nested: &mut u8,
    ) -> PolarsResult<Field> {
        use AExpr::*;
        use DataType::*;
        match self {
            Len => {
                *nested = 0;
                Ok(Field::new(LEN, IDX_DTYPE))
            },
            Window { function, .. } => {
                let e = arena.get(*function);
                e.to_field_impl(schema, arena, nested)
            },
            Explode(expr) => {
                let field = arena.get(*expr).to_field_impl(schema, arena, nested)?;

                if let List(inner) = field.data_type() {
                    Ok(Field::new(field.name(), *inner.clone()))
                } else {
                    Ok(field)
                }
            },
            Alias(expr, name) => Ok(Field::new(
                name,
                arena.get(*expr).to_field_impl(schema, arena, nested)?.dtype,
            )),
            Column(name) => schema
                .get_field(name)
                .ok_or_else(|| PolarsError::ColumnNotFound(name.to_string().into())),
            Literal(sv) => {
                *nested = 0;
                Ok(match sv {
                    LiteralValue::Series(s) => s.field().into_owned(),
                    _ => Field::new(sv.output_name(), sv.get_datatype()),
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
                            out_field = arena.get(*left).to_field_impl(schema, arena, nested)?;
                            out_field.name().as_str()
                        };
                        Field::new(out_name, Boolean)
                    },
                    Operator::TrueDivide => {
                        return get_truediv_field(*left, *right, arena, schema, nested)
                    },
                    _ => return get_arithmetic_field(*left, *right, arena, *op, schema, nested),
                };

                Ok(field)
            },
            Sort { expr, .. } => arena.get(*expr).to_field_impl(schema, arena, nested),
            Gather {
                expr,
                returns_scalar,
                ..
            } => {
                if *returns_scalar {
                    *nested = nested.saturating_sub(1);
                }
                arena.get(*expr).to_field_impl(schema, arena, nested)
            },
            SortBy { expr, .. } => arena.get(*expr).to_field_impl(schema, arena, nested),
            Filter { input, .. } => arena.get(*input).to_field_impl(schema, arena, nested),
            Agg(agg) => {
                use IRAggExpr::*;
                match agg {
                    Max { input: expr, .. }
                    | Min { input: expr, .. }
                    | First(expr)
                    | Last(expr) => {
                        *nested = nested.saturating_sub(1);
                        arena.get(*expr).to_field_impl(schema, arena, nested)
                    },
                    Sum(expr) => {
                        *nested = nested.saturating_sub(1);
                        let mut field = arena.get(*expr).to_field_impl(schema, arena, nested)?;
                        let dt = match field.data_type() {
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
                        *nested = nested.saturating_sub(1);
                        let mut field = arena.get(*expr).to_field_impl(schema, arena, nested)?;
                        match field.dtype {
                            Date => field.coerce(Datetime(TimeUnit::Milliseconds, None)),
                            _ => float_type(&mut field),
                        }
                        Ok(field)
                    },
                    Mean(expr) => {
                        *nested = nested.saturating_sub(1);
                        let mut field = arena.get(*expr).to_field_impl(schema, arena, nested)?;
                        match field.dtype {
                            Date => field.coerce(Datetime(TimeUnit::Milliseconds, None)),
                            _ => float_type(&mut field),
                        }
                        Ok(field)
                    },
                    Implode(expr) => {
                        let mut field = arena.get(*expr).to_field_impl(schema, arena, nested)?;
                        field.coerce(DataType::List(field.data_type().clone().into()));
                        Ok(field)
                    },
                    Std(expr, _) => {
                        *nested = nested.saturating_sub(1);
                        let mut field = arena.get(*expr).to_field_impl(schema, arena, nested)?;
                        float_type(&mut field);
                        Ok(field)
                    },
                    Var(expr, _) => {
                        *nested = nested.saturating_sub(1);
                        let mut field = arena.get(*expr).to_field_impl(schema, arena, nested)?;
                        float_type(&mut field);
                        Ok(field)
                    },
                    NUnique(expr) => {
                        *nested = 0;
                        let mut field = arena.get(*expr).to_field_impl(schema, arena, nested)?;
                        field.coerce(IDX_DTYPE);
                        Ok(field)
                    },
                    Count(expr, _) => {
                        *nested = 0;
                        let mut field = arena.get(*expr).to_field_impl(schema, arena, nested)?;
                        field.coerce(IDX_DTYPE);
                        Ok(field)
                    },
                    AggGroups(expr) => {
                        *nested = 1;
                        let mut field = arena.get(*expr).to_field_impl(schema, arena, nested)?;
                        field.coerce(List(IDX_DTYPE.into()));
                        Ok(field)
                    },
                    Quantile { expr, .. } => {
                        *nested = nested.saturating_sub(1);
                        let mut field = arena.get(*expr).to_field_impl(schema, arena, nested)?;
                        float_type(&mut field);
                        Ok(field)
                    },
                }
            },
            Cast {
                expr, data_type, ..
            } => {
                let field = arena.get(*expr).to_field_impl(schema, arena, nested)?;
                Ok(Field::new(field.name(), data_type.clone()))
            },
            Ternary { truthy, falsy, .. } => {
                let mut nested_truthy = *nested;
                let mut nested_falsy = *nested;

                // During aggregation:
                // left: col(foo):              list<T>         nesting: 1
                // right; col(foo).first():     T               nesting: 0
                // col(foo) + col(foo).first() will have nesting 1 as we still maintain the groups list.
                let mut truthy =
                    arena
                        .get(*truthy)
                        .to_field_impl(schema, arena, &mut nested_truthy)?;
                let falsy = arena
                    .get(*falsy)
                    .to_field_impl(schema, arena, &mut nested_falsy)?;

                let st = if let DataType::Null = *truthy.data_type() {
                    falsy.data_type().clone()
                } else {
                    try_get_supertype(truthy.data_type(), falsy.data_type())?
                };

                *nested = std::cmp::max(nested_truthy, nested_falsy);

                truthy.coerce(st);
                Ok(truthy)
            },
            AnonymousFunction {
                output_type,
                input,
                function,
                options,
                ..
            } => {
                *nested = nested
                    .saturating_sub(options.flags.contains(FunctionFlags::RETURNS_SCALAR) as _);
                let tmp = function.get_output();
                let output_type = tmp.as_ref().unwrap_or(output_type);
                let fields = func_args_to_fields(input, schema, arena, nested)?;
                polars_ensure!(!fields.is_empty(), ComputeError: "expression: '{}' didn't get any inputs", options.fmt_str);
                output_type.get_field(schema, Context::Default, &fields)
            },
            Function {
                function,
                input,
                options,
            } => {
                *nested = nested
                    .saturating_sub(options.flags.contains(FunctionFlags::RETURNS_SCALAR) as _);
                let fields = func_args_to_fields(input, schema, arena, nested)?;
                polars_ensure!(!fields.is_empty(), ComputeError: "expression: '{}' didn't get any inputs", function);
                function.get_field(schema, Context::Default, &fields)
            },
            Slice { input, .. } => arena.get(*input).to_field_impl(schema, arena, nested),
        }
    }
}

fn func_args_to_fields(
    input: &[ExprIR],
    schema: &Schema,
    arena: &Arena<AExpr>,
    nested: &mut u8,
) -> PolarsResult<Vec<Field>> {
    input
        .iter()
        // Default context because `col()` would return a list in aggregation context
        .map(|e| {
            arena
                .get(e.node())
                .to_field_impl(schema, arena, nested)
                .map(|mut field| {
                    field.name = e.output_name().into();
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
    schema: &Schema,
    nested: &mut u8,
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
    let mut left_field = left_ae.to_field_impl(schema, arena, nested)?;

    let super_type = match op {
        Operator::Minus => {
            let right_type = right_ae.to_field_impl(schema, arena, nested)?.dtype;
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
                | (Time, Duration(_)) => try_get_supertype(left_field.data_type(), &right_type)?,
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
                (left, right) => try_get_supertype(left, right)?,
            }
        },
        Operator::Plus => {
            let right_type = right_ae.to_field_impl(schema, arena, nested)?.dtype;
            match (&left_field.dtype, &right_type) {
                (Duration(_), Datetime(_, _))
                | (Datetime(_, _), Duration(_))
                | (Duration(_), Date)
                | (Date, Duration(_))
                | (Duration(_), Time)
                | (Time, Duration(_)) => try_get_supertype(left_field.data_type(), &right_type)?,
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
                (left, right) => try_get_supertype(left, right)?,
            }
        },
        _ => {
            let right_type = right_ae.to_field_impl(schema, arena, nested)?.dtype;

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
    schema: &Schema,
    nested: &mut u8,
) -> PolarsResult<Field> {
    let mut left_field = arena.get(left).to_field_impl(schema, arena, nested)?;
    use DataType::*;
    let out_type = match left_field.data_type() {
        Float32 => Float32,
        dt if dt.is_numeric() => Float64,
        #[cfg(feature = "dtype-duration")]
        Duration(_) => match arena
            .get(right)
            .to_field_impl(schema, arena, nested)?
            .data_type()
        {
            Duration(_) => Float64,
            dt if dt.is_numeric() => return Ok(left_field),
            dt => {
                polars_bail!(InvalidOperation: "true division of {} with {} is not allowed", left_field.data_type(), dt)
            },
        },
        #[cfg(feature = "dtype-datetime")]
        Datetime(_, _) => {
            polars_bail!(InvalidOperation: "division of 'Datetime' datatype is not allowed")
        },
        #[cfg(feature = "dtype-time")]
        Time => polars_bail!(InvalidOperation: "division of 'Time' datatype is not allowed"),
        #[cfg(feature = "dtype-date")]
        Date => polars_bail!(InvalidOperation: "division of 'Date' datatype is not allowed"),
        // we don't know what to do here, best return the dtype
        dt => dt.clone(),
    };

    left_field.coerce(out_type);
    Ok(left_field)
}
