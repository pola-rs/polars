use super::*;

impl AExpr {
    /// Get Field result of the expression. The schema is the input data.
    pub fn to_field(
        &self,
        schema: &Schema,
        ctxt: Context,
        arena: &Arena<AExpr>,
    ) -> PolarsResult<Field> {
        use AExpr::*;
        use DataType::*;
        match self {
            Count => Ok(Field::new(COUNT, IDX_DTYPE)),
            Window { function, .. } => {
                let e = arena.get(*function);
                e.to_field(schema, ctxt, arena)
            }
            Explode(expr) => {
                let field = arena.get(*expr).to_field(schema, ctxt, arena)?;

                if let DataType::List(inner) = field.data_type() {
                    Ok(Field::new(field.name(), *inner.clone()))
                } else {
                    Ok(field)
                }
            }
            Alias(expr, name) => Ok(Field::new(
                name,
                arena.get(*expr).get_type(schema, ctxt, arena)?,
            )),
            Column(name) => {
                let field = schema
                    .get_field(name)
                    .ok_or_else(|| PolarsError::ColumnNotFound(name.to_string().into()));

                match ctxt {
                    Context::Default => field,
                    Context::Aggregation => field.map(|mut field| {
                        let dtype = DataType::List(Box::new(field.data_type().clone()));
                        field.coerce(dtype);
                        field
                    }),
                }
            }
            Literal(sv) => Ok(match sv {
                LiteralValue::Series(s) => s.field().into_owned(),
                _ => Field::new("literal", sv.get_datatype()),
            }),
            BinaryExpr { left, right, op } => {
                use DataType::*;

                let field = match op {
                    Operator::Lt
                    | Operator::Gt
                    | Operator::Eq
                    | Operator::NotEq
                    | Operator::And
                    | Operator::LtEq
                    | Operator::GtEq
                    | Operator::Or => {
                        let out_field;
                        let out_name = {
                            out_field = arena.get(*left).to_field(schema, ctxt, arena)?;
                            out_field.name().as_str()
                        };
                        Field::new(out_name, Boolean)
                    }
                    Operator::TrueDivide => return get_truediv_field(*left, arena, ctxt, schema),
                    _ => return get_arithmetic_field(*left, *right, arena, *op, ctxt, schema),
                };

                Ok(field)
            }
            Sort { expr, .. } => arena.get(*expr).to_field(schema, ctxt, arena),
            Take { expr, .. } => arena.get(*expr).to_field(schema, ctxt, arena),
            SortBy { expr, .. } => arena.get(*expr).to_field(schema, ctxt, arena),
            Filter { input, .. } => arena.get(*input).to_field(schema, ctxt, arena),
            Agg(agg) => {
                use AAggExpr::*;
                match agg {
                    Max { input: expr, .. }
                    | Min { input: expr, .. }
                    | First(expr)
                    | Last(expr) => {
                        // default context because `col()` would return a list in aggregation context
                        arena.get(*expr).to_field(schema, Context::Default, arena)
                    }
                    Sum(expr) => {
                        let mut field =
                            arena.get(*expr).to_field(schema, Context::Default, arena)?;
                        if matches!(field.data_type(), UInt8 | Int8 | Int16 | UInt16) {
                            field.coerce(DataType::Int64);
                        }
                        Ok(field)
                    }
                    Median(expr) => {
                        let mut field =
                            arena.get(*expr).to_field(schema, Context::Default, arena)?;
                        if field.data_type() != &DataType::Utf8 {
                            field.coerce(DataType::Float64);
                        }
                        Ok(field)
                    }
                    Mean(expr) => {
                        let mut field =
                            arena.get(*expr).to_field(schema, Context::Default, arena)?;
                        coerce_numeric_aggregation(&mut field);
                        Ok(field)
                    }
                    List(expr) => {
                        // default context because `col()` would return a list in aggregation context
                        let mut field =
                            arena.get(*expr).to_field(schema, Context::Default, arena)?;
                        field.coerce(DataType::List(field.data_type().clone().into()));
                        Ok(field)
                    }
                    Std(expr, _) => {
                        let mut field =
                            arena.get(*expr).to_field(schema, Context::Default, arena)?;
                        coerce_numeric_aggregation(&mut field);
                        Ok(field)
                    }
                    Var(expr, _) => {
                        let mut field =
                            arena.get(*expr).to_field(schema, Context::Default, arena)?;
                        coerce_numeric_aggregation(&mut field);
                        Ok(field)
                    }
                    NUnique(expr) => {
                        let mut field =
                            arena.get(*expr).to_field(schema, Context::Default, arena)?;
                        field.coerce(DataType::UInt32);
                        Ok(field)
                    }
                    Count(expr) => {
                        let mut field =
                            arena.get(*expr).to_field(schema, Context::Default, arena)?;
                        field.coerce(IDX_DTYPE);
                        Ok(field)
                    }
                    AggGroups(expr) => {
                        let mut field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        field.coerce(DataType::List(IDX_DTYPE.into()));
                        Ok(field)
                    }
                    Quantile { expr, .. } => {
                        let mut field =
                            arena.get(*expr).to_field(schema, Context::Default, arena)?;
                        coerce_numeric_aggregation(&mut field);
                        Ok(field)
                    }
                }
            }
            Cast {
                expr, data_type, ..
            } => {
                let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                Ok(Field::new(field.name(), data_type.clone()))
            }
            Ternary { truthy, falsy, .. } => {
                let mut truthy = arena.get(*truthy).to_field(schema, ctxt, arena)?;
                let falsy = arena.get(*falsy).to_field(schema, ctxt, arena)?;
                if let DataType::Null = *truthy.data_type() {
                    truthy.coerce(falsy.data_type().clone());
                    Ok(truthy)
                } else {
                    let st = try_get_supertype(truthy.data_type(), falsy.data_type())?;
                    truthy.coerce(st);
                    Ok(truthy)
                }
            }
            AnonymousFunction {
                output_type, input, ..
            } => {
                let fields = input
                    .iter()
                    // default context because `col()` would return a list in aggregation context
                    .map(|node| arena.get(*node).to_field(schema, Context::Default, arena))
                    .collect::<PolarsResult<Vec<_>>>()?;
                Ok(output_type.get_field(schema, ctxt, &fields))
            }
            Function {
                function, input, ..
            } => {
                let fields = input
                    .iter()
                    // default context because `col()` would return a list in aggregation context
                    .map(|node| arena.get(*node).to_field(schema, Context::Default, arena))
                    .collect::<PolarsResult<Vec<_>>>()?;
                function.get_field(schema, ctxt, &fields)
            }
            Slice { input, .. } => arena.get(*input).to_field(schema, ctxt, arena),
            Wildcard => panic!("should be no wildcard at this point"),
            Nth(_) => panic!("should be no nth at this point"),
        }
    }
}

fn get_arithmetic_field(
    left: Node,
    right: Node,
    arena: &Arena<AExpr>,
    op: Operator,
    ctxt: Context,
    schema: &Schema,
) -> PolarsResult<Field> {
    // don't traverse tree until strictly needed. Can have terrible performance.
    // # 3210

    // take the left field as a whole.
    // don't take dtype and name separate as that splits the tree every node
    // leading to quadratic behavior. # 4736
    use DataType::*;
    let mut left_field = arena.get(left).to_field(schema, ctxt, arena)?;
    let right_type = arena.get(right).get_type(schema, ctxt, arena)?;

    let super_type = match op {
        Operator::Minus => match (&left_field.dtype, right_type) {
            // T - T != T if T is a datetime / date
            (Datetime(tul, _), Datetime(tur, _)) => Duration(get_time_units(tul, &tur)),
            (Date, Date) => Duration(TimeUnit::Milliseconds),
            (left, right) => try_get_supertype(left, &right)?,
        },
        _ => try_get_supertype(&left_field.dtype, &right_type)?,
    };
    left_field.coerce(super_type);
    Ok(left_field)
}

fn coerce_numeric_aggregation(field: &mut Field) {
    if field.dtype.is_numeric() && !matches!(&field.dtype, DataType::Float32) {
        field.coerce(DataType::Float64)
    }
}

fn get_truediv_field(
    left: Node,
    arena: &Arena<AExpr>,
    ctxt: Context,
    schema: &Schema,
) -> PolarsResult<Field> {
    let mut left_field = arena.get(left).to_field(schema, ctxt, arena)?;
    use DataType::*;
    let out_type = match left_field.data_type() {
        Float32 => Float32,
        dt if dt.is_numeric() => Float64,
        #[cfg(feature = "dtype-duration")]
        Duration(_) => Float64,
        // we don't know what to do here, best return the dtype
        dt => dt.clone(),
    };

    left_field.coerce(out_type);
    Ok(left_field)
}
