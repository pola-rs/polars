#[cfg(feature = "dtype-decimal")]
use polars_core::chunked_array::arithmetic::{
    _get_decimal_scale_add_sub, _get_decimal_scale_div, _get_decimal_scale_mul,
};
use polars_utils::format_pl_smallstr;
use recursive::recursive;

use super::*;

fn validate_expr(node: Node, arena: &Arena<AExpr>, schema: &Schema) -> PolarsResult<()> {
    let mut ctx = ToFieldContext {
        schema,
        arena,
        validate: true,
    };
    arena.get(node).to_field_impl(&mut ctx).map(|_| ())
}

struct ToFieldContext<'a> {
    schema: &'a Schema,
    arena: &'a Arena<AExpr>,
    // Traverse all expressions to validate they are in the schema.
    validate: bool,
}

impl AExpr {
    pub fn to_dtype(&self, schema: &Schema, arena: &Arena<AExpr>) -> PolarsResult<DataType> {
        self.to_field(schema, arena).map(|f| f.dtype)
    }

    /// Get Field result of the expression. The schema is the input data. The provided
    /// context will be used to coerce the type into a List if needed, also known as auto-implode.
    pub fn to_field_with_ctx(
        &self,
        schema: &Schema,
        ctx: Context,
        arena: &Arena<AExpr>,
    ) -> PolarsResult<Field> {
        // Indicates whether we should auto-implode the result. This is initialized to true if we are
        // in an aggregation context, so functions that return scalars should explicitly set this
        // to false in `to_field_impl`.
        let agg_list = matches!(ctx, Context::Aggregation);
        let mut ctx = ToFieldContext {
            schema,
            arena,
            validate: true,
        };
        let mut field = self.to_field_impl(&mut ctx)?;

        if agg_list {
            if !self.is_scalar(arena) {
                field.coerce(field.dtype().clone().implode());
            }
        }

        Ok(field)
    }

    /// Get Field result of the expression. The schema is the input data. The result will
    /// not be coerced (also known as auto-implode): this is the responsibility of the caller.
    pub fn to_field(&self, schema: &Schema, arena: &Arena<AExpr>) -> PolarsResult<Field> {
        let mut ctx = ToFieldContext {
            schema,
            arena,
            validate: true,
        };

        let field = self.to_field_impl(&mut ctx)?;

        Ok(field)
    }

    /// Get Field result of the expression. The schema is the input data.
    ///
    /// This is taken as `&mut bool` as for some expressions this is determined by the upper node
    /// (e.g. `alias`, `cast`).
    #[recursive]
    pub fn to_field_impl(&self, ctx: &mut ToFieldContext) -> PolarsResult<Field> {
        use AExpr::*;
        use DataType::*;
        match self {
            Len => Ok(Field::new(PlSmallStr::from_static(LEN), IDX_DTYPE)),
            Window {
                function,
                options,
                partition_by,
                order_by,
            } => {
                if ctx.validate {
                    for node in partition_by {
                        validate_expr(*node, ctx.arena, ctx.schema)?;
                    }
                    if let Some((node, _)) = order_by {
                        validate_expr(*node, ctx.arena, ctx.schema)?;
                    }
                }

                let e = ctx.arena.get(*function);
                let mut field = e.to_field_impl(ctx)?;

                let mut implicit_implode = false;

                implicit_implode |= matches!(options, WindowType::Over(WindowMapping::Join));
                #[cfg(feature = "dynamic_group_by")]
                {
                    implicit_implode |= matches!(options, WindowType::Rolling(_));
                }

                if implicit_implode && !is_scalar_ae(*function, ctx.arena) {
                    field.dtype = field.dtype.implode();
                }

                Ok(field)
            },
            Explode { expr, .. } => {
                let field = ctx.arena.get(*expr).to_field_impl(ctx)?;
                let field = match field.dtype() {
                    List(inner) => Field::new(field.name().clone(), *inner.clone()),
                    #[cfg(feature = "dtype-array")]
                    Array(inner, ..) => Field::new(field.name().clone(), *inner.clone()),
                    _ => field,
                };

                Ok(field)
            },
            Column(name) => ctx
                .schema
                .get_field(name)
                .ok_or_else(|| PolarsError::ColumnNotFound(name.to_string().into())),
            Literal(sv) => Ok(match sv {
                LiteralValue::Series(s) => s.field().into_owned(),
                _ => Field::new(sv.output_column_name().clone(), sv.get_datatype()),
            }),
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
                            out_field = ctx.arena.get(*left).to_field_impl(ctx)?;
                            out_field.name()
                        };
                        Field::new(out_name.clone(), Boolean)
                    },
                    Operator::TrueDivide => get_truediv_field(*left, *right, ctx)?,
                    _ => get_arithmetic_field(*left, *right, *op, ctx)?,
                };

                Ok(field)
            },
            Sort { expr, .. } => ctx.arena.get(*expr).to_field_impl(ctx),
            Gather { expr, idx, .. } => {
                if ctx.validate {
                    validate_expr(*idx, ctx.arena, ctx.schema)?
                }
                ctx.arena.get(*expr).to_field_impl(ctx)
            },
            SortBy { expr, .. } => ctx.arena.get(*expr).to_field_impl(ctx),
            Filter { input, by } => {
                if ctx.validate {
                    validate_expr(*by, ctx.arena, ctx.schema)?
                }
                ctx.arena.get(*input).to_field_impl(ctx)
            },
            Agg(agg) => {
                use IRAggExpr::*;
                match agg {
                    Max { input: expr, .. }
                    | Min { input: expr, .. }
                    | First(expr)
                    | Last(expr) => ctx.arena.get(*expr).to_field_impl(ctx),
                    Sum(expr) => {
                        let mut field = ctx.arena.get(*expr).to_field_impl(ctx)?;
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
                        let mut field = ctx.arena.get(*expr).to_field_impl(ctx)?;
                        match field.dtype {
                            Date => field.coerce(Datetime(TimeUnit::Microseconds, None)),
                            _ => {
                                let field = [ctx.arena.get(*expr).to_field_impl(ctx)?];
                                let mapper = FieldsMapper::new(&field);
                                return mapper.moment_dtype();
                            },
                        }
                        Ok(field)
                    },
                    Mean(expr) => {
                        let mut field = ctx.arena.get(*expr).to_field_impl(ctx)?;
                        match field.dtype {
                            Date => field.coerce(Datetime(TimeUnit::Microseconds, None)),
                            _ => {
                                let field = [ctx.arena.get(*expr).to_field_impl(ctx)?];
                                let mapper = FieldsMapper::new(&field);
                                return mapper.moment_dtype();
                            },
                        }
                        Ok(field)
                    },
                    Implode(expr) => {
                        let mut field = ctx.arena.get(*expr).to_field_impl(ctx)?;
                        field.coerce(DataType::List(field.dtype().clone().into()));
                        Ok(field)
                    },
                    Std(expr, _) => {
                        let field = [ctx.arena.get(*expr).to_field_impl(ctx)?];
                        let mapper = FieldsMapper::new(&field);
                        mapper.moment_dtype()
                    },
                    Var(expr, _) => {
                        let field = [ctx.arena.get(*expr).to_field_impl(ctx)?];
                        let mapper = FieldsMapper::new(&field);
                        mapper.var_dtype()
                    },
                    NUnique(expr) => {
                        let mut field = ctx.arena.get(*expr).to_field_impl(ctx)?;
                        field.coerce(IDX_DTYPE);
                        Ok(field)
                    },
                    Count { input, .. } => {
                        let mut field = ctx.arena.get(*input).to_field_impl(ctx)?;
                        field.coerce(IDX_DTYPE);
                        Ok(field)
                    },
                    AggGroups(expr) => {
                        let mut field = ctx.arena.get(*expr).to_field_impl(ctx)?;
                        field.coerce(IDX_DTYPE.implode());
                        Ok(field)
                    },
                    Quantile { expr, .. } => {
                        let field = [ctx.arena.get(*expr).to_field_impl(ctx)?];
                        let mapper = FieldsMapper::new(&field);
                        mapper.map_numeric_to_float_dtype(true)
                    },
                }
            },
            Cast { expr, dtype, .. } => {
                let field = ctx.arena.get(*expr).to_field_impl(ctx)?;
                Ok(Field::new(field.name().clone(), dtype.clone()))
            },
            Ternary { truthy, falsy, .. } => {
                // During aggregation:
                // left: col(foo):              list<T>         nesting: 1
                // right; col(foo).first():     T               nesting: 0
                // col(foo) + col(foo).first() will have nesting 1 as we still maintain the groups list.
                let mut truthy = ctx.arena.get(*truthy).to_field_impl(ctx)?;
                let falsy = ctx.arena.get(*falsy).to_field_impl(ctx)?;

                let st = if let DataType::Null = *truthy.dtype() {
                    falsy.dtype().clone()
                } else {
                    try_get_supertype(truthy.dtype(), falsy.dtype())?
                };

                truthy.coerce(st);
                Ok(truthy)
            },
            AnonymousFunction {
                input,
                function,
                fmt_str,
                ..
            } => {
                let fields = func_args_to_fields(input, ctx)?;
                polars_ensure!(!fields.is_empty(), ComputeError: "expression: '{}' didn't get any inputs", fmt_str);
                let function = function.clone().materialize()?;
                let out = function.get_field(ctx.schema, &fields)?;
                Ok(out)
            },
            Eval {
                expr,
                evaluation,
                variant,
            } => {
                let field = ctx.arena.get(*expr).to_field_impl(ctx)?;

                let element_dtype = variant.element_dtype(field.dtype())?;
                let schema = Schema::from_iter([(PlSmallStr::EMPTY, element_dtype.clone())]);

                let mut ctx = ToFieldContext {
                    schema: &schema,
                    arena: ctx.arena,
                    validate: ctx.validate,
                };
                let mut output_field = ctx.arena.get(*evaluation).to_field_impl(&mut ctx)?;
                output_field.dtype = output_field.dtype.materialize_unknown(false)?;

                output_field.dtype = match variant {
                    EvalVariant::List => DataType::List(Box::new(output_field.dtype)),
                    EvalVariant::Cumulative { .. } => output_field.dtype,
                };
                output_field.name = field.name;

                Ok(output_field)
            },
            Function {
                function,
                input,
                options: _,
            } => {
                let fields = func_args_to_fields(input, ctx)?;
                polars_ensure!(!fields.is_empty(), ComputeError: "expression: '{}' didn't get any inputs", function);
                let out = function.get_field(ctx.schema, &fields)?;

                Ok(out)
            },
            Slice {
                input,
                offset,
                length,
            } => {
                if ctx.validate {
                    validate_expr(*offset, ctx.arena, ctx.schema)?;
                    validate_expr(*length, ctx.arena, ctx.schema)?;
                }

                ctx.arena.get(*input).to_field_impl(ctx)
            },
        }
    }

    pub fn to_name(&self, expr_arena: &Arena<AExpr>) -> PlSmallStr {
        use AExpr::*;
        use IRAggExpr::*;
        match self {
            Len => crate::constants::get_len_name(),
            Window {
                function: expr,
                options: _,
                partition_by: _,
                order_by: _,
            }
            | BinaryExpr { left: expr, .. }
            | Explode { expr, .. }
            | Sort { expr, .. }
            | Gather { expr, .. }
            | SortBy { expr, .. }
            | Filter { input: expr, .. }
            | Cast { expr, .. }
            | Ternary { truthy: expr, .. }
            | Eval { expr, .. }
            | Slice { input: expr, .. }
            | Agg(Max { input: expr, .. })
            | Agg(Min { input: expr, .. })
            | Agg(First(expr))
            | Agg(Last(expr))
            | Agg(Sum(expr))
            | Agg(Median(expr))
            | Agg(Mean(expr))
            | Agg(Implode(expr))
            | Agg(Std(expr, _))
            | Agg(Var(expr, _))
            | Agg(NUnique(expr))
            | Agg(Count { input: expr, .. })
            | Agg(AggGroups(expr))
            | Agg(Quantile { expr, .. }) => expr_arena.get(*expr).to_name(expr_arena),
            AnonymousFunction { input, fmt_str, .. } => {
                if input.is_empty() {
                    fmt_str.as_ref().clone()
                } else {
                    input[0].output_name().clone()
                }
            },
            Function {
                input, function, ..
            } => match function.output_name().and_then(|v| v.into_inner()) {
                Some(name) => name,
                None if input.is_empty() => format_pl_smallstr!("{}", &function),
                None => input[0].output_name().clone(),
            },
            Column(name) => name.clone(),
            Literal(lv) => lv.output_column_name().clone(),
        }
    }
}

fn func_args_to_fields(input: &[ExprIR], ctx: &mut ToFieldContext) -> PolarsResult<Vec<Field>> {
    input
        .iter()
        .map(|e| {
            ctx.arena.get(e.node()).to_field_impl(ctx).map(|mut field| {
                field.name = e.output_name().clone();
                field
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn get_arithmetic_field(
    left: Node,
    right: Node,
    op: Operator,
    ctx: &mut ToFieldContext,
) -> PolarsResult<Field> {
    use DataType::*;
    let left_ae = ctx.arena.get(left);
    let right_ae = ctx.arena.get(right);

    // don't traverse tree until strictly needed. Can have terrible performance.
    // # 3210

    // take the left field as a whole.
    // don't take dtype and name separate as that splits the tree every node
    // leading to quadratic behavior. # 4736
    //
    // further right_type is only determined when needed.
    let mut left_field = left_ae.to_field_impl(ctx)?;

    let super_type = match op {
        Operator::Minus => {
            let right_type = right_ae.to_field_impl(ctx)?.dtype;
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
                (Date, Date) => Duration(TimeUnit::Microseconds),
                (_, Date) | (Date, _) => {
                    polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                },
                (Duration(tul), Duration(tur)) => Duration(get_time_units(tul, tur)),
                (_, Duration(_)) | (Duration(_), _) => {
                    polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                },
                (Time, Time) => Duration(TimeUnit::Nanoseconds),
                (_, Time) | (Time, _) => {
                    polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                },
                (l @ List(a), r @ List(b))
                    if ![a, b]
                        .into_iter()
                        .all(|x| x.is_supported_list_arithmetic_input()) =>
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
                #[cfg(feature = "dtype-array")]
                (list_dtype @ Array(..), other_dtype) | (other_dtype, list_dtype @ Array(..)) => {
                    list_dtype.cast_leaf(try_get_supertype(
                        list_dtype.leaf_dtype(),
                        other_dtype.leaf_dtype(),
                    )?)
                },
                #[cfg(feature = "dtype-decimal")]
                (Decimal(_, Some(scale_left)), Decimal(_, Some(scale_right))) => {
                    let scale = _get_decimal_scale_add_sub(*scale_left, *scale_right);
                    Decimal(None, Some(scale))
                },
                (left, right) => try_get_supertype(left, right)?,
            }
        },
        Operator::Plus => {
            let right_type = right_ae.to_field_impl(ctx)?.dtype;
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
                        .all(|x| x.is_supported_list_arithmetic_input()) =>
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
                #[cfg(feature = "dtype-array")]
                (list_dtype @ Array(..), other_dtype) | (other_dtype, list_dtype @ Array(..)) => {
                    list_dtype.cast_leaf(try_get_supertype(
                        list_dtype.leaf_dtype(),
                        other_dtype.leaf_dtype(),
                    )?)
                },
                #[cfg(feature = "dtype-decimal")]
                (Decimal(_, Some(scale_left)), Decimal(_, Some(scale_right))) => {
                    let scale = _get_decimal_scale_add_sub(*scale_left, *scale_right);
                    Decimal(None, Some(scale))
                },
                (left, right) => try_get_supertype(left, right)?,
            }
        },
        _ => {
            let right_type = right_ae.to_field_impl(ctx)?.dtype;

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
                (l, Duration(_)) if l.is_primitive_numeric() => match op {
                    Operator::Multiply => {
                        left_field.coerce(right_type);
                        return Ok(left_field);
                    },
                    _ => {
                        polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                    },
                },
                (Duration(_), r) if r.is_primitive_numeric() => match op {
                    Operator::Multiply => {
                        return Ok(left_field);
                    },
                    _ => {
                        polars_bail!(InvalidOperation: "{} not allowed on {} and {}", op, left_field.dtype, right_type)
                    },
                },
                #[cfg(feature = "dtype-decimal")]
                (Decimal(_, Some(scale_left)), Decimal(_, Some(scale_right))) => {
                    let scale = match op {
                        Operator::Multiply => _get_decimal_scale_mul(*scale_left, *scale_right),
                        Operator::Divide | Operator::TrueDivide => {
                            _get_decimal_scale_div(*scale_left)
                        },
                        _ => {
                            debug_assert!(false);
                            *scale_left
                        },
                    };
                    let dtype = Decimal(None, Some(scale));
                    left_field.coerce(dtype);
                    return Ok(left_field);
                },

                (l @ List(a), r @ List(b))
                    if ![a, b]
                        .into_iter()
                        .all(|x| x.is_supported_list_arithmetic_input()) =>
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
                #[cfg(feature = "dtype-array")]
                (list_dtype @ Array(..), other_dtype) | (other_dtype, list_dtype @ Array(..)) => {
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

fn get_truediv_field(left: Node, right: Node, ctx: &mut ToFieldContext) -> PolarsResult<Field> {
    let mut left_field = ctx.arena.get(left).to_field_impl(ctx)?;
    let right_field = ctx.arena.get(right).to_field_impl(ctx)?;
    let out_type = get_truediv_dtype(left_field.dtype(), right_field.dtype())?;
    left_field.coerce(out_type);
    Ok(left_field)
}

fn get_truediv_dtype(left_dtype: &DataType, right_dtype: &DataType) -> PolarsResult<DataType> {
    use DataType::*;

    // TODO: Re-investigate this. A lot of "_" is being used on the RHS match because this code
    // originally (mostly) only looked at the LHS dtype.
    let out_type = match (left_dtype, right_dtype) {
        (l @ List(a), r @ List(b))
            if ![a, b]
                .into_iter()
                .all(|x| x.is_supported_list_arithmetic_input()) =>
        {
            polars_bail!(
                InvalidOperation:
                "cannot {} two list columns with non-numeric inner types: (left: {}, right: {})",
                "div", l, r,
            )
        },
        (list_dtype @ List(_), other_dtype) | (other_dtype, list_dtype @ List(_)) => {
            let dtype = get_truediv_dtype(list_dtype.leaf_dtype(), other_dtype.leaf_dtype())?;
            list_dtype.cast_leaf(dtype)
        },
        #[cfg(feature = "dtype-array")]
        (list_dtype @ Array(..), other_dtype) | (other_dtype, list_dtype @ Array(..)) => {
            let dtype = get_truediv_dtype(list_dtype.leaf_dtype(), other_dtype.leaf_dtype())?;
            list_dtype.cast_leaf(dtype)
        },
        (Boolean, Float32) => Float32,
        (Boolean, b) if b.is_numeric() => Float64,
        (Boolean, Boolean) => Float64,
        #[cfg(feature = "dtype-u8")]
        (Float32, UInt8 | Int8) => Float32,
        #[cfg(feature = "dtype-u16")]
        (Float32, UInt16 | Int16) => Float32,
        (Float32, other) if other.is_integer() => Float64,
        (Float32, Float64) => Float64,
        (Float32, _) => Float32,
        (String, _) | (_, String) => polars_bail!(
            InvalidOperation: "division with 'String' datatypes is not allowed"
        ),
        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, Some(scale_left)), Decimal(_, _)) => {
            let scale = _get_decimal_scale_div(*scale_left);
            Decimal(None, Some(scale))
        },
        #[cfg(feature = "dtype-u8")]
        (UInt8 | Int8, Float32) => Float32,
        #[cfg(feature = "dtype-u16")]
        (UInt16 | Int16, Float32) => Float32,
        (dt, _) if dt.is_primitive_numeric() => Float64,
        #[cfg(feature = "dtype-duration")]
        (Duration(_), Duration(_)) => Float64,
        #[cfg(feature = "dtype-duration")]
        (Duration(_), dt) if dt.is_primitive_numeric() => left_dtype.clone(),
        #[cfg(feature = "dtype-duration")]
        (Duration(_), dt) => {
            polars_bail!(InvalidOperation: "true division of {} with {} is not allowed", left_dtype, dt)
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
    Ok(out_type)
}
