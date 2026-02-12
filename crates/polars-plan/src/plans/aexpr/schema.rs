#[cfg(feature = "dtype-decimal")]
use polars_compute::decimal::DEC128_MAX_PREC;
use polars_core::series::arithmetic::NumericListOp;
use polars_utils::format_pl_smallstr;
use recursive::recursive;

use super::*;
use crate::constants::{
    POLARS_ELEMENT, POLARS_STRUCTFIELDS, get_literal_name, get_pl_element_name,
    get_pl_structfields_name,
};

fn validate_expr(node: Node, ctx: &ToFieldContext) -> PolarsResult<()> {
    ctx.arena.get(node).to_field_impl(ctx).map(|_| ())
}

#[derive(Debug)]
pub struct ToFieldContext<'a> {
    arena: &'a Arena<AExpr>,
    schema: &'a Schema,
}

impl<'a> ToFieldContext<'a> {
    pub fn new(arena: &'a Arena<AExpr>, schema: &'a Schema) -> Self {
        Self { arena, schema }
    }
}

impl AExpr {
    pub fn to_dtype(&self, ctx: &ToFieldContext<'_>) -> PolarsResult<DataType> {
        self.to_field(ctx).map(|f| f.dtype)
    }

    /// Get Field result of the expression. The schema is the input data. The result will
    /// not be coerced (also known as auto-implode): this is the responsibility of the caller.
    pub fn to_field(&self, ctx: &ToFieldContext<'_>) -> PolarsResult<Field> {
        self.to_field_impl(ctx)
    }

    /// Get Field result of the expression. The schema is the input data.
    ///
    /// This is taken as `&mut bool` as for some expressions this is determined by the upper node
    /// (e.g. `alias`, `cast`).
    #[recursive]
    pub fn to_field_impl(&self, ctx: &ToFieldContext) -> PolarsResult<Field> {
        use AExpr::*;
        use DataType::*;
        match self {
            Element => ctx
                .schema
                .get_field(POLARS_ELEMENT)
                .ok_or_else(|| polars_err!(invalid_element_use)),

            Len => Ok(Field::new(PlSmallStr::from_static(LEN), IDX_DTYPE)),
            #[cfg(feature = "dynamic_group_by")]
            Rolling { function, .. } => {
                let e = ctx.arena.get(*function);
                let mut field = e.to_field_impl(ctx)?;
                // Implicit implode
                if !is_scalar_ae(*function, ctx.arena) {
                    field.dtype = field.dtype.implode();
                }
                Ok(field)
            },
            Over {
                function,
                partition_by,
                order_by,
                mapping,
            } => {
                for node in partition_by {
                    validate_expr(*node, ctx)?;
                }
                if let Some((node, _)) = order_by {
                    validate_expr(*node, ctx)?;
                }

                let e = ctx.arena.get(*function);
                let mut field = e.to_field_impl(ctx)?;

                if matches!(mapping, WindowMapping::Join) && !is_scalar_ae(*function, ctx.arena) {
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
            #[cfg(feature = "dtype-struct")]
            StructField(name) => {
                let struct_field = ctx
                    .schema
                    .get_field(POLARS_STRUCTFIELDS)
                    .ok_or_else(|| polars_err!(invalid_field_use))?;
                let DataType::Struct(fields) = struct_field.dtype() else {
                    return Err(polars_err!(
                        InvalidOperation: "expected `Struct` dtype for `with_fields` Expr, got `{}`", 
                        struct_field.dtype()));
                };
                // @NOTE. Linear search performance is not ideal. An alternative approach
                // would be to map each field to a new column with a temporary name (see streaming engine),
                // and extend the schema accordingly.
                for f in fields {
                    if f.name() == name {
                        return Ok(f.clone());
                    }
                }
                Err(PolarsError::StructFieldNotFound(name.to_string().into()))
            },
            Literal(sv) => Ok(match sv {
                LiteralValue::Series(s) => s.field().into_owned(),
                _ => Field::new(sv.output_column_name(), sv.get_datatype()),
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
                validate_expr(*idx, ctx)?;
                ctx.arena.get(*expr).to_field_impl(ctx)
            },
            SortBy { expr, .. } => ctx.arena.get(*expr).to_field_impl(ctx),
            Filter { input, by } => {
                validate_expr(*by, ctx)?;
                ctx.arena.get(*input).to_field_impl(ctx)
            },
            Agg(agg) => {
                use IRAggExpr::*;
                match agg {
                    Max { input: expr, .. }
                    | Min { input: expr, .. }
                    | First(expr)
                    | FirstNonNull(expr)
                    | Last(expr)
                    | LastNonNull(expr) => ctx.arena.get(*expr).to_field_impl(ctx),
                    Item { input: expr, .. } => ctx.arena.get(*expr).to_field_impl(ctx),
                    Sum(expr) => {
                        let mut field = ctx.arena.get(*expr).to_field_impl(ctx)?;
                        let dt = match field.dtype() {
                            String | Binary | BinaryOffset | List(_) => {
                                polars_bail!(
                                    InvalidOperation: "`sum` operation not supported for dtype `{}`",
                                    field.dtype()
                                )
                            },
                            #[cfg(feature = "dtype-array")]
                            Array(_, _) => {
                                polars_bail!(
                                    InvalidOperation: "`sum` operation not supported for dtype `{}`",
                                    field.dtype()
                                )
                            },
                            #[cfg(feature = "dtype-struct")]
                            Struct(_) => {
                                polars_bail!(
                                    InvalidOperation: "`sum` operation not supported for dtype `{}`",
                                    field.dtype()
                                )
                            },
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
                        let field = [ctx.arena.get(*expr).to_field_impl(ctx)?];
                        let mapper = FieldsMapper::new(&field);
                        mapper.moment_dtype()
                    },
                    Mean(expr) => {
                        let field = [ctx.arena.get(*expr).to_field_impl(ctx)?];
                        let mapper = FieldsMapper::new(&field);
                        mapper.moment_dtype()
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
                        mapper.moment_dtype()
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
            AnonymousAgg {
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
                let mut evaluation_schema = ctx.schema.clone();
                evaluation_schema.insert(get_pl_element_name(), element_dtype.clone());
                let mut output_field = ctx
                    .arena
                    .get(*evaluation)
                    .to_field_impl(&ToFieldContext::new(ctx.arena, &evaluation_schema))?;
                output_field.dtype = output_field.dtype.materialize_unknown(false)?;
                let eval_is_scalar = is_scalar_ae(*evaluation, ctx.arena);

                output_field.dtype =
                    variant.output_dtype(field.dtype(), output_field.dtype, eval_is_scalar)?;
                output_field.name = field.name;

                Ok(output_field)
            },
            #[cfg(feature = "dtype-struct")]
            StructEval { expr, evaluation } => {
                let struct_field = ctx.arena.get(*expr).to_field_impl(ctx)?;
                let mut evaluation_schema = ctx.schema.clone();
                evaluation_schema.insert(get_pl_structfields_name(), struct_field.dtype().clone());

                let eval_fields = func_args_to_fields(
                    evaluation,
                    &ToFieldContext::new(ctx.arena, &evaluation_schema),
                )?;

                // Merge evaluation fields into the expr Struct
                if let DataType::Struct(expr_fields) = struct_field.dtype() {
                    let mut fields_map =
                        PlIndexMap::with_capacity(expr_fields.len() + eval_fields.len());
                    for field in expr_fields {
                        fields_map.insert(field.name(), field.dtype());
                    }
                    for field in &eval_fields {
                        fields_map.insert(field.name(), field.dtype());
                    }
                    let dtype = DataType::Struct(
                        fields_map
                            .iter()
                            .map(|(&name, &dtype)| Field::new(name.clone(), dtype.clone()))
                            .collect(),
                    );
                    let mut out = struct_field.clone();
                    out.coerce(dtype);
                    Ok(out)
                } else {
                    let dt = struct_field.dtype();
                    polars_bail!(op = "with_fields", got = dt, expected = "Struct")
                }
            },
            Function {
                function,
                input,
                options: _,
            } => {
                #[cfg(feature = "strings")]
                {
                    if input.is_empty()
                        && matches!(
                            &function,
                            IRFunctionExpr::StringExpr(IRStringFunction::Format { .. })
                        )
                    {
                        return Ok(Field::new(get_literal_name(), DataType::String));
                    }
                }

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
                validate_expr(*offset, ctx)?;
                validate_expr(*length, ctx)?;

                ctx.arena.get(*input).to_field_impl(ctx)
            },
        }
    }

    pub fn to_name(&self, expr_arena: &Arena<AExpr>) -> PlSmallStr {
        use AExpr::*;
        use IRAggExpr::*;
        match self {
            Element => PlSmallStr::EMPTY,
            Len => crate::constants::get_len_name(),
            #[cfg(feature = "dynamic_group_by")]
            Rolling {
                function,
                index_column: _,
                period: _,
                offset: _,
                closed_window: _,
            } => expr_arena.get(*function).to_name(expr_arena),
            Over {
                function: expr,
                partition_by: _,
                order_by: _,
                mapping: _,
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
            | Agg(Min { input: expr, .. })
            | Agg(Max { input: expr, .. })
            | Agg(First(expr))
            | Agg(FirstNonNull(expr))
            | Agg(Last(expr))
            | Agg(LastNonNull(expr))
            | Agg(Item { input: expr, .. })
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
            AnonymousFunction { input, fmt_str, .. } | AnonymousAgg { input, fmt_str, .. } => {
                if input.is_empty() {
                    fmt_str.as_ref().clone()
                } else {
                    input[0].output_name().clone()
                }
            },
            #[cfg(feature = "dtype-struct")]
            StructEval { expr, .. } => expr_arena.get(*expr).to_name(expr_arena),
            Function {
                input, function, ..
            } => match function.output_name().and_then(|v| v.into_inner()) {
                Some(name) => name,
                None if input.is_empty() => format_pl_smallstr!("{}", &function),
                None => input[0].output_name().clone(),
            },
            Column(name) => name.clone(),
            #[cfg(feature = "dtype-struct")]
            StructField(name) => name.clone(),
            Literal(lv) => lv.output_column_name().clone(),
        }
    }
}

fn func_args_to_fields(input: &[ExprIR], ctx: &ToFieldContext) -> PolarsResult<Vec<Field>> {
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
    ctx: &ToFieldContext,
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
                // This matches the engine output. TODO: revisit pending resolution of GH issue #23797
                #[cfg(feature = "dtype-struct")]
                (Struct(_), r) if r.is_numeric() => {
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
                    // TODO: This should not use `try_get_supertype()`! It should instead recursively use the enclosing match block.
                    // Otherwise we will silently permit addition operations between logical types (see above).
                    // This currently doesn't cause any problems because the list arithmetic implementation checks and raises errors
                    // if the leaf types aren't numeric, but it means we don't raise an error until execution and the DSL schema
                    // may be incorrect.
                    list_dtype.cast_leaf(NumericListOp::sub().try_get_leaf_supertype(
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
                (Decimal(_, scale_left), Decimal(_, scale_right)) => {
                    Decimal(DEC128_MAX_PREC, *scale_left.max(scale_right))
                },
                (left, right) => try_get_supertype(left, right)?,
            }
        },
        Operator::Plus => {
            let right_type = right_ae.to_field_impl(ctx)?.dtype;
            match (&left_field.dtype, &right_type) {
                #[cfg(feature = "dtype-struct")]
                (Struct(_), Struct(_)) => {
                    return Ok(left_field);
                },
                // This matches the engine output. TODO: revisit pending resolution of GH issue #23797
                #[cfg(feature = "dtype-struct")]
                (Struct(_), r) if r.is_numeric() => {
                    return Ok(left_field);
                },
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
                    list_dtype.cast_leaf(NumericListOp::add().try_get_leaf_supertype(
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
                (Decimal(_, scale_left), Decimal(_, scale_right)) => {
                    Decimal(DEC128_MAX_PREC, *scale_left.max(scale_right))
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
                // This matches the engine output. TODO: revisit pending resolution of GH issue #23797
                #[cfg(feature = "dtype-struct")]
                (Struct(_), r) if r.is_numeric() => {
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
                (Decimal(_, scale_left), Decimal(_, scale_right)) => {
                    let dtype = Decimal(DEC128_MAX_PREC, *scale_left.max(scale_right));
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
                            (AExpr::Literal(_), _) if left_field.dtype.is_unknown() => {
                                // literal will be coerced to match right type
                                left_field.coerce(right_type);
                                return Ok(left_field);
                            },
                            (_, AExpr::Literal(_)) if right_type.is_unknown() => {
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

fn get_truediv_field(left: Node, right: Node, ctx: &ToFieldContext) -> PolarsResult<Field> {
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
        #[cfg(feature = "dtype-struct")]
        (Struct(a), Struct(b)) => {
            polars_ensure!(a.len() == b.len() || b.len() == 1,
                InvalidOperation: "cannot {} two structs of different length (left: {}, right: {})",
                "div", a.len(), b.len()
            );
            let mut fields = Vec::with_capacity(a.len());
            // In case b.len() == 1, we broadcast the first field (b[0]).
            // Safety is assured by the constraints above.
            let b_iter = (0..a.len()).map(|i| b.get(i.min(b.len() - 1)).unwrap());
            for (left, right) in a.iter().zip(b_iter) {
                let name = left.name.clone();
                let (left, right) = (left.dtype(), right.dtype());
                if !(left.is_numeric() && right.is_numeric()) {
                    polars_bail!(InvalidOperation:
                        "cannot {} two structs with non-numeric fields: (left: {}, right: {})",
                        "div", left, right,)
                };
                let field = Field::new(name, get_truediv_dtype(left, right)?);
                fields.push(field);
            }
            Struct(fields)
        },
        #[cfg(feature = "dtype-struct")]
        (Struct(a), n) if n.is_numeric() => {
            let mut fields = Vec::with_capacity(a.len());
            for left in a.iter() {
                let name = left.name.clone();
                let left = left.dtype();
                if !(left.is_numeric()) {
                    polars_bail!(InvalidOperation:
                        "cannot {} a struct with non-numeric field: (left: {})",
                        "div", left)
                };
                let field = Field::new(name, get_truediv_dtype(left, n)?);
                fields.push(field);
            }
            Struct(fields)
        },
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
        #[cfg(feature = "dtype-f16")]
        (Boolean, Float16) => Float16,
        (Boolean, Float32) => Float32,
        (Boolean, b) if b.is_numeric() => Float64,
        (Boolean, Boolean) => Float64,
        #[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
        (Float16, UInt8 | Int8) => Float16,
        #[cfg(all(feature = "dtype-f16", feature = "dtype-u16"))]
        (Float16, UInt16 | Int16) => Float32,
        #[cfg(feature = "dtype-f16")]
        (Float16, Unknown(UnknownKind::Int(_))) => Float16,
        #[cfg(feature = "dtype-f16")]
        (Float16, other) if other.is_integer() => Float64,
        #[cfg(feature = "dtype-f16")]
        (Float16, Float32) => Float32,
        #[cfg(feature = "dtype-f16")]
        (Float16, Float64) => Float64,
        #[cfg(feature = "dtype-f16")]
        (Float16, _) => Float16,
        #[cfg(feature = "dtype-u8")]
        (Float32, UInt8 | Int8) => Float32,
        #[cfg(feature = "dtype-u16")]
        (Float32, UInt16 | Int16) => Float32,
        (Float32, Unknown(UnknownKind::Int(_))) => Float32,
        (Float32, other) if other.is_integer() => Float64,
        (Float32, Float64) => Float64,
        (Float32, _) => Float32,
        (String, _) | (_, String) => polars_bail!(
            InvalidOperation: "division with 'String' datatypes is not allowed"
        ),
        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, scale_left), Decimal(_, scale_right)) => {
            Decimal(DEC128_MAX_PREC, *scale_left.max(scale_right))
        },
        #[cfg(all(feature = "dtype-u8", feature = "dtype-f16"))]
        (UInt8 | Int8, Float16) => Float16,
        #[cfg(all(feature = "dtype-u16", feature = "dtype-f16"))]
        (UInt16 | Int16, Float16) => Float32,
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
