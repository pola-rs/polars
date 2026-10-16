use polars_core::prelude::*;
use polars_core::utils::get_numeric_upcast_supertype_lossless;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::{AExpr, IRFunctionExpr, ToFieldContext};
use crate::dsl::Operator;
use crate::plans::{DynLiteralValue, LiteralValue};

/// An expression of one column that never puts two values in the other order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderKeeping {
    pub column: PlSmallStr,
    /// Whether `a <= b` gives `f(a) <= f(b)`. Otherwise it gives `f(a) >= f(b)`.
    pub increasing: bool,
}

impl OrderKeeping {
    fn flip(mut self, flip: bool) -> Self {
        self.increasing ^= flip;
        self
    }
}

/// Whether `node` is an order-keeping expression of a single column. Null maps to null and other
/// values to non-null values, and the expression cannot raise. Integer overflow is ignored.
///
/// `schema` is the schema the expression is evaluated on.
pub fn order_keeping_column(
    node: Node,
    arena: &Arena<AExpr>,
    schema: &Schema,
) -> Option<OrderKeeping> {
    order_keeping_column_rec(node, arena, schema, 0)
}

/// Deeper expressions are not order-keeping, so that the dtype checks stay cheap.
const MAX_DEPTH: usize = 16;

fn order_keeping_column_rec(
    node: Node,
    arena: &Arena<AExpr>,
    schema: &Schema,
    depth: usize,
) -> Option<OrderKeeping> {
    if depth > MAX_DEPTH {
        return None;
    }
    let ae = arena.get(node);
    let rec = |node: Node| order_keeping_column_rec(node, arena, schema, depth + 1);

    if let AExpr::Column(column) = ae {
        return has_ordered_values(schema.get(column)?).then(|| OrderKeeping {
            column: column.clone(),
            increasing: true,
        });
    }
    let ctx = ToFieldContext::new(arena, schema);
    let dtype_of = |node: Node| arena.get(node).to_dtype(&ctx).ok();
    let dtype = dtype_of(node)?;
    if !has_ordered_values(&dtype) {
        return None;
    }

    match ae {
        AExpr::Cast {
            expr,
            dtype: to,
            options: _,
        } => {
            let order_keeping = rec(*expr)?;
            cast_keeps_order(&dtype_of(*expr)?, to).then_some(order_keeping)
        },
        AExpr::Function {
            input,
            function: IRFunctionExpr::Negate,
            ..
        } => {
            let input = input[0].node();
            let input_dtype = dtype_of(input)?;
            let can_negate = matches!(
                input_dtype,
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            ) || input_dtype.is_duration();
            if !can_negate {
                return None;
            }
            Some(rec(input)?.flip(true))
        },
        AExpr::BinaryExpr { left, op, right } => {
            let (left, right) = (*left, *right);
            match op {
                Operator::Plus | Operator::Minus => {
                    let (input, literal) = if is_scalar_literal(right, arena) {
                        (left, right)
                    } else if is_scalar_literal(left, arena) {
                        (right, left)
                    } else {
                        return None;
                    };
                    let input_dtype = dtype_of(input)?;
                    let subtracts_input = *op == Operator::Minus && literal == left;
                    let keeps_values = if input_dtype.is_integer() {
                        is_lossless_upcast(&input_dtype, &dtype)
                            && int_literal(literal, arena).is_some_and(|v| int_fits(v, &dtype))
                    } else {
                        // Only `datetime ± duration` and `duration ± duration` of one time unit.
                        let unit = time_unit(&input_dtype);
                        let literal_dtype = dtype_of(literal)?;
                        unit.is_some()
                            && literal_dtype.is_duration()
                            && unit == time_unit(&literal_dtype)
                            && unit == time_unit(&dtype)
                            && (!subtracts_input || input_dtype.is_duration())
                    };
                    if !keeps_values {
                        return None;
                    }
                    Some(rec(input)?.flip(subtracts_input))
                },
                Operator::Multiply | Operator::FloorDivide => {
                    let (input, factor) =
                        match (int_literal(left, arena), int_literal(right, arena)) {
                            (None, Some(factor)) => (left, factor),
                            (Some(factor), None) if *op == Operator::Multiply => (right, factor),
                            _ => return None,
                        };
                    let input_dtype = dtype_of(input)?;
                    let keeps_values = factor != 0
                        && int_fits(factor, &dtype)
                        && is_lossless_upcast(&input_dtype, &dtype);
                    if !keeps_values {
                        return None;
                    }
                    Some(rec(input)?.flip(factor < 0))
                },
                _ => None,
            }
        },
        _ => None,
    }
}

/// Integers and the temporal types that are an integer underneath keep their order through
/// arithmetic. Floats don't, because of NaN.
fn has_ordered_values(dtype: &DataType) -> bool {
    match dtype {
        #[cfg(feature = "dtype-date")]
        DataType::Date => true,
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(_, _) => true,
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(_) => true,
        dt => dt.is_integer(),
    }
}

fn is_lossless_upcast(from: &DataType, to: &DataType) -> bool {
    from == to || get_numeric_upcast_supertype_lossless(from, to).as_ref() == Some(to)
}

/// Whether the integer `value` is a value of `dtype`, so a cast to it doesn't give null.
fn int_fits(value: i128, dtype: &DataType) -> bool {
    let value = match (i64::try_from(value), u64::try_from(value)) {
        (Ok(v), _) => AnyValue::Int64(v),
        (_, Ok(v)) => AnyValue::UInt64(v),
        _ => return false,
    };
    dtype.to_physical().value_within_range(value)
}

fn time_unit(dtype: &DataType) -> Option<TimeUnit> {
    match dtype {
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(unit, _) => Some(*unit),
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(unit) => Some(*unit),
        _ => None,
    }
}

/// Casts that keep order and can't overflow, so they can't raise or give null.
fn cast_keeps_order(from: &DataType, to: &DataType) -> bool {
    match (from, to) {
        (from, to) if from.is_integer() => is_lossless_upcast(from, &to.to_physical()),
        // Days in an `i32` fit in an `i64` of milliseconds, not of microseconds.
        #[cfg(all(feature = "dtype-date", feature = "dtype-datetime"))]
        (DataType::Date, DataType::Datetime(TimeUnit::Milliseconds, _)) => true,
        #[cfg(feature = "dtype-datetime")]
        (DataType::Datetime(from_unit, from_tz), DataType::Datetime(to_unit, to_tz)) => {
            from_tz == to_tz && !is_finer(*to_unit, *from_unit)
        },
        #[cfg(feature = "dtype-duration")]
        (DataType::Duration(from_unit), DataType::Duration(to_unit)) => {
            !is_finer(*to_unit, *from_unit)
        },
        _ => from == to,
    }
}

#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
fn is_finer(unit: TimeUnit, other: TimeUnit) -> bool {
    use TimeUnit::*;
    matches!(
        (unit, other),
        (Nanoseconds, Microseconds | Milliseconds) | (Microseconds, Milliseconds)
    )
}

fn is_scalar_literal(node: Node, arena: &Arena<AExpr>) -> bool {
    matches!(arena.get(node), AExpr::Literal(lv) if lv.is_scalar() && !lv.is_null())
}

/// The value of an integer scalar literal.
pub(crate) fn int_literal(node: Node, arena: &Arena<AExpr>) -> Option<i128> {
    match arena.get(node) {
        AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(v))) => Some(*v),
        AExpr::Literal(LiteralValue::Scalar(sc)) if sc.dtype().is_integer() => sc.value().extract(),
        _ => None,
    }
}
