use polars_core::prelude::*;
use polars_core::utils::get_numeric_upcast_supertype_lossless;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::{AExpr, AExprBuilder, IRFunctionExpr, ToFieldContext};
use crate::dsl::Operator;
use crate::plans::{DynLiteralValue, LiteralValue};

/// An expression of one column that never puts two values in the other order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderKeeping {
    pub column: PlSmallStr,
    /// Whether `a <= b` gives `f(a) <= f(b)`. Otherwise it gives `f(a) >= f(b)`.
    pub increasing: bool,
    /// The steps that can overflow. The expression keeps the order of the values for which none
    /// does.
    pub overflow_guards: Vec<OverflowGuard>,
}

/// The values of `input`, the input of a step that can overflow, for which it does not: those
/// between `lo` and `hi` as values of `physical`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverflowGuard {
    input: Node,
    physical: DataType,
    lo: Option<i128>,
    hi: Option<i128>,
}

impl OrderKeeping {
    fn flip(mut self, flip: bool) -> Self {
        self.increasing ^= flip;
        self
    }

    /// Require `input`, of `input_dtype`, to lie in `range` for the step that takes it to not
    /// overflow.
    fn guard(mut self, input: Node, input_dtype: &DataType, range: (i128, i128)) -> Option<Self> {
        let physical = input_dtype.to_physical();
        let (min, max) = int_range(&physical)?;
        let lo = (range.0 > min).then_some(range.0);
        let hi = (range.1 < max).then_some(range.1);
        if lo.is_some() || hi.is_some() {
            self.overflow_guards.push(OverflowGuard {
                input,
                physical,
                lo,
                hi,
            });
        }
        Some(self)
    }

    /// `expr`, this order-keeping expression of type `dtype`, giving null where a step overflows,
    /// so that it keeps the order of the values it gives non-null.
    pub fn null_on_overflow(&self, expr: Node, dtype: &DataType, arena: &mut Arena<AExpr>) -> Node {
        match self.no_overflow(arena) {
            Some(no_overflow) => null_unless(no_overflow, expr, dtype, arena),
            None => expr,
        }
    }

    /// Whether no step overflows for a value of the column, or `None` when none can.
    pub fn no_overflow(&self, arena: &mut Arena<AExpr>) -> Option<Node> {
        let mut in_range: Option<AExprBuilder> = None;
        for guard in &self.overflow_guards {
            let input = AExprBuilder::new_from_node(guard.input).to_physical(arena);
            for (bound, op) in [(guard.lo, Operator::GtEq), (guard.hi, Operator::LtEq)] {
                let Some(bound) = bound else { continue };
                let bound =
                    AExprBuilder::lit(LiteralValue::Dyn(DynLiteralValue::Int(bound)), arena)
                        .cast(guard.physical.clone(), arena);
                let cmp = input.binary_op(bound, op, arena);
                in_range = Some(match in_range {
                    Some(in_range) => in_range.and(cmp, arena),
                    None => cmp,
                });
            }
        }
        in_range.map(|in_range| in_range.node())
    }
}

/// `expr`, of type `dtype`, where `predicate` holds, and null elsewhere.
pub fn null_unless(
    predicate: Node,
    expr: Node,
    dtype: &DataType,
    arena: &mut Arena<AExpr>,
) -> Node {
    let null = AExprBuilder::lit_scalar(Scalar::new(dtype.clone(), AnyValue::Null), arena);
    AExprBuilder::new_from_node(predicate)
        .ternary(expr, null, arena)
        .node()
}

/// Whether `node` is an order-keeping expression of a single column. Null maps to null and other
/// values to non-null values, and the expression cannot raise. Integer overflow wraps, so the order
/// is only kept where none of [`OrderKeeping::overflow_guards`] is violated.
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
            overflow_guards: Vec::new(),
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
            let (min, max) = int_range(&dtype)?;
            rec(input)?.flip(true).guard(
                input,
                &input_dtype,
                (-max, min.checked_neg().unwrap_or(i128::MAX)),
            )
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
                    let c = physical_int_literal(literal, arena)?;
                    let (min, max) = int_range(&dtype)?;
                    let range = match op {
                        _ if subtracts_input => (c.saturating_sub(max), c.saturating_sub(min)),
                        Operator::Plus => (min.saturating_sub(c), max.saturating_sub(c)),
                        _ => (min.saturating_add(c), max.saturating_add(c)),
                    };
                    rec(input)?
                        .flip(subtracts_input)
                        .guard(input, &input_dtype, range)
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
                    let (min, max) = int_range(&dtype)?;
                    let range = match op {
                        Operator::Multiply if factor > 0 => {
                            (div_ceil(min, factor)?, div_floor(max, factor)?)
                        },
                        Operator::Multiply => (div_ceil(max, factor)?, div_floor(min, factor)?),
                        // Only `MIN // -1` overflows.
                        _ if factor == -1 => (-max, min.checked_neg().unwrap_or(i128::MAX)),
                        _ => (min, max),
                    };
                    rec(input)?
                        .flip(factor < 0)
                        .guard(input, &input_dtype, range)
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

/// The range of the physical integer values of `dtype`, with `UInt128` cut off at `i128::MAX`.
fn int_range(dtype: &DataType) -> Option<(i128, i128)> {
    use DataType::*;
    Some(match dtype.to_physical() {
        Int8 => (i8::MIN.into(), i8::MAX.into()),
        Int16 => (i16::MIN.into(), i16::MAX.into()),
        Int32 => (i32::MIN.into(), i32::MAX.into()),
        Int64 => (i64::MIN.into(), i64::MAX.into()),
        Int128 => (i128::MIN, i128::MAX),
        UInt8 => (0, u8::MAX.into()),
        UInt16 => (0, u16::MAX.into()),
        UInt32 => (0, u32::MAX.into()),
        UInt64 => (0, u64::MAX.into()),
        UInt128 => (0, i128::MAX),
        _ => return None,
    })
}

/// `a / b` rounded down, or `None` when it overflows.
fn div_floor(a: i128, b: i128) -> Option<i128> {
    let q = a.checked_div(b)?;
    Some(if q * b != a && (a < 0) != (b < 0) {
        q - 1
    } else {
        q
    })
}

/// `a / b` rounded up, or `None` when it overflows.
fn div_ceil(a: i128, b: i128) -> Option<i128> {
    div_floor(a.checked_neg()?, b)?.checked_neg()
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

/// The physical value of an integer or temporal scalar literal.
fn physical_int_literal(node: Node, arena: &Arena<AExpr>) -> Option<i128> {
    match arena.get(node) {
        AExpr::Literal(LiteralValue::Scalar(sc)) => sc.clone().to_physical().value().extract(),
        _ => int_literal(node, arena),
    }
}

/// The value of an integer scalar literal.
pub(crate) fn int_literal(node: Node, arena: &Arena<AExpr>) -> Option<i128> {
    match arena.get(node) {
        AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(v))) => Some(*v),
        AExpr::Literal(LiteralValue::Scalar(sc)) if sc.dtype().is_integer() => sc.value().extract(),
        _ => None,
    }
}
