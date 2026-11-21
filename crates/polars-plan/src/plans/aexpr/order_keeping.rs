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
    /// The physical values of the column for which no step overflows, as a lower and upper bound.
    /// `None` is no bound.
    pub safe: (Option<i128>, Option<i128>),
    column_physical: DataType,
    column_dtype: DataType,
}

impl OrderKeeping {
    /// Whether the column values are all within [`OrderKeeping::safe`], given expressions of the
    /// column for its lowest and highest value (they may be one node), or `None` when every value
    /// of the column is.
    pub fn in_safe_range(
        &self,
        lowest: Node,
        highest: Node,
        arena: &mut Arena<AExpr>,
    ) -> Option<Node> {
        let physical = |node: Node, arena: &mut Arena<AExpr>| {
            let b = AExprBuilder::new_from_node(node);
            if self.column_physical == self.column_dtype {
                b
            } else {
                b.to_physical(arena)
            }
        };
        let same = highest == lowest;
        let lowest = physical(lowest, arena);
        let highest = if same {
            lowest
        } else {
            physical(highest, arena)
        };
        let mut in_range: Option<AExprBuilder> = None;
        for (bound, value, op) in [
            (self.safe.0, lowest, Operator::GtEq),
            (self.safe.1, highest, Operator::LtEq),
        ] {
            let Some(bound) = bound else { continue };
            let bound = self.bound_literal(bound, arena);
            let cmp = value.binary_op(bound, op, arena);
            in_range = Some(match in_range {
                Some(in_range) => in_range.and(cmp, arena),
                None => cmp,
            });
        }
        in_range.map(|in_range| in_range.node())
    }

    /// The literal `bound` of the physical dtype of the column.
    fn bound_literal(&self, bound: i128, arena: &mut Arena<AExpr>) -> AExprBuilder {
        let value = match &self.column_physical {
            DataType::Int8 => i8::try_from(bound).ok().map(AnyValue::Int8),
            DataType::Int16 => i16::try_from(bound).ok().map(AnyValue::Int16),
            DataType::Int32 => i32::try_from(bound).ok().map(AnyValue::Int32),
            DataType::Int64 => i64::try_from(bound).ok().map(AnyValue::Int64),
            DataType::UInt8 => u8::try_from(bound).ok().map(AnyValue::UInt8),
            DataType::UInt16 => u16::try_from(bound).ok().map(AnyValue::UInt16),
            DataType::UInt32 => u32::try_from(bound).ok().map(AnyValue::UInt32),
            DataType::UInt64 => u64::try_from(bound).ok().map(AnyValue::UInt64),
            _ => None,
        };
        match value {
            Some(value) => {
                AExprBuilder::lit_scalar(Scalar::new(self.column_physical.clone(), value), arena)
            },
            None => AExprBuilder::lit(LiteralValue::Dyn(DynLiteralValue::Int(bound)), arena)
                .cast(self.column_physical.clone(), arena),
        }
    }

    /// `expr`, this order-keeping expression of type `dtype`, giving null for a column value where
    /// a step overflows, so that it keeps the order of the values it gives non-null.
    pub fn null_on_overflow(&self, expr: Node, dtype: &DataType, arena: &mut Arena<AExpr>) -> Node {
        let column = arena.add(AExpr::Column(self.column.clone()));
        match self.in_safe_range(column, column, arena) {
            Some(in_range) => null_unless(in_range, expr, dtype, arena),
            None => expr,
        }
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

/// A range of integers, with `None` for no bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Interval {
    lo: Option<i128>,
    hi: Option<i128>,
}

/// An integer that is beyond `i128` in a direction.
#[derive(Debug, Clone, Copy)]
enum Overflow {
    Below,
    Above,
}

type Wide = Result<i128, Overflow>;

impl Interval {
    const ALL: Self = Self { lo: None, hi: None };

    /// The interval between two bounds. `None` if it is empty, which also covers a bound that
    /// isn't exact: a lower bound beyond `i128::MAX` or an upper bound below `i128::MIN`.
    fn from_wide(lo: Option<Wide>, hi: Option<Wide>) -> Option<Self> {
        let lo = match lo {
            None | Some(Err(Overflow::Below)) => None,
            Some(Ok(v)) => Some(v),
            Some(Err(Overflow::Above)) => return None,
        };
        let hi = match hi {
            None | Some(Err(Overflow::Above)) => None,
            Some(Ok(v)) => Some(v),
            Some(Err(Overflow::Below)) => return None,
        };
        Self { lo, hi }.non_empty()
    }

    fn non_empty(self) -> Option<Self> {
        match (self.lo, self.hi) {
            (Some(lo), Some(hi)) if lo > hi => None,
            _ => Some(self),
        }
    }

    fn intersect(self, other: Self) -> Option<Self> {
        let lo = match (self.lo, other.lo) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        let hi = match (self.hi, other.hi) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        Self { lo, hi }.non_empty()
    }
}

fn add(a: i128, b: i128) -> Wide {
    a.checked_add(b).ok_or(if b < 0 {
        Overflow::Below
    } else {
        Overflow::Above
    })
}

fn sub(a: i128, b: i128) -> Wide {
    a.checked_sub(b).ok_or(if b > 0 {
        Overflow::Below
    } else {
        Overflow::Above
    })
}

fn mul(a: i128, b: i128) -> Wide {
    a.checked_mul(b).ok_or(if (a < 0) == (b < 0) {
        Overflow::Above
    } else {
        Overflow::Below
    })
}

fn neg(a: i128) -> Wide {
    a.checked_neg().ok_or(Overflow::Above)
}

/// `a / b` rounded down, for `b != 0`.
fn div_floor(a: i128, b: i128) -> Wide {
    let q = a.checked_div(b).ok_or(Overflow::Above)?;
    Ok(if a % b != 0 && (a < 0) != (b < 0) {
        q - 1
    } else {
        q
    })
}

/// `a / b` rounded up, for `b != 0`.
fn div_ceil(a: i128, b: i128) -> Wide {
    let q = a.checked_div(b).ok_or(Overflow::Above)?;
    Ok(if a % b != 0 && (a < 0) == (b < 0) {
        q + 1
    } else {
        q
    })
}

/// A map of the physical values that keeps their order, or reverses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// `x + c`
    Add(i128),
    /// `-x`
    Neg,
    /// `x * c`, for `c != 0`
    Mul(i128),
    /// `x // c`, rounded down, for `c != 0`
    FloorDiv(i128),
    /// `x / d` rounded towards zero, for `d > 0`, a cast of a duration to a coarser unit
    #[cfg_attr(not(feature = "dtype-duration"), allow(dead_code))]
    TruncDiv(i128),
}

impl Step {
    /// The values the step maps into `out`. Where a bound is not exact, it is within the exact
    /// range.
    fn preimage(self, out: Interval) -> Option<Interval> {
        let (lo, hi) = (out.lo, out.hi);
        match self {
            Step::Add(c) => Interval::from_wide(lo.map(|v| sub(v, c)), hi.map(|v| sub(v, c))),
            Step::Neg => Interval::from_wide(hi.map(neg), lo.map(neg)),
            Step::Mul(c) if c > 0 => {
                Interval::from_wide(lo.map(|v| div_ceil(v, c)), hi.map(|v| div_floor(v, c)))
            },
            Step::Mul(c) => {
                Interval::from_wide(hi.map(|v| div_ceil(v, c)), lo.map(|v| div_floor(v, c)))
            },
            Step::FloorDiv(c) if c > 0 => Interval::from_wide(
                lo.map(|v| mul(v, c)),
                hi.map(|v| mul(v, c).and_then(|v| add(v, c - 1))),
            ),
            Step::FloorDiv(c) => Interval::from_wide(
                hi.map(|v| mul(v, c).and_then(|v| add(v, c + 1))),
                lo.map(|v| mul(v, c)),
            ),
            Step::TruncDiv(d) => Interval::from_wide(
                lo.map(|v| {
                    if v > 0 {
                        mul(v, d)
                    } else {
                        mul(v, d).and_then(|v| sub(v, d - 1))
                    }
                }),
                hi.map(|v| {
                    if v >= 0 {
                        mul(v, d).and_then(|v| add(v, d - 1))
                    } else {
                        mul(v, d)
                    }
                }),
            ),
        }
    }
}

/// An [`OrderKeeping`] under construction: the steps from the column outward.
#[derive(Clone)]
struct Chain {
    column: PlSmallStr,
    column_dtype: DataType,
    increasing: bool,
    steps: Vec<Step>,
    safe: Interval,
}

impl Chain {
    fn flip(mut self, flip: bool) -> Self {
        self.increasing ^= flip;
        self
    }

    /// Apply a step that does not overflow.
    fn apply(mut self, step: Option<Step>) -> Self {
        self.steps.extend(step);
        self
    }

    /// Apply `steps`, which must give values within `out` to not overflow. `None` if no value
    /// of the column is safe.
    fn apply_guarded(mut self, steps: &[Step], out: Interval) -> Option<Self> {
        let column_values = (self.steps.iter().chain(steps))
            .rev()
            .try_fold(out, |values, step| step.preimage(values))?;
        self.safe = self.safe.intersect(column_values)?;
        self.steps.extend(steps);
        Some(self)
    }

    /// Apply `steps` that end in the dtype `dtype`, and overflow outside of it.
    fn apply_in_dtype(self, steps: &[Step], dtype: &DataType) -> Option<Self> {
        let (min, max) = int_range(dtype)?;
        self.apply_guarded(
            steps,
            Interval {
                lo: Some(min),
                hi: Some(max),
            },
        )
    }

    fn finish(self) -> Option<OrderKeeping> {
        let column_physical = self.column_dtype.to_physical();
        let (min, max) = int_range(&column_physical)?;
        let lo = self.safe.lo.filter(|&lo| lo > min);
        let hi = self.safe.hi.filter(|&hi| hi < max);
        if lo.is_some_and(|lo| lo > max) || hi.is_some_and(|hi| hi < min) {
            return None;
        }
        Some(OrderKeeping {
            column: self.column,
            increasing: self.increasing,
            safe: (lo, hi),
            column_physical,
            column_dtype: self.column_dtype,
        })
    }
}

/// Whether `node` is an order-keeping expression of a single column. Null maps to null and other
/// values to non-null values, and the expression cannot raise. Integer overflow wraps, so the order
/// is only kept for the values of the column within [`OrderKeeping::safe`].
///
/// `schema` is the schema the expression is evaluated on.
pub fn order_keeping_column(
    node: Node,
    arena: &Arena<AExpr>,
    schema: &Schema,
) -> Option<OrderKeeping> {
    order_keeping_column_rec(node, arena, schema, 0)?.finish()
}

/// Deeper expressions are not order-keeping, so that the dtype checks stay cheap.
const MAX_DEPTH: usize = 16;

fn order_keeping_column_rec(
    node: Node,
    arena: &Arena<AExpr>,
    schema: &Schema,
    depth: usize,
) -> Option<Chain> {
    if depth > MAX_DEPTH {
        return None;
    }
    let ae = arena.get(node);
    let rec = |node: Node| order_keeping_column_rec(node, arena, schema, depth + 1);

    if let AExpr::Column(column) = ae {
        let column_dtype = schema.get(column)?;
        return has_ordered_values(column_dtype).then(|| Chain {
            column: column.clone(),
            column_dtype: column_dtype.clone(),
            increasing: true,
            steps: Vec::new(),
            safe: Interval::ALL,
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
            let chain = rec(*expr)?;
            Some(chain.apply(cast_step(&dtype_of(*expr)?, to)?))
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
            rec(input)?.flip(true).apply_in_dtype(&[Step::Neg], &dtype)
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
                    let steps = match op {
                        _ if subtracts_input => vec![Step::Neg, Step::Add(c)],
                        Operator::Plus => vec![Step::Add(c)],
                        _ => vec![Step::Add(c.checked_neg()?)],
                    };
                    rec(input)?
                        .flip(subtracts_input)
                        .apply_in_dtype(&steps, &dtype)
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
                    let step = if *op == Operator::Multiply {
                        Step::Mul(factor)
                    } else {
                        Step::FloorDiv(factor)
                    };
                    rec(input)?.flip(factor < 0).apply_in_dtype(&[step], &dtype)
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

/// The step that a cast, which keeps order and can't overflow, applies to the physical values, or
/// `Some(None)` if it applies none. `None` if the cast is not one of those.
fn cast_step(from: &DataType, to: &DataType) -> Option<Option<Step>> {
    match (from, to) {
        (from, to) if from.is_integer() => {
            is_lossless_upcast(from, &to.to_physical()).then_some(None)
        },
        // Days in an `i32` fit in an `i64` of milliseconds, not of microseconds.
        #[cfg(all(feature = "dtype-date", feature = "dtype-datetime"))]
        (DataType::Date, DataType::Datetime(TimeUnit::Milliseconds, _)) => {
            Some(Some(Step::Mul(86_400_000)))
        },
        #[cfg(feature = "dtype-datetime")]
        (DataType::Datetime(from_unit, from_tz), DataType::Datetime(to_unit, to_tz)) => {
            (from_tz == to_tz && !is_finer(*to_unit, *from_unit))
                .then(|| unit_divisor(*from_unit, *to_unit).map(Step::FloorDiv))
        },
        #[cfg(feature = "dtype-duration")]
        (DataType::Duration(from_unit), DataType::Duration(to_unit)) => {
            (!is_finer(*to_unit, *from_unit))
                .then(|| unit_divisor(*from_unit, *to_unit).map(Step::TruncDiv))
        },
        _ => (from == to).then_some(None),
    }
}

/// How many of `from` are in one of the coarser `to`, or `None` for the same unit.
#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
fn unit_divisor(from: TimeUnit, to: TimeUnit) -> Option<i128> {
    use TimeUnit::*;
    match (from, to) {
        (Nanoseconds, Microseconds) | (Microseconds, Milliseconds) => Some(1_000),
        (Nanoseconds, Milliseconds) => Some(1_000_000),
        _ => None,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn int(value: i128, arena: &mut Arena<AExpr>) -> AExprBuilder {
        AExprBuilder::lit(LiteralValue::Dyn(DynLiteralValue::Int(value)), arena)
    }

    /// The order-keeping analysis of the expression `build` makes of the column `x`.
    fn analyse(
        dtype: DataType,
        build: impl FnOnce(AExprBuilder, &mut Arena<AExpr>) -> AExprBuilder,
    ) -> Option<OrderKeeping> {
        let schema = Schema::from_iter([Field::new("x".into(), dtype)]);
        let mut arena = Arena::new();
        let x = AExprBuilder::col("x", &mut arena);
        let node = build(x, &mut arena).node();
        order_keeping_column(node, &arena, &schema)
    }

    fn safe(
        dtype: DataType,
        build: impl FnOnce(AExprBuilder, &mut Arena<AExpr>) -> AExprBuilder,
    ) -> (Option<i128>, Option<i128>) {
        analyse(dtype, build).unwrap().safe
    }

    const I64_MAX: i128 = i64::MAX as i128;
    const I64_MIN: i128 = i64::MIN as i128;

    #[test]
    fn test_add_guards_one_side() {
        let s = safe(DataType::Int64, |x, a| {
            let c = int(2, a);
            x.plus(c, a)
        });
        assert_eq!(s, (None, Some(I64_MAX - 2)));
        let s = safe(DataType::Int64, |x, a| {
            let c = int(2, a);
            x.minus(c, a)
        });
        assert_eq!(s, (Some(I64_MIN + 2), None));
    }

    #[test]
    fn test_chain() {
        // `x * 2` needs x in [-2^62, 2^62 - 1]; `+ 3` needs x * 2 <= MAX - 3.
        let s = safe(DataType::Int64, |x, a| {
            let (two, three) = (int(2, a), int(3, a));
            x.multiply(two, a).plus(three, a)
        });
        assert_eq!(s, (Some(-(1 << 62)), Some((1 << 62) - 2)));
    }

    #[test]
    fn test_negate_and_floor_divide_by_minus_one() {
        let s = safe(DataType::Int64, |x, a| x.negate(a));
        assert_eq!(s, (Some(I64_MIN + 1), None));
        let s = safe(DataType::Int64, |x, a| {
            let c = int(-1, a);
            x.floor_divide(c, a)
        });
        assert_eq!(s, (Some(I64_MIN + 1), None));
    }

    #[test]
    fn test_subtract_from_literal() {
        let ok = analyse(DataType::Int64, |x, a| {
            let c = int(5, a);
            c.minus(x, a)
        })
        .unwrap();
        assert!(!ok.increasing);
        assert_eq!(ok.safe, (Some(5 - I64_MAX), None));
    }

    #[test]
    fn test_int128_multiply_is_recognized() {
        let s = safe(DataType::Int128, |x, a| {
            let c = int(2, a);
            x.multiply(c, a)
        });
        assert_eq!(s, (Some(i128::MIN / 2), Some(i128::MAX / 2)));
        let s = safe(DataType::Int128, |x, a| {
            let c = int(-1, a);
            x.multiply(c, a)
        });
        assert_eq!(s, (Some(i128::MIN + 1), None));
    }

    #[test]
    fn test_no_overflow_has_no_guard() {
        let s = safe(DataType::Int32, |x, a| {
            let c = int(1, a);
            x.cast(DataType::Int64, a).plus(c, a)
        });
        assert_eq!(s, (None, None));
        let s = safe(DataType::Int64, |x, a| x.cast(DataType::Int64, a));
        assert_eq!(s, (None, None));
    }

    #[test]
    fn test_always_overflowing_is_not_order_keeping() {
        // `x + 300` can't be written in an `Int8`; three steps of 100 leave no safe value.
        let q = analyse(DataType::Int8, |x, a| {
            let c = (int(100, a), int(100, a), int(100, a));
            x.plus(c.0, a).plus(c.1, a).plus(c.2, a)
        });
        assert_eq!(q, None);
        let q = analyse(DataType::Int8, |x, a| {
            let c = (int(100, a), int(100, a));
            x.plus(c.0, a).plus(c.1, a)
        });
        assert_eq!(q.unwrap().safe, (None, Some(-73)));
    }

    #[cfg(all(
        feature = "dtype-date",
        feature = "dtype-datetime",
        feature = "dtype-duration"
    ))]
    #[test]
    fn test_date_cast_to_datetime_then_add() {
        // Days become milliseconds, so the safe milliseconds are pulled back to days.
        let days = 100;
        let c = I64_MAX - 86_400_000 * days;
        let s = safe(DataType::Date, |x, a| {
            let duration = AExprBuilder::lit_scalar(
                Scalar::new(
                    DataType::Duration(TimeUnit::Milliseconds),
                    AnyValue::Duration(c as i64, TimeUnit::Milliseconds),
                ),
                a,
            );
            x.cast(DataType::Datetime(TimeUnit::Milliseconds, None), a)
                .plus(duration, a)
        });
        assert_eq!(s, (None, Some(days)));
    }

    /// The preimage of every step is exact, checked against all values of a small range.
    #[test]
    fn test_step_preimage_is_exact() {
        let apply = |step: Step, x: i128| match step {
            Step::Add(c) => x + c,
            Step::Neg => -x,
            Step::Mul(c) => x * c,
            Step::FloorDiv(c) => {
                x.div_euclid(c) - if c < 0 && x.rem_euclid(c) != 0 { 1 } else { 0 }
            },
            Step::TruncDiv(d) => x / d,
        };
        let mut steps = vec![Step::Neg];
        for c in [-7, -3, -2, -1, 1, 2, 3, 7] {
            steps.extend([Step::Add(c), Step::Mul(c), Step::FloorDiv(c)]);
        }
        steps.extend([Step::TruncDiv(1), Step::TruncDiv(3), Step::TruncDiv(10)]);
        for step in steps {
            for lo in -12..=12 {
                for hi in lo..=12 {
                    let expected: Vec<i128> = (-100..=100)
                        .filter(|&x| (lo..=hi).contains(&apply(step, x)))
                        .collect();
                    let got = step.preimage(Interval {
                        lo: Some(lo),
                        hi: Some(hi),
                    });
                    let got: Vec<i128> = (-100..=100)
                        .filter(|&x| {
                            got.is_some_and(|g| {
                                g.lo.is_none_or(|l| x >= l) && g.hi.is_none_or(|h| x <= h)
                            })
                        })
                        .collect();
                    assert_eq!(got, expected, "{step:?} on [{lo}, {hi}]");
                }
            }
        }
    }

    #[test]
    fn test_preimage_at_the_limits_of_i128() {
        let all = Interval {
            lo: Some(i128::MIN),
            hi: Some(i128::MAX),
        };
        let half = |i: Option<Interval>| i.map(|i| (i.lo, i.hi));
        assert_eq!(
            half(Step::Mul(2).preimage(all)),
            Some((Some(i128::MIN / 2), Some(i128::MAX / 2)))
        );
        // `-MIN` doesn't exist, so `-x <= MAX` holds for all x and `-x >= MIN` for x <= MAX.
        assert_eq!(
            half(Step::Neg.preimage(all)),
            Some((Some(-i128::MAX), None))
        );
        assert_eq!(
            half(Step::FloorDiv(-1).preimage(all)),
            Some((Some(-i128::MAX), None))
        );
        // Nothing maps to a value beyond `i128::MAX`.
        let above = Interval {
            lo: Some(i128::MAX),
            hi: None,
        };
        assert_eq!(Step::Add(-1).preimage(above), None);
    }
}
