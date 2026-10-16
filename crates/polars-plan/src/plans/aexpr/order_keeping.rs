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

    let order_keeping = match ae {
        AExpr::Column(column) => {
            return has_ordered_values(schema.get(column)?).then(|| OrderKeeping {
                column: column.clone(),
                increasing: true,
            });
        },
        AExpr::Cast {
            expr,
            dtype,
            options: _,
        } => {
            let order_keeping = rec(*expr)?;
            let input_dtype = arena
                .get(*expr)
                .to_dtype(&ToFieldContext::new(arena, schema))
                .ok()?;
            if !cast_keeps_order(&input_dtype, dtype) {
                return None;
            }
            order_keeping
        },
        AExpr::Function {
            input,
            function: IRFunctionExpr::Negate,
            ..
        } => rec(input[0].node())?.flip(true),
        AExpr::BinaryExpr { left, op, right } => {
            let (left, right) = (*left, *right);
            match op {
                Operator::Plus if is_scalar_literal(right, arena) => rec(left)?,
                Operator::Plus if is_scalar_literal(left, arena) => rec(right)?,
                Operator::Minus if is_scalar_literal(right, arena) => rec(left)?,
                Operator::Minus if is_scalar_literal(left, arena) => rec(right)?.flip(true),
                Operator::Multiply => {
                    let (input, factor) =
                        match (int_literal(left, arena), int_literal(right, arena)) {
                            (None, Some(factor)) => (left, factor),
                            (Some(factor), None) => (right, factor),
                            _ => return None,
                        };
                    if factor == 0 {
                        return None;
                    }
                    rec(input)?.flip(factor < 0)
                },
                Operator::FloorDivide => {
                    let divisor = int_literal(right, arena)?;
                    if divisor == 0 {
                        return None;
                    }
                    rec(left)?.flip(divisor < 0)
                },
                _ => return None,
            }
        },
        _ => return None,
    };
    let dtype = ae.to_dtype(&ToFieldContext::new(arena, schema)).ok()?;
    has_ordered_values(&dtype).then_some(order_keeping)
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

/// Casts that keep every value apart from overflow, so they can't raise and keep order.
fn cast_keeps_order(from: &DataType, to: &DataType) -> bool {
    let lossless = |from: &DataType, to: &DataType| {
        from == to || get_numeric_upcast_supertype_lossless(from, to).as_ref() == Some(to)
    };
    match (from, to) {
        (from, to) if from.is_integer() => lossless(from, &to.to_physical()),
        #[cfg(all(feature = "dtype-date", feature = "dtype-datetime"))]
        (DataType::Date, DataType::Datetime(_, _)) => true,
        #[cfg(feature = "dtype-datetime")]
        (DataType::Datetime(_, from_tz), DataType::Datetime(_, to_tz)) => from_tz == to_tz,
        #[cfg(feature = "dtype-duration")]
        (DataType::Duration(_), DataType::Duration(_)) => true,
        _ => from == to,
    }
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
