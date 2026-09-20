use polars_utils::floor_divmod::FloorDivMod;

use super::*;

/// Folds arithmetic on two dynamic literals.
pub(crate) fn fold_dyn_binary(
    l: &DynLiteralValue,
    r: &DynLiteralValue,
    op: Operator,
) -> Option<DynLiteralValue> {
    use DynLiteralValue::*;
    match (l, r) {
        (Int(a), Int(b)) => {
            let (a, b) = (*a, *b);
            let v = match op {
                Operator::Plus => a.wrapping_add(b),
                Operator::Minus => a.wrapping_sub(b),
                Operator::Multiply => a.wrapping_mul(b),
                Operator::Modulus | Operator::FloorDivide | Operator::RustDivide => {
                    if b == 0 {
                        return None;
                    }
                    let (div, rem) = a.wrapping_floor_div_mod(b);
                    if op == Operator::Modulus { rem } else { div }
                },
                _ => return None,
            };
            Some(Int(v))
        },
        (Int(_) | Float(_), Int(_) | Float(_)) => {
            let as_f64 = |v: &DynLiteralValue| match v {
                Int(i) => *i as f64,
                Float(f) => *f,
                _ => unreachable!(),
            };
            let (a, b) = (as_f64(l), as_f64(r));
            let v = match op {
                Operator::Plus => a + b,
                Operator::Minus => a - b,
                Operator::Multiply => a * b,
                Operator::Modulus | Operator::FloorDivide => {
                    if b == 0.0 {
                        return None;
                    }
                    let (div, rem) = a.wrapping_floor_div_mod(b);
                    if op == Operator::Modulus { rem } else { div }
                },
                _ => return None,
            };
            Some(Float(v))
        },
        _ => None,
    }
}

pub(crate) fn fold_dyn_negate(v: &DynLiteralValue) -> Option<DynLiteralValue> {
    match v {
        DynLiteralValue::Int(v) => v.checked_neg().map(DynLiteralValue::Int),
        DynLiteralValue::Float(v) => Some(DynLiteralValue::Float(-v)),
        _ => None,
    }
}

/// Folds an expression made only of dynamic literals and arithmetic.
pub(crate) fn try_fold_dyn(ae: &AExpr, arena: &Arena<AExpr>) -> Option<DynLiteralValue> {
    match ae {
        AExpr::Literal(LiteralValue::Dyn(
            v @ (DynLiteralValue::Int(_) | DynLiteralValue::Float(_)),
        )) => Some(v.clone()),
        AExpr::BinaryExpr { left, op, right } => {
            let l = try_fold_dyn(arena.get(*left), arena)?;
            let r = try_fold_dyn(arena.get(*right), arena)?;
            fold_dyn_binary(&l, &r, *op)
        },
        AExpr::Function {
            input,
            function: IRFunctionExpr::Negate,
            ..
        } if input.len() == 1 => fold_dyn_negate(&try_fold_dyn(arena.get(input[0].node()), arena)?),
        _ => None,
    }
}
