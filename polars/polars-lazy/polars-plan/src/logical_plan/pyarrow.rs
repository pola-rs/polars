use polars_core::datatypes::{AnyValue, DataType};

use crate::prelude::*;

// convert to a pyarrow expression that can be evaluated with pythons eval
pub(super) fn predicate_to_pa(predicate: Node, expr_arena: &Arena<AExpr>) -> Option<String> {
    match expr_arena.get(predicate) {
        AExpr::BinaryExpr { left, right, op } => {
            if op.is_comparison() {
                let left = predicate_to_pa(*left, expr_arena)?;
                let right = predicate_to_pa(*right, expr_arena)?;
                Some(format!("({left} {op} {right})"))
            } else {
                None
            }
        }
        AExpr::Column(name) => Some(format!("pa.dataset.field('{}')", name.as_ref())),
        AExpr::Alias(input, _) => predicate_to_pa(*input, expr_arena),
        AExpr::Literal(lv) => {
            let av = lv.to_anyvalue()?;
            let dtype = av.dtype();
            if dtype.is_float() {
                let val = av.extract::<f64>()?;
                Some(format!("{val}"))
            } else if dtype.is_integer() {
                let val = av.extract::<i64>()?;
                Some(format!("{val}"))
            } else if matches!(dtype, DataType::Utf8) {
                let val = match &av {
                    AnyValue::Utf8(s) => s,
                    AnyValue::Utf8Owned(s) => s.as_str(),
                    _ => unreachable!(),
                };
                Some(format!("'{val}'"))
            } else if matches!(dtype, DataType::Boolean) {
                if let AnyValue::Boolean(val) = av {
                    // python bools are capitalized
                    if val {
                        Some("True".to_string())
                    } else {
                        Some("False".to_string())
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }
        AExpr::Function {
            function: FunctionExpr::Not,
            input,
            ..
        } => {
            let input = input.first().unwrap();
            let input = predicate_to_pa(*input, expr_arena)?;
            Some(format!("~({input})"))
        }
        AExpr::Function {
            function: FunctionExpr::IsNull,
            input,
            ..
        } => {
            let input = input.first().unwrap();
            let input = predicate_to_pa(*input, expr_arena)?;
            Some(format!("({input}).is_null()"))
        }
        AExpr::Function {
            function: FunctionExpr::IsNotNull,
            input,
            ..
        } => {
            let input = input.first().unwrap();
            let input = predicate_to_pa(*input, expr_arena)?;
            Some(format!("~({input}).is_null()"))
        }
        _ => None,
    }
}
