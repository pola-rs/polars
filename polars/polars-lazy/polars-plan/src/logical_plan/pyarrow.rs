use polars_core::datatypes::AnyValue;

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
            match av.as_borrowed() {
                AnyValue::Utf8(s) => Some(format!("'{s}'")),
                AnyValue::Boolean(val) => {
                    // python bools are capitalized
                    if val {
                        Some("True".to_string())
                    } else {
                        Some("False".to_string())
                    }
                }
                #[cfg(feature = "dtype-date")]
                AnyValue::Date(v) => {
                    // the function `_to_python_datetime` and `Date` have to be in scope
                    // on the python side
                    Some(format!("_to_python_datetime(value={v}, dtype=Date)"))
                }
                #[cfg(feature = "dtype-datetime")]
                AnyValue::Datetime(v, tu, tz) => {
                    // the function `_to_python_datetime` and `Datetime` have to be in scope
                    // on the python side
                    match tz {
                        None => Some(format!(
                            "_to_python_datetime(value={}, dtype=Datetime, tu='{}')",
                            v,
                            tu.to_ascii()
                        )),
                        Some(tz) => Some(format!(
                            "to_python_datetime(value={}, dtype=Datetime, tu='{}', tz={})",
                            v,
                            tu.to_ascii(),
                            tz
                        )),
                    }
                }
                // Activate once pyarrow supports them
                // #[cfg(feature = "dtype-time")]
                // AnyValue::Time(v) => {
                //     // the function `_to_python_time` has to be in scope
                //     // on the python side
                //     Some(format!("_to_python_time(value={v})"))
                // }
                // #[cfg(feature = "dtype-duration")]
                // AnyValue::Duration(v, tu) => {
                //     // the function `_to_python_timedelta` has to be in scope
                //     // on the python side
                //     Some(format!(
                //         "_to_python_timedelta(value={}, tu='{}')",
                //         v,
                //         tu.to_ascii()
                //     ))
                // }
                av => {
                    if dtype.is_float() {
                        let val = av.extract::<f64>()?;
                        Some(format!("{val}"))
                    } else if dtype.is_integer() {
                        let val = av.extract::<i64>()?;
                        Some(format!("{val}"))
                    } else {
                        None
                    }
                }
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
