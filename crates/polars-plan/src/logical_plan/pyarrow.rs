use std::fmt::Write;

use polars_core::datatypes::AnyValue;
use polars_core::prelude::{TimeUnit, TimeZone};

use crate::prelude::*;

#[derive(Default, Copy, Clone)]
pub(super) struct Args {
    // pyarrow doesn't allow `filter([True, False])`
    // but does allow `filter(field("a").isin([True, False]))`
    allow_literal_series: bool,
}

fn to_py_datetime(v: i64, tu: &TimeUnit, tz: Option<&TimeZone>) -> String {
    // note: `_to_python_datetime` and the `Datetime`
    // dtype have to be in-scope on the python side
    match tz {
        None => format!("_to_python_datetime({},'{}')", v, tu.to_ascii()),
        Some(tz) => format!("_to_python_datetime({},'{}',{})", v, tu.to_ascii(), tz),
    }
}

// convert to a pyarrow expression that can be evaluated with pythons eval
pub(super) fn predicate_to_pa(
    predicate: Node,
    expr_arena: &Arena<AExpr>,
    args: Args,
) -> Option<String> {
    match expr_arena.get(predicate) {
        AExpr::BinaryExpr { left, right, op } => {
            if op.is_comparison() {
                let left = predicate_to_pa(*left, expr_arena, args)?;
                let right = predicate_to_pa(*right, expr_arena, args)?;
                Some(format!("({left} {op} {right})"))
            } else {
                None
            }
        },
        AExpr::Column(name) => Some(format!("pa.compute.field('{}')", name.as_ref())),
        AExpr::Alias(input, _) => predicate_to_pa(*input, expr_arena, args),
        AExpr::Literal(LiteralValue::Series(s)) => {
            if !args.allow_literal_series || s.is_empty() || s.len() > 100 {
                None
            } else {
                let mut list_repr = String::with_capacity(s.len() * 5);
                list_repr.push('[');
                for av in s.iter() {
                    if let AnyValue::Boolean(v) = av {
                        let s = if v { "True" } else { "False" };
                        write!(list_repr, "{},", s).unwrap();
                    } else if let AnyValue::Datetime(v, tu, tz) = av {
                        let dtm = to_py_datetime(v, &tu, tz.as_ref());
                        write!(list_repr, "{dtm},").unwrap();
                    } else if let AnyValue::Date(v) = av {
                        write!(list_repr, "_to_python_date({v}),").unwrap();
                    } else {
                        write!(list_repr, "{av},").unwrap();
                    }
                }
                // pop last comma
                list_repr.pop();
                list_repr.push(']');
                Some(list_repr)
            }
        },
        AExpr::Literal(lv) => {
            let av = lv.to_anyvalue()?;
            let dtype = av.dtype();
            match av.as_borrowed() {
                AnyValue::Utf8(s) => Some(format!("'{s}'")),
                AnyValue::Boolean(val) => {
                    // python bools are capitalized
                    if val {
                        Some("pa.compute.scalar(True)".to_string())
                    } else {
                        Some("pa.compute.scalar(False)".to_string())
                    }
                },
                #[cfg(feature = "dtype-date")]
                AnyValue::Date(v) => {
                    // the function `_to_python_date` and the `Date`
                    // dtype have to be in scope on the python side
                    Some(format!("_to_python_date({v})"))
                },
                #[cfg(feature = "dtype-datetime")]
                AnyValue::Datetime(v, tu, tz) => Some(to_py_datetime(v, &tu, tz.as_ref())),
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
                },
            }
        },
        AExpr::Function {
            function: FunctionExpr::Boolean(BooleanFunction::Not),
            input,
            ..
        } => {
            let input = input.first().unwrap();
            let input = predicate_to_pa(*input, expr_arena, args)?;
            Some(format!("~({input})"))
        },
        AExpr::Function {
            function: FunctionExpr::Boolean(BooleanFunction::IsNull),
            input,
            ..
        } => {
            let input = input.first().unwrap();
            let input = predicate_to_pa(*input, expr_arena, args)?;
            Some(format!("({input}).is_null()"))
        },
        AExpr::Function {
            function: FunctionExpr::Boolean(BooleanFunction::IsNotNull),
            input,
            ..
        } => {
            let input = input.first().unwrap();
            let input = predicate_to_pa(*input, expr_arena, args)?;
            Some(format!("~({input}).is_null()"))
        },
        #[cfg(feature = "is_in")]
        AExpr::Function {
            function: FunctionExpr::Boolean(BooleanFunction::IsIn),
            input,
            ..
        } => {
            let col = predicate_to_pa(*input.get(0)?, expr_arena, args)?;
            let mut args = args;
            args.allow_literal_series = true;
            let values = predicate_to_pa(*input.get(1)?, expr_arena, args)?;

            Some(format!("({col}).isin({values})"))
        },
        _ => None,
    }
}
