use std::fmt::Write;

use polars_core::datatypes::AnyValue;
use polars_core::prelude::{TimeUnit, TimeZone};

use crate::prelude::*;

#[derive(Default, Copy, Clone)]
pub struct PyarrowArgs {
    // pyarrow doesn't allow `filter([True, False])`
    // but does allow `filter(field("a").isin([True, False]))`
    allow_literal_series: bool,
}

fn to_py_datetime(v: i64, tu: &TimeUnit, tz: Option<&TimeZone>) -> String {
    // note: `to_py_datetime` and the `Datetime`
    // dtype have to be in-scope on the python side
    match tz {
        None => format!("to_py_datetime({},'{}')", v, tu.to_ascii()),
        Some(tz) => format!("to_py_datetime({},'{}',{})", v, tu.to_ascii(), tz),
    }
}

// convert to a pyarrow expression that can be evaluated with pythons eval
pub fn predicate_to_pa(
    predicate: Node,
    expr_arena: &Arena<AExpr>,
    args: PyarrowArgs,
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
                        write!(list_repr, "to_py_date({v}),").unwrap();
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
            let av = lv.to_any_value()?;
            let dtype = av.dtype();
            match av.as_borrowed() {
                AnyValue::String(s) => Some(format!("'{s}'")),
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
                    // the function `to_py_date` and the `Date`
                    // dtype have to be in scope on the python side
                    Some(format!("to_py_date({v})"))
                },
                #[cfg(feature = "dtype-datetime")]
                AnyValue::Datetime(v, tu, tz) => Some(to_py_datetime(v, &tu, tz.as_ref())),
                // Activate once pyarrow supports them
                // #[cfg(feature = "dtype-time")]
                // AnyValue::Time(v) => {
                //     // the function `to_py_time` has to be in scope
                //     // on the python side
                //     Some(format!("to_py_time(value={v})"))
                // }
                // #[cfg(feature = "dtype-duration")]
                // AnyValue::Duration(v, tu) => {
                //     // the function `to_py_timedelta` has to be in scope
                //     // on the python side
                //     Some(format!(
                //         "to_py_timedelta(value={}, tu='{}')",
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
        #[cfg(feature = "is_in")]
        AExpr::Function {
            function: FunctionExpr::Boolean(BooleanFunction::IsIn),
            input,
            ..
        } => {
            let col = predicate_to_pa(input.first()?.node(), expr_arena, args)?;
            let mut args = args;
            args.allow_literal_series = true;
            let values = predicate_to_pa(input.get(1)?.node(), expr_arena, args)?;

            Some(format!("({col}).isin({values})"))
        },
        #[cfg(feature = "is_between")]
        AExpr::Function {
            function: FunctionExpr::Boolean(BooleanFunction::IsBetween { closed }),
            input,
            ..
        } => {
            if !matches!(expr_arena.get(input.first()?.node()), AExpr::Column(_)) {
                None
            } else {
                let col = predicate_to_pa(input.first()?.node(), expr_arena, args)?;
                let left_cmp_op = match closed {
                    ClosedInterval::None | ClosedInterval::Right => Operator::Gt,
                    ClosedInterval::Both | ClosedInterval::Left => Operator::GtEq,
                };
                let right_cmp_op = match closed {
                    ClosedInterval::None | ClosedInterval::Left => Operator::Lt,
                    ClosedInterval::Both | ClosedInterval::Right => Operator::LtEq,
                };

                let lower = predicate_to_pa(input.get(1)?.node(), expr_arena, args)?;
                let upper = predicate_to_pa(input.get(2)?.node(), expr_arena, args)?;

                Some(format!(
                    "(({col} {left_cmp_op} {lower}) & ({col} {right_cmp_op} {upper}))"
                ))
            }
        },
        AExpr::Function {
            function, input, ..
        } => {
            let input = input.first().unwrap().node();
            let input = predicate_to_pa(input, expr_arena, args)?;

            match function {
                FunctionExpr::Boolean(BooleanFunction::Not) => Some(format!("~({input})")),
                FunctionExpr::Boolean(BooleanFunction::IsNull) => {
                    Some(format!("({input}).is_null()"))
                },
                FunctionExpr::Boolean(BooleanFunction::IsNotNull) => {
                    Some(format!("~({input}).is_null()"))
                },
                _ => None,
            }
        },
        _ => None,
    }
}
