//! Filters as JSON documents (`filter_expr.v1`, specified in
//! `polars_io_ext_ffi::lazyframe_resolver_v1`), for resolvers of I/O plugins.
//!
//! The translation is structural only: it never decides whether a plugin can apply a filter.
//! Expressions the document has no node for become `{"unknown": …}`, so a plugin still sees the
//! shape of the filter around them.
use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};
use serde_json::{Value, json};

use crate::plans::{AExpr, IRBooleanFunction, IRFunctionExpr, LiteralValue};
use crate::prelude::Operator;

/// The filter document of a predicate.
pub fn filter_document(node: Node, arena: &Arena<AExpr>) -> String {
    expr(node, arena).to_string()
}

fn unknown(what: &str) -> Value {
    json!({ "unknown": what })
}

fn op(name: &str, args: Vec<Value>) -> Value {
    json!({ "op": name, "args": args })
}

fn expr(node: Node, arena: &Arena<AExpr>) -> Value {
    match arena.get(node) {
        AExpr::Column(name) => json!({ "col": name.as_str() }),
        AExpr::Literal(lv) if lv.is_scalar() => match lv.to_any_value().and_then(|av| literal(&av))
        {
            Some(lit) => json!({ "lit": lit }),
            None => unknown("literal"),
        },
        AExpr::BinaryExpr { left, op: o, right } => {
            let name = match o {
                Operator::And | Operator::LogicalAnd => "and",
                Operator::Or | Operator::LogicalOr => "or",
                Operator::Eq => "eq",
                Operator::NotEq => "neq",
                Operator::Lt => "lt",
                Operator::LtEq => "lt_eq",
                Operator::Gt => "gt",
                Operator::GtEq => "gt_eq",
                Operator::EqValidity => "eq_missing",
                Operator::NotEqValidity => "neq_missing",
                _ => return unknown("binary expression"),
            };
            op(name, vec![expr(*left, arena), expr(*right, arena)])
        },
        AExpr::Cast { expr: e, dtype, .. } => match dtype_name(dtype) {
            Some(d) => json!({ "op": "cast", "dtype": d, "args": [expr(*e, arena)] }),
            None => unknown("cast"),
        },
        AExpr::Function {
            input, function, ..
        } => {
            let args = || {
                input
                    .iter()
                    .map(|e| expr(e.node(), arena))
                    .collect::<Vec<_>>()
            };
            match function {
                IRFunctionExpr::Boolean(f) => match f {
                    IRBooleanFunction::Not => op("not", args()),
                    IRBooleanFunction::IsNull => op("is_null", args()),
                    IRBooleanFunction::IsNotNull => op("is_not_null", args()),
                    #[cfg(feature = "is_between")]
                    IRBooleanFunction::IsBetween { closed } => {
                        use crate::prelude::ClosedInterval;
                        let closed = match closed {
                            ClosedInterval::Both => "both",
                            ClosedInterval::Left => "left",
                            ClosedInterval::Right => "right",
                            ClosedInterval::None => "none",
                        };
                        json!({ "op": "is_between", "closed": closed, "args": args() })
                    },
                    #[cfg(feature = "is_in")]
                    IRBooleanFunction::IsIn {
                        nulls_equal,
                        needle_cast: None,
                    } => {
                        let [e, list] = input.as_slice() else {
                            return unknown("is_in");
                        };
                        match is_in_values(list.node(), arena) {
                            Some(values) => json!({
                                "op": "is_in",
                                "nulls_equal": nulls_equal,
                                "values": values,
                                "args": [expr(e.node(), arena)],
                            }),
                            None => unknown("is_in"),
                        }
                    },
                    _ => unknown("boolean function"),
                },
                #[cfg(feature = "strings")]
                IRFunctionExpr::StringExpr(f) => {
                    use crate::plans::IRStringFunction;
                    let name = match f {
                        IRStringFunction::StartsWith => "starts_with",
                        IRStringFunction::EndsWith => "ends_with",
                        #[cfg(feature = "regex")]
                        IRStringFunction::Contains { literal: true, .. } => "contains_literal",
                        _ => return unknown("string function"),
                    };
                    op(name, args())
                },
                _ => unknown("function"),
            }
        },
        _ => unknown("expression"),
    }
}

/// The haystack of `is_in`: a list scalar, or a Series (possibly of one list).
#[cfg(feature = "is_in")]
fn is_in_values(node: Node, arena: &Arena<AExpr>) -> Option<Value> {
    let values = match arena.get(node) {
        AExpr::Literal(LiteralValue::Series(s)) => match s.dtype() {
            DataType::List(_) if s.len() == 1 => s.list().ok()?.get_as_series(0)?,
            _ => (**s).clone(),
        },
        AExpr::Literal(LiteralValue::Scalar(sc)) => match sc.value() {
            AnyValue::List(s) => s.clone(),
            _ => return None,
        },
        _ => return None,
    };
    let dtype = dtype_name(values.dtype())?;
    let values = values
        .iter()
        .map(|av| value(&av).map(|(_, v)| v))
        .collect::<Option<Vec<_>>>()?;
    Some(json!({ "dtype": dtype, "values": values }))
}

fn time_unit(tu: &TimeUnit) -> &'static str {
    match tu {
        TimeUnit::Nanoseconds => "ns",
        TimeUnit::Microseconds => "us",
        TimeUnit::Milliseconds => "ms",
    }
}

/// The dtype's name in the document, if it has one.
fn dtype_name(dtype: &DataType) -> Option<String> {
    Some(match dtype {
        DataType::Null => "null".into(),
        DataType::Boolean => "bool".into(),
        DataType::String => "str".into(),
        DataType::Int8 => "i8".into(),
        DataType::Int16 => "i16".into(),
        DataType::Int32 => "i32".into(),
        DataType::Int64 => "i64".into(),
        DataType::Int128 => "i128".into(),
        DataType::UInt8 => "u8".into(),
        DataType::UInt16 => "u16".into(),
        DataType::UInt32 => "u32".into(),
        DataType::UInt64 => "u64".into(),
        DataType::UInt128 => "u128".into(),
        DataType::Float32 => "f32".into(),
        DataType::Float64 => "f64".into(),
        #[cfg(feature = "dtype-date")]
        DataType::Date => "date".into(),
        #[cfg(feature = "dtype-time")]
        DataType::Time => "time".into(),
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(tu, None) => format!("datetime[{}]", time_unit(tu)),
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(tu, Some(tz)) => format!("datetime[{}, {tz}]", time_unit(tu)),
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(tu) => format!("duration[{}]", time_unit(tu)),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(p, s) => format!("decimal[{p}, {s}]"),
        _ => return None,
    })
}

/// The dtype name and the value of `av` in the document.
fn value(av: &AnyValue<'_>) -> Option<(String, Value)> {
    let text = |v: &dyn std::fmt::Display| Value::String(v.to_string());
    Some(match av {
        AnyValue::Null => ("null".into(), Value::Null),
        AnyValue::Boolean(b) => ("bool".into(), Value::Bool(*b)),
        AnyValue::String(s) => ("str".into(), Value::String(s.to_string())),
        AnyValue::StringOwned(s) => ("str".into(), Value::String(s.to_string())),
        AnyValue::Int8(v) => ("i8".into(), text(v)),
        AnyValue::Int16(v) => ("i16".into(), text(v)),
        AnyValue::Int32(v) => ("i32".into(), text(v)),
        AnyValue::Int64(v) => ("i64".into(), text(v)),
        AnyValue::Int128(v) => ("i128".into(), text(v)),
        AnyValue::UInt8(v) => ("u8".into(), text(v)),
        AnyValue::UInt16(v) => ("u16".into(), text(v)),
        AnyValue::UInt32(v) => ("u32".into(), text(v)),
        AnyValue::UInt64(v) => ("u64".into(), text(v)),
        AnyValue::UInt128(v) => ("u128".into(), text(v)),
        // `{:?}` is the shortest text that parses back to the same float.
        AnyValue::Float32(v) => ("f32".into(), Value::String(format!("{v:?}"))),
        AnyValue::Float64(v) => ("f64".into(), Value::String(format!("{v:?}"))),
        #[cfg(feature = "dtype-date")]
        AnyValue::Date(d) => ("date".into(), Value::from(*d)),
        #[cfg(feature = "dtype-time")]
        AnyValue::Time(ns) => ("time".into(), text(ns)),
        #[cfg(feature = "dtype-datetime")]
        AnyValue::Datetime(v, tu, tz) => {
            (dtype_name(&DataType::Datetime(*tu, tz.cloned()))?, text(v))
        },
        #[cfg(feature = "dtype-datetime")]
        AnyValue::DatetimeOwned(v, tu, tz) => (
            dtype_name(&DataType::Datetime(*tu, tz.as_deref().cloned()))?,
            text(v),
        ),
        #[cfg(feature = "dtype-duration")]
        AnyValue::Duration(v, tu) => (dtype_name(&DataType::Duration(*tu))?, text(v)),
        #[cfg(feature = "dtype-decimal")]
        AnyValue::Decimal(v, p, s) => (dtype_name(&DataType::Decimal(*p, *s))?, text(v)),
        _ => return None,
    })
}

/// A scalar literal: `{"dtype": …, "value": …}`.
fn literal(av: &AnyValue<'_>) -> Option<Value> {
    let (dtype, value) = value(av)?;
    Some(json!({ "dtype": dtype, "value": value }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plans::ExprIR;

    #[test]
    fn documents() {
        let mut arena = Arena::new();
        let a = arena.add(AExpr::Column("a".into()));
        let five = arena.add(AExpr::Literal(LiteralValue::Scalar(Scalar::from(5i64))));
        let cmp = arena.add(AExpr::BinaryExpr {
            left: a,
            op: Operator::GtEq,
            right: five,
        });
        let not = arena.add(AExpr::Function {
            input: vec![ExprIR::from_node(cmp, &arena)],
            function: IRFunctionExpr::Boolean(IRBooleanFunction::Not),
            options: Default::default(),
        });
        let v: Value = serde_json::from_str(&filter_document(not, &arena)).unwrap();
        assert_eq!(
            v,
            json!({"op": "not", "args": [{"op": "gt_eq", "args": [
                {"col": "a"}, {"lit": {"dtype": "i64", "value": "5"}}
            ]}]})
        );
        assert_eq!(
            value(&AnyValue::Float64(f64::NAN)).unwrap().1,
            Value::String("NaN".into())
        );
        assert_eq!(
            value(&AnyValue::Float64(0.1)).unwrap().1,
            Value::String("0.1".into())
        );
    }
}
