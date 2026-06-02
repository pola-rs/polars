use std::fmt::Write;

use polars_core::datatypes::AnyValue;
#[cfg(feature = "dtype-datetime")]
use polars_core::prelude::TimeZone;
use polars_core::prelude::{DataType, ExplodeOptions, TimeUnit};
use polars_core::series::Series;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
use pyo3::types::PyDate;
use pyo3::types::PyList;
#[cfg(feature = "dtype-datetime")]
use pyo3::types::{PyDateTime, PyTzInfo};

use crate::prelude::*;

// Don't convert more than this amount of items to Python objects.
const LIST_ITEM_LIMIT: usize = 100;

#[cfg(feature = "is_in")]
pub(crate) enum IsInHaystack {
    Empty, // fast path for when haystack is empty; returns False
    Series(Series),
}

#[cfg(feature = "is_in")]
pub(crate) fn needle_isin_haystack(lv: &LiteralValue, nulls_equal: bool) -> Option<IsInHaystack> {
    if !lv.get_datatype().is_list() {
        return None;
    }

    let mut haystack_series = if let LiteralValue::Series(s) = lv
        && s.dtype().is_list()
        && s.len() == 1
    {
        if s.null_count() == 0 {
            s.explode(ExplodeOptions {
                empty_as_null: false,
                keep_nulls: false,
            })
            .ok()?
        } else {
            Series::full_null(PlSmallStr::EMPTY, 0, &DataType::Null)
        }
    } else if let Some(AnyValue::List(s)) = lv.to_any_value() {
        s
    } else if lv.is_null() {
        Series::full_null(PlSmallStr::EMPTY, 0, &DataType::Null)
    } else {
        return None;
    };

    let converted_len = haystack_series.len()
        - if nulls_equal {
            0
        } else {
            haystack_series.null_count()
        };

    if converted_len > LIST_ITEM_LIMIT {
        return None;
    }
    if converted_len == 0 {
        return Some(IsInHaystack::Empty);
    }
    if !nulls_equal {
        haystack_series = haystack_series.drop_nulls();
    }
    Some(IsInHaystack::Series(haystack_series))
}

#[derive(Default, Copy, Clone)]
struct PyarrowArgs {
    // pyarrow doesn't allow `filter([True, False])`
    // but does allow `filter(field("a").isin([True, False]))`
    allow_literal_series: bool,
}

#[cfg(feature = "dtype-datetime")]
fn to_py_datetime(v: i64, tu: &TimeUnit, tz: Option<&TimeZone>) -> String {
    // note: `to_py_datetime` and the `Datetime`
    // dtype have to be in-scope on the python side
    match tz {
        None => format!("to_py_datetime({},'{}')", v, tu.to_ascii()),
        Some(tz) => format!("to_py_datetime({},'{}','{}')", v, tu.to_ascii(), tz),
    }
}

fn sanitize(name: &str) -> Option<&str> {
    if name.chars().all(|c| match c {
        ' ' => true,
        '-' => true,
        '_' => true,
        c => c.is_alphanumeric(),
    }) {
        Some(name)
    } else {
        None
    }
}

// Build an eval-able / AST-walker-compatible string predicate (e.g.
// `pa.compute.field('x') > pa.compute.scalar(1)`). Used by the iceberg and
// delta paths which feed the string into Python (delta `eval`s it,
// iceberg walks it via `try_convert_pyarrow_predicate`).
pub fn predicate_to_pa(predicate: Node, expr_arena: &Arena<AExpr>) -> Option<String> {
    predicate_to_pa_inner(predicate, expr_arena, PyarrowArgs::default())
}

fn predicate_to_pa_inner(
    predicate: Node,
    expr_arena: &Arena<AExpr>,
    args: PyarrowArgs,
) -> Option<String> {
    match expr_arena.get(predicate) {
        AExpr::BinaryExpr { left, right, op } => {
            if op.is_comparison_or_bitwise() {
                let left = predicate_to_pa_inner(*left, expr_arena, args)?;
                let right = predicate_to_pa_inner(*right, expr_arena, args)?;
                Some(format!("({left} {op} {right})"))
            } else {
                None
            }
        },
        AExpr::Column(name) => {
            let name = sanitize(name)?;
            Some(format!("pa.compute.field('{name}')"))
        },
        AExpr::Literal(LiteralValue::Series(s)) => {
            if !args.allow_literal_series || s.is_empty() || s.len() > 100 {
                None
            } else {
                let mut list_repr = String::with_capacity(s.len() * 5);
                list_repr.push('[');
                for av in s.iter() {
                    match av {
                        AnyValue::Boolean(v) => {
                            let s = if v { "True" } else { "False" };
                            write!(list_repr, "{s},").unwrap();
                        },
                        #[cfg(feature = "dtype-datetime")]
                        AnyValue::Datetime(v, tu, tz) => {
                            let dtm = to_py_datetime(v, &tu, tz);
                            write!(list_repr, "{dtm},").unwrap();
                        },
                        #[cfg(feature = "dtype-date")]
                        AnyValue::Date(v) => {
                            write!(list_repr, "to_py_date({v}),").unwrap();
                        },
                        AnyValue::String(s) => {
                            let _ = sanitize(s)?;
                            write!(list_repr, "{av},").unwrap();
                        },
                        AnyValue::Binary(_) | AnyValue::List(_) => return None,
                        #[cfg(feature = "dtype-array")]
                        AnyValue::Array(_, _) => return None,
                        #[cfg(feature = "dtype-struct")]
                        AnyValue::Struct(_, _, _) => return None,
                        _ => {
                            write!(list_repr, "{av},").unwrap();
                        },
                    }
                }
                list_repr.pop();
                list_repr.push(']');
                Some(list_repr)
            }
        },
        AExpr::Literal(lv) => {
            let av = lv.to_any_value()?;
            let dtype = av.dtype();
            match av.as_borrowed() {
                AnyValue::String(s) => {
                    let s = sanitize(s)?;
                    Some(format!("'{s}'"))
                },
                AnyValue::Boolean(val) => {
                    if val {
                        Some("pa.compute.scalar(True)".to_string())
                    } else {
                        Some("pa.compute.scalar(False)".to_string())
                    }
                },
                #[cfg(feature = "dtype-date")]
                AnyValue::Date(v) => Some(format!("to_py_date({v})")),
                #[cfg(feature = "dtype-datetime")]
                AnyValue::Datetime(v, tu, tz) => Some(to_py_datetime(v, &tu, tz)),
                AnyValue::Binary(_) | AnyValue::List(_) => None,
                #[cfg(feature = "dtype-array")]
                AnyValue::Array(_, _) => None,
                #[cfg(feature = "dtype-struct")]
                AnyValue::Struct(_, _, _) => None,
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
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { .. }),
            input,
            ..
        } => {
            let col = predicate_to_pa_inner(input.first()?.node(), expr_arena, args)?;
            let mut args = args;
            args.allow_literal_series = true;
            let values = predicate_to_pa_inner(input.get(1)?.node(), expr_arena, args)?;

            Some(format!("({col}).isin({values})"))
        },
        #[cfg(feature = "is_between")]
        AExpr::Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsBetween { closed }),
            input,
            ..
        } => {
            if !matches!(expr_arena.get(input.first()?.node()), AExpr::Column(_)) {
                None
            } else {
                let col = predicate_to_pa_inner(input.first()?.node(), expr_arena, args)?;
                let left_cmp_op = match closed {
                    ClosedInterval::None | ClosedInterval::Right => Operator::Gt,
                    ClosedInterval::Both | ClosedInterval::Left => Operator::GtEq,
                };
                let right_cmp_op = match closed {
                    ClosedInterval::None | ClosedInterval::Left => Operator::Lt,
                    ClosedInterval::Both | ClosedInterval::Right => Operator::LtEq,
                };

                let lower = predicate_to_pa_inner(input.get(1)?.node(), expr_arena, args)?;
                let upper = predicate_to_pa_inner(input.get(2)?.node(), expr_arena, args)?;

                Some(format!(
                    "(({col} {left_cmp_op} {lower}) & ({col} {right_cmp_op} {upper}))"
                ))
            }
        },
        AExpr::Function {
            function, input, ..
        } => {
            let input = input.first().unwrap().node();
            let input = predicate_to_pa_inner(input, expr_arena, args)?;

            match function {
                IRFunctionExpr::Boolean(IRBooleanFunction::Not) => Some(format!("~({input})")),
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNull) => {
                    Some(format!("({input}).is_null()"))
                },
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull) => {
                    Some(format!("~({input}).is_null()"))
                },
                _ => None,
            }
        },
        _ => None,
    }
}

fn binary_op_method(op: &Operator) -> Option<&'static str> {
    Some(match op {
        Operator::Eq => "__eq__",
        Operator::NotEq => "__ne__",
        Operator::Lt => "__lt__",
        Operator::LtEq => "__le__",
        Operator::Gt => "__gt__",
        Operator::GtEq => "__ge__",
        Operator::And | Operator::LogicalAnd => "__and__",
        Operator::Or | Operator::LogicalOr => "__or__",
        Operator::Xor => "__xor__",
        _ => return None,
    })
}

// The main engine of converting AnyValue to a python object
fn anyvalue_to_py<'py>(py: Python<'py>, av: AnyValue<'_>) -> Option<Bound<'py, PyAny>> {
    use pyo3::IntoPyObjectExt;

    let dtype = av.dtype();
    match av.as_borrowed() {
        AnyValue::Null => Some(py.None().into_bound(py)),
        AnyValue::Boolean(v) => v.into_bound_py_any(py).ok(),
        AnyValue::String(s) => s.into_pyobject(py).ok().map(|b| b.into_any()),
        #[cfg(feature = "dtype-date")]
        AnyValue::Date(days) => {
            use chrono::Datelike;
            let date = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)?
                .checked_add_signed(chrono::Duration::days(days as i64))?;
            PyDate::new(py, date.year(), date.month() as u8, date.day() as u8)
                .ok()
                .map(|b| b.into_any())
        },
        #[cfg(feature = "dtype-datetime")]
        AnyValue::Datetime(value, time_unit, time_zone) => {
            use chrono::{Datelike, Timelike};
            let micros: i64 = match time_unit {
                TimeUnit::Nanoseconds => value / 1000,
                TimeUnit::Microseconds => value,
                TimeUnit::Milliseconds => value * 1000,
            };
            let dt = chrono::DateTime::<chrono::Utc>::from_timestamp_micros(micros)?.naive_utc();
            let tzinfo: Option<Bound<'py, PyTzInfo>> = if let Some(tz) = time_zone {
                let zi = py.import("zoneinfo").ok()?;
                let obj = zi.getattr("ZoneInfo").ok()?.call1((tz.to_string(),)).ok()?;
                Some(obj.cast_into().ok()?)
            } else {
                None
            };
            PyDateTime::new(
                py,
                dt.year(),
                dt.month() as u8,
                dt.day() as u8,
                dt.hour() as u8,
                dt.minute() as u8,
                dt.second() as u8,
                dt.nanosecond() / 1000,
                tzinfo.as_ref(),
            )
            .ok()
            .map(|b| b.into_any())
        },
        // TODO: Worth supporting?
        AnyValue::Binary(_) | AnyValue::List(_) => None,
        #[cfg(feature = "dtype-array")]
        AnyValue::Array(_, _) => None,
        #[cfg(feature = "dtype-struct")]
        AnyValue::Struct(_, _, _) => None,
        av => {
            if dtype.is_float() {
                // TODO: Opportunity to downcast
                let v = av.extract::<f64>()?;
                v.into_bound_py_any(py).ok()
            } else if dtype.is_integer() {
                let v = av.extract::<i64>()?;
                v.into_bound_py_any(py).ok()
            } else {
                None
            }
        },
    }
}

fn series_to_py_list<'py>(py: Python<'py>, s: &Series) -> Option<Bound<'py, PyList>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(s.len());
    for av in s.iter() {
        items.push(anyvalue_to_py(py, av)?);
    }
    PyList::new(py, &items).ok()
}

// Convert an AExpr predicate to a pyarrow expression using python.
pub fn aexpr_to_pyarrow<'py>(
    py: Python<'py>,
    pc: &Bound<'py, PyAny>,
    predicate: Node,
    expr_arena: &Arena<AExpr>,
) -> Option<Bound<'py, PyAny>> {
    match expr_arena.get(predicate) {
        AExpr::BinaryExpr { left, right, op } => {
            let method = binary_op_method(op)?;
            let l = aexpr_to_pyarrow(py, pc, *left, expr_arena)?;
            let r = aexpr_to_pyarrow(py, pc, *right, expr_arena)?;
            l.call_method1(method, (r,)).ok()
        },
        AExpr::Column(name) => pc.call_method1("field", (name,)).ok(),
        AExpr::Literal(LiteralValue::Series(_)) => None,
        AExpr::Literal(lv) => {
            let av = lv.to_any_value()?;
            let val = anyvalue_to_py(py, av)?;
            pc.call_method1("scalar", (val,)).ok()
        },
        #[cfg(feature = "is_in")]
        AExpr::Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { nulls_equal }),
            input,
            ..
        } => {
            let col = aexpr_to_pyarrow(py, pc, input.first()?.node(), expr_arena)?;
            let rhs_node = input.get(1)?.node();

            let AExpr::Literal(lv) = expr_arena.get(rhs_node) else {
                return None;
            };
            let values_list = match needle_isin_haystack(lv, *nulls_equal)? {
                IsInHaystack::Empty => return pc.call_method1("scalar", (false,)).ok(),
                IsInHaystack::Series(s) => series_to_py_list(py, &s)?,
            };

            col.call_method1("isin", (values_list,)).ok()
        },
        #[cfg(feature = "is_between")]
        AExpr::Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsBetween { closed }),
            input,
            ..
        } => {
            if !matches!(expr_arena.get(input.first()?.node()), AExpr::Column(_)) {
                return None;
            }
            let col = aexpr_to_pyarrow(py, pc, input.first()?.node(), expr_arena)?;
            let left_method = match closed {
                ClosedInterval::None | ClosedInterval::Right => "__gt__",
                ClosedInterval::Both | ClosedInterval::Left => "__ge__",
            };
            let right_method = match closed {
                ClosedInterval::None | ClosedInterval::Left => "__lt__",
                ClosedInterval::Both | ClosedInterval::Right => "__le__",
            };

            let lower = aexpr_to_pyarrow(py, pc, input.get(1)?.node(), expr_arena)?;
            let upper = aexpr_to_pyarrow(py, pc, input.get(2)?.node(), expr_arena)?;

            let lower_cmp = col.call_method1(left_method, (lower,)).ok()?;
            let upper_cmp = col.call_method1(right_method, (upper,)).ok()?;
            lower_cmp.call_method1("__and__", (upper_cmp,)).ok()
        },
        AExpr::Function {
            function, input, ..
        } => {
            let input = input.first().unwrap().node();
            let input = aexpr_to_pyarrow(py, pc, input, expr_arena)?;

            match function {
                IRFunctionExpr::Boolean(IRBooleanFunction::Not) => {
                    // ~ operator
                    input.call_method0("__invert__").ok()
                },
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNull) => {
                    input.call_method0("is_null").ok()
                },
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull) => input
                    .call_method0("is_null")
                    .ok()?
                    .call_method0("__invert__")
                    .ok(),
                _ => None,
            }
        },
        _ => None,
    }
}
