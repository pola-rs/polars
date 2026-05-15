use polars_core::datatypes::AnyValue;
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

// Convert predicate to a pyarrow expression
pub fn predicate_to_pa(
    predicate: Node,
    expr_arena: &Arena<AExpr>,
) -> Option<polars_utils::python_function::PythonObject> {
    use polars_utils::python_function::PythonObject;

    Python::attach(|py| {
        let pc = py.import("pyarrow.compute").ok()?;
        let expr = aexpr_to_pyarrow(py, &pc, predicate, expr_arena)?;
        Some(PythonObject(expr.unbind()))
    })
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

fn extract_column_name(node: Node, expr_arena: &Arena<AExpr>) -> Option<&str> {
    match expr_arena.get(node) {
        AExpr::Column(name) => Some(name.as_str()),
        _ => None,
    }
}

// Get a literal out of a node w/series guard
fn _extract_literal_value<'py>(
    py: Python<'py>,
    node: Node,
    expr_arena: &Arena<AExpr>,
) -> Option<Bound<'py, PyAny>> {
    if let AExpr::Literal(lv) = expr_arena.get(node) {
        if matches!(lv, LiteralValue::Series(_)) {
            return None;
        }
        let av = lv.to_any_value()?;
        anyvalue_to_py(py, av)
    } else {
        None
    }
}

// Converts AExpr predicate to `pyiceberg.expressions.BooleanExpression``
// eventually passed to `table.scan(row_filter=...)`
pub fn aexpr_to_pyiceberg<'py>(
    py: Python<'py>,
    pe: &Bound<'py, PyAny>,
    predicate: Node,
    expr_arena: &Arena<AExpr>,
) -> Option<Bound<'py, PyAny>> {
    match expr_arena.get(predicate) {
        AExpr::BinaryExpr { left, right, op } => match op {
            Operator::And | Operator::LogicalAnd => {
                let l = aexpr_to_pyiceberg(py, pe, *left, expr_arena)?;
                let r = aexpr_to_pyiceberg(py, pe, *right, expr_arena)?;
                pe.getattr("And").ok()?.call1((l, r)).ok()
            },
            Operator::Or | Operator::LogicalOr => {
                let l = aexpr_to_pyiceberg(py, pe, *left, expr_arena)?;
                let r = aexpr_to_pyiceberg(py, pe, *right, expr_arena)?;
                pe.getattr("Or").ok()?.call1((l, r)).ok()
            },
            Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq => {
                // Iceberg seems to only support `column op literal` format.
                // Have to normalize the expression to figure out which side is the column, and which the literal.
                let (col_name, lit, op) = if let Some(name) = extract_column_name(*left, expr_arena)
                {
                    (name, _extract_literal_value(py, *right, expr_arena)?, *op)
                } else {
                    let name = extract_column_name(*right, expr_arena)?;
                    let lit = _extract_literal_value(py, *left, expr_arena)?;
                    let mirrored = match op {
                        Operator::Lt => Operator::Gt,
                        Operator::LtEq => Operator::GtEq,
                        Operator::Gt => Operator::Lt,
                        Operator::GtEq => Operator::LtEq,
                        other => *other,
                    };
                    (name, lit, mirrored)
                };
                let class = match op {
                    Operator::Eq => "EqualTo",
                    Operator::NotEq => "NotEqualTo",
                    Operator::Lt => "LessThan",
                    Operator::LtEq => "LessThanOrEqual",
                    Operator::Gt => "GreaterThan",
                    Operator::GtEq => "GreaterThanOrEqual",
                    _ => return None,
                };
                pe.getattr(class).ok()?.call1((col_name, lit)).ok()
            },
            _ => None,
        },
        // Iceberg doesn't have a way of expressing a column as a mask so we have to convert it to `column ==True`
        AExpr::Column(name) => pe
            .getattr("EqualTo")
            .ok()?
            .call1((name.as_str(), true))
            .ok(),
        #[cfg(feature = "is_in")]
        AExpr::Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { nulls_equal }),
            input,
            ..
        } => {
            let colname = extract_column_name(input.first()?.node(), expr_arena)?;
            let rhs_node = input.get(1)?.node();
            let AExpr::Literal(lv) = expr_arena.get(rhs_node) else {
                return None;
            };
            let values_list = match needle_isin_haystack(lv, *nulls_equal)? {
                IsInHaystack::Empty => return pe.getattr("AlwaysFalse").ok()?.call0().ok(),
                IsInHaystack::Series(s) => series_to_py_list(py, &s)?,
            };
            pe.getattr("In").ok()?.call1((colname, values_list)).ok()
        },
        #[cfg(feature = "is_between")]
        AExpr::Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsBetween { closed }),
            input,
            ..
        } => {
            let colname = extract_column_name(input.first()?.node(), expr_arena)?;
            let lower = _extract_literal_value(py, input.get(1)?.node(), expr_arena)?;
            let upper = _extract_literal_value(py, input.get(2)?.node(), expr_arena)?;
            let left_class = match closed {
                ClosedInterval::None | ClosedInterval::Right => "GreaterThan",
                ClosedInterval::Both | ClosedInterval::Left => "GreaterThanOrEqual",
            };
            let right_class = match closed {
                ClosedInterval::None | ClosedInterval::Left => "LessThan",
                ClosedInterval::Both | ClosedInterval::Right => "LessThanOrEqual",
            };
            let l = pe.getattr(left_class).ok()?.call1((colname, lower)).ok()?;
            let r = pe.getattr(right_class).ok()?.call1((colname, upper)).ok()?;
            pe.getattr("And").ok()?.call1((l, r)).ok()
        },
        AExpr::Function {
            function, input, ..
        } => match function {
            IRFunctionExpr::Boolean(IRBooleanFunction::Not) => {
                let inner = aexpr_to_pyiceberg(py, pe, input.first()?.node(), expr_arena)?;
                pe.getattr("Not").ok()?.call1((inner,)).ok()
            },
            IRFunctionExpr::Boolean(IRBooleanFunction::IsNull) => {
                let colname = extract_column_name(input.first()?.node(), expr_arena)?;
                pe.getattr("IsNull").ok()?.call1((colname,)).ok()
            },
            IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull) => {
                let colname = extract_column_name(input.first()?.node(), expr_arena)?;
                let is_null = pe.getattr("IsNull").ok()?.call1((colname,)).ok()?;
                pe.getattr("Not").ok()?.call1((is_null,)).ok()
            },
            _ => None,
        },
        _ => None,
    }
}
