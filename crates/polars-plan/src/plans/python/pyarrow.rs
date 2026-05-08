use std::fmt::Write;

use polars_core::datatypes::AnyValue;
use polars_core::prelude::{DataType, TimeUnit, TimeZone};
use polars_core::series::Series;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::prelude::*;

// Don't convert more than this amount of items to Python objects.
const LIST_ITEM_LIMIT: usize = 100;

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

fn series_to_pyarrow_list(s: &polars_core::prelude::Series) -> Option<String> {
    if s.is_empty() {
        return Some("[]".to_string());
    }

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
            // Hard to sanitize
            AnyValue::Binary(_) | AnyValue::List(_) => return None,
            #[cfg(feature = "dtype-array")]
            AnyValue::Array(_, _) => return None,
            #[cfg(feature = "dtype-struct")]
            AnyValue::Struct(_, _, _) => return None,
            AnyValue::Null => write!(list_repr, "None,").unwrap(),
            _ => {
                write!(list_repr, "{av},").unwrap();
            },
        }
    }
    // pop last comma
    list_repr.pop();
    list_repr.push(']');
    Some(list_repr)
}

// convert to a pyarrow expression that can be evaluated with pythons eval
pub fn predicate_to_pa(
    predicate: Node,
    expr_arena: &Arena<AExpr>,
    args: PyarrowArgs,
) -> Option<String> {
    match expr_arena.get(predicate) {
        AExpr::BinaryExpr { left, right, op } => {
            if op.is_comparison_or_bitwise() {
                let left = predicate_to_pa(*left, expr_arena, args)?;
                let right = predicate_to_pa(*right, expr_arena, args)?;
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
            if args.allow_literal_series && s.len() <= LIST_ITEM_LIMIT {
                series_to_pyarrow_list(s)
            } else {
                None
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
                AnyValue::Datetime(v, tu, tz) => Some(to_py_datetime(v, &tu, tz)),
                // Hard to sanitize
                AnyValue::Binary(_) | AnyValue::List(_) => None,
                #[cfg(feature = "dtype-array")]
                AnyValue::Array(_, _) => None,
                #[cfg(feature = "dtype-struct")]
                AnyValue::Struct(_, _, _) => None,
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
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { nulls_equal }),
            input,
            ..
        } => {
            let col = predicate_to_pa(input.first()?.node(), expr_arena, args)?;
            let rhs_node = input.get(1)?.node();

            // Explode from length-1 list RHS.
            let values = if let AExpr::Literal(lv) = expr_arena.get(rhs_node)
                && lv.get_datatype().is_list()
            {
                use polars_core::prelude::ExplodeOptions;

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
                    - if *nulls_equal {
                        0
                    } else {
                        haystack_series.null_count()
                    };

                if converted_len > LIST_ITEM_LIMIT {
                    return None;
                }

                if converted_len == 0 {
                    return Some("pa.compute.scalar(False)".to_string());
                }

                if !*nulls_equal {
                    haystack_series = haystack_series.drop_nulls();
                }

                series_to_pyarrow_list(&haystack_series)?
            } else {
                return None;
            };

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

fn anyvalue_to_py<'py>(py: Python<'py>, av: AnyValue<'_>) -> Option<Bound<'py, PyAny>> {
    use pyo3::IntoPyObjectExt;

    let dtype = av.dtype();
    match av.as_borrowed() {
        AnyValue::Null => Some(py.None().into_bound(py)),
        AnyValue::Boolean(v) => v.into_bound_py_any(py).ok(),
        AnyValue::String(s) => {
            let s = sanitize(s)?;
            s.into_pyobject(py).ok().map(|b| b.into_any())
        },
        #[cfg(feature = "dtype-date")]
        AnyValue::Date(days) => {
            let dt_mod = py.import("datetime").ok()?;
            let epoch = dt_mod
                .getattr("date")
                .ok()?
                // Unix epoch start
                .call1((1970i32, 1i32, 1i32))
                .ok()?;
            let delta = dt_mod
                .getattr("timedelta")
                .ok()?
                .call1((days as i64,))
                .ok()?;
            epoch.call_method1("__add__", (delta,)).ok()
        },
        #[cfg(feature = "dtype-datetime")]
        AnyValue::Datetime(value, time_unit, time_zone) => {
            let dt_mod = py.import("datetime").ok()?;
            let epoch = dt_mod
                .getattr("datetime")
                .ok()?
                // Unix epoch start
                .call1((1970i32, 1i32, 1i32))
                .ok()?;
            let micros: i64 = match time_unit {
                TimeUnit::Nanoseconds => value / 1000,
                TimeUnit::Microseconds => value,
                TimeUnit::Milliseconds => value * 1000,
            };
            let kwargs = PyDict::new(py);
            kwargs.set_item("microseconds", micros).ok()?;
            let delta = dt_mod
                .getattr("timedelta")
                .ok()?
                .call((), Some(&kwargs))
                .ok()?;
            let naive = epoch.call_method1("__add__", (delta,)).ok()?;
            if let Some(tz) = time_zone {
                // Datetimes have to get standardized to timezone aware
                let zi = py.import("zoneinfo").ok()?;
                let tz_obj = zi.getattr("ZoneInfo").ok()?.call1((tz.to_string(),)).ok()?;
                let kw = PyDict::new(py);
                kw.set_item("tzinfo", tz_obj).ok()?;
                naive.call_method("replace", (), Some(&kw)).ok()
            } else {
                Some(naive)
            }
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
        AExpr::Column(name) => {
            let name = sanitize(name)?;
            pc.call_method1("field", (name,)).ok()
        },
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

            let values_list = if let AExpr::Literal(lv) = expr_arena.get(rhs_node)
                && lv.get_datatype().is_list()
            {
                use polars_core::prelude::ExplodeOptions;

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
                    - if *nulls_equal {
                        0
                    } else {
                        haystack_series.null_count()
                    };

                if converted_len > LIST_ITEM_LIMIT {
                    return None;
                }

                // Fast path for empty lists
                if converted_len == 0 {
                    return pc.call_method1("scalar", (false,)).ok();
                }

                if !*nulls_equal {
                    haystack_series = haystack_series.drop_nulls();
                }

                series_to_py_list(py, &haystack_series)?
            } else {
                return None;
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
