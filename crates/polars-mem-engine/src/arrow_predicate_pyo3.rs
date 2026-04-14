use polars_core::datatypes::TimeUnit;
use polars_io::arrow_predicate::{ArrowPredicate, ComparisonOp, LiteralValue};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// Convert an [`ArrowPredicate`] python PyArrow expression.
pub fn arrow_predicate_to_pyobject<'py>(
    py: Python<'py>,
    pred: &ArrowPredicate,
) -> PyResult<Bound<'py, PyAny>> {
    let pc = py.import("pyarrow.compute")?;
    build_expr(py, &pc, pred)
}

fn build_expr<'py>(
    py: Python<'py>,
    pc: &Bound<'py, PyAny>,
    p: &ArrowPredicate,
) -> PyResult<Bound<'py, PyAny>> {
    // Build the pyarrow expression based on ArrowPredicate `p`.
    match p {
        ArrowPredicate::Column(name) => pc.call_method1("field", (name.as_str(),)),
        ArrowPredicate::Literal(lv) => {
            let val = literal_to_py(py, lv)?;
            pc.call_method1("scalar", (val,))
        },
        ArrowPredicate::Comparison { left, op, right } => {
            let l = build_expr(py, pc, left)?;
            let r = build_expr(py, pc, right)?;
            let method = match op {
                ComparisonOp::Eq => "__eq__",
                ComparisonOp::NotEq => "__ne__",
                ComparisonOp::Lt => "__lt__",
                ComparisonOp::Lte => "__le__",
                ComparisonOp::Gt => "__gt__",
                ComparisonOp::Gte => "__ge__",
            };
            l.call_method1(method, (r,))
        },
        ArrowPredicate::And(l, r) => {
            let l = build_expr(py, pc, l)?;
            let r = build_expr(py, pc, r)?;
            l.call_method1("__and__", (r,))
        },
        ArrowPredicate::Or(l, r) => {
            let l = build_expr(py, pc, l)?;
            let r = build_expr(py, pc, r)?;
            l.call_method1("__or__", (r,))
        },
        ArrowPredicate::Xor(l, r) => {
            let l = build_expr(py, pc, l)?;
            let r = build_expr(py, pc, r)?;
            l.call_method1("__xor__", (r,))
        },
        ArrowPredicate::Not(inner) => {
            let i = build_expr(py, pc, inner)?;
            i.call_method0("__invert__")
        },
        ArrowPredicate::IsNull(inner) => {
            let i = build_expr(py, pc, inner)?;
            i.call_method0("is_null")
        },
        ArrowPredicate::IsIn { expr, values } => {
            if values.is_empty() {
                return pc.call_method1("scalar", (false,));
            }
            let e = build_expr(py, pc, expr)?;
            let py_values: Vec<Py<PyAny>> = values
                .iter()
                .map(|v| literal_to_py(py, v).map(|b| b.unbind()))
                .collect::<PyResult<_>>()?;
            let list = PyList::new(py, &py_values)?;
            e.call_method1("isin", (list,))
        },
    }
}

fn literal_to_py<'py>(py: Python<'py>, lv: &LiteralValue) -> PyResult<Bound<'py, PyAny>> {
    match lv {
        LiteralValue::Null => Ok(py.None().into_bound(py)),
        LiteralValue::Int(v) => Ok(v.into_pyobject(py)?.into_any()),
        LiteralValue::Float(v) => Ok(v.into_pyobject(py)?.into_any()),
        LiteralValue::String(s) => Ok(s.into_pyobject(py)?.into_any()),
        LiteralValue::Bool(v) => v.into_bound_py_any(py),
        LiteralValue::Date(days) => {
            let dt_mod = py.import("datetime")?;
            let epoch = dt_mod.getattr("date")?.call1((1970i32, 1i32, 1i32))?;
            let delta = dt_mod.getattr("timedelta")?.call1((*days as i64,))?;
            epoch.call_method1("__add__", (delta,))
        },
        LiteralValue::Datetime {
            value,
            time_unit,
            time_zone,
        } => {
            // This conversion does not feel idiomatic but it's probably the best way to do this
            // within the confines of the python standard lib.
            let dt_mod = py.import("datetime")?;
            let epoch = dt_mod.getattr("datetime")?.call1((1970i32, 1i32, 1i32))?;
            let micros: i64 = match time_unit {
                TimeUnit::Nanoseconds => value / 1000,
                TimeUnit::Microseconds => *value,
                TimeUnit::Milliseconds => value * 1000,
            };
            let kwargs = PyDict::new(py);
            kwargs.set_item("microseconds", micros)?;
            let delta = dt_mod.getattr("timedelta")?.call((), Some(&kwargs))?;
            let naive = epoch.call_method1("__add__", (delta,))?;
            if let Some(tz) = time_zone {
                let zi = py.import("zoneinfo")?;
                let tz_obj = zi.getattr("ZoneInfo")?.call1((tz.to_string(),))?;
                let kw = PyDict::new(py);
                kw.set_item("tzinfo", tz_obj)?;
                naive.call_method("replace", (), Some(&kw))
            } else {
                Ok(naive)
            }
        },
    }
}
