use polars::lazy::dsl;
use polars::lazy::dsl::Expr;
use polars::prelude::*;
use polars_core::datatypes::TimeZone;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyString};

use crate::apply::lazy::binary_lambda;
use crate::conversion::{get_lf, Wrap};
use crate::expr::ToExprs;
use crate::prelude::{
    vec_extract_wrapped, ClosedWindow, DataType, DatetimeArgs, Duration, DurationArgs, ObjectValue,
};
use crate::{apply, PyDataFrame, PyExpr, PyLazyFrame, PyPolarsErr, PySeries};

macro_rules! set_unwrapped_or_0 {
    ($($var:ident),+ $(,)?) => {
        $(let $var = $var.map(|e| e.inner).unwrap_or(dsl::lit(0));)+
    };
}

#[pyfunction]
pub fn arange(start: PyExpr, end: PyExpr, step: i64) -> PyExpr {
    dsl::arange(start.inner, end.inner, step).into()
}

#[pyfunction]
pub fn rolling_corr(
    x: PyExpr,
    y: PyExpr,
    window_size: IdxSize,
    min_periods: IdxSize,
    ddof: u8,
) -> PyExpr {
    dsl::rolling_corr(
        x.inner,
        y.inner,
        RollingCovOptions {
            min_periods,
            window_size,
            ddof,
        },
    )
    .into()
}

#[pyfunction]
pub fn rolling_cov(
    x: PyExpr,
    y: PyExpr,
    window_size: IdxSize,
    min_periods: IdxSize,
    ddof: u8,
) -> PyExpr {
    dsl::rolling_cov(
        x.inner,
        y.inner,
        RollingCovOptions {
            min_periods,
            window_size,
            ddof,
        },
    )
    .into()
}

#[pyfunction]
pub fn arg_sort_by(by: Vec<PyExpr>, descending: Vec<bool>) -> PyExpr {
    let by = by.into_iter().map(|e| e.inner).collect::<Vec<Expr>>();
    dsl::arg_sort_by(by, &descending).into()
}

#[pyfunction]
pub fn arg_where(condition: PyExpr) -> PyExpr {
    dsl::arg_where(condition.inner).into()
}

#[pyfunction]
pub fn as_struct(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::as_struct(&exprs).into()
}

#[pyfunction]
pub fn coalesce(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::coalesce(&exprs).into()
}

#[pyfunction]
pub fn col(name: &str) -> PyExpr {
    dsl::col(name).into()
}

#[pyfunction]
pub fn collect_all(lfs: Vec<PyLazyFrame>, py: Python) -> PyResult<Vec<PyDataFrame>> {
    use polars_core::utils::rayon::prelude::*;

    let out = py.allow_threads(|| {
        polars_core::POOL.install(|| {
            lfs.par_iter()
                .map(|lf| {
                    let df = lf.ldf.clone().collect()?;
                    Ok(PyDataFrame::new(df))
                })
                .collect::<polars_core::error::PolarsResult<Vec<_>>>()
                .map_err(PyPolarsErr::from)
        })
    });

    Ok(out?)
}

#[pyfunction]
pub fn cols(names: Vec<String>) -> PyExpr {
    dsl::cols(names).into()
}

#[pyfunction]
pub fn concat_lf(
    seq: &PyAny,
    rechunk: bool,
    parallel: bool,
    to_supertypes: bool,
) -> PyResult<PyLazyFrame> {
    let len = seq.len()?;
    let mut lfs = Vec::with_capacity(len);

    for res in seq.iter()? {
        let item = res?;
        let lf = get_lf(item)?;
        lfs.push(lf);
    }

    let lf = dsl::concat(
        lfs,
        UnionArgs {
            rechunk,
            parallel,
            to_supertypes,
        },
    )
    .map_err(PyPolarsErr::from)?;
    Ok(lf.into())
}

#[pyfunction]
pub fn concat_list(s: Vec<PyExpr>) -> PyResult<PyExpr> {
    let s = s.into_iter().map(|e| e.inner).collect::<Vec<_>>();
    let expr = dsl::concat_list(s).map_err(PyPolarsErr::from)?;
    Ok(expr.into())
}

#[pyfunction]
pub fn concat_str(s: Vec<PyExpr>, separator: &str) -> PyExpr {
    let s = s.into_iter().map(|e| e.inner).collect::<Vec<_>>();
    dsl::concat_str(s, separator).into()
}

#[pyfunction]
pub fn count() -> PyExpr {
    dsl::count().into()
}

#[pyfunction]
pub fn cov(a: PyExpr, b: PyExpr) -> PyExpr {
    dsl::cov(a.inner, b.inner).into()
}

#[pyfunction]
pub fn cumfold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>, include_init: bool) -> PyExpr {
    let exprs = exprs.to_exprs();

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    dsl::cumfold_exprs(acc.inner, func, exprs, include_init).into()
}

#[pyfunction]
pub fn cumreduce(lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    dsl::cumreduce_exprs(func, exprs).into()
}

#[pyfunction]
pub fn date_range_lazy(
    start: PyExpr,
    end: PyExpr,
    every: &str,
    closed: Wrap<ClosedWindow>,
    time_zone: Option<TimeZone>,
) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::parse(every);
    dsl::functions::date_range(start, end, every, closed.0, time_zone).into()
}

#[pyfunction]
pub fn datetime(
    year: PyExpr,
    month: PyExpr,
    day: PyExpr,
    hour: Option<PyExpr>,
    minute: Option<PyExpr>,
    second: Option<PyExpr>,
    microsecond: Option<PyExpr>,
) -> PyExpr {
    let year = year.inner;
    let month = month.inner;
    let day = day.inner;

    set_unwrapped_or_0!(hour, minute, second, microsecond);

    let args = DatetimeArgs {
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
    };
    dsl::datetime(args).into()
}

#[pyfunction]
pub fn diag_concat_lf(lfs: &PyAny, rechunk: bool, parallel: bool) -> PyResult<PyLazyFrame> {
    let iter = lfs.iter()?;

    let lfs = iter
        .map(|item| {
            let item = item?;
            get_lf(item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let lf = dsl::functions::diag_concat_lf(lfs, rechunk, parallel).map_err(PyPolarsErr::from)?;
    Ok(lf.into())
}

#[pyfunction]
pub fn concat_expr(e: Vec<PyExpr>, rechunk: bool) -> PyResult<PyExpr> {
    let e = e.to_exprs();
    let e = dsl::functions::concat_expr(e, rechunk).map_err(PyPolarsErr::from)?;
    Ok(e.into())
}

#[pyfunction]
pub fn dtype_cols(dtypes: Vec<Wrap<DataType>>) -> PyResult<PyExpr> {
    let dtypes = vec_extract_wrapped(dtypes);
    Ok(dsl::dtype_cols(dtypes).into())
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn duration(
    days: Option<PyExpr>,
    seconds: Option<PyExpr>,
    nanoseconds: Option<PyExpr>,
    microseconds: Option<PyExpr>,
    milliseconds: Option<PyExpr>,
    minutes: Option<PyExpr>,
    hours: Option<PyExpr>,
    weeks: Option<PyExpr>,
) -> PyExpr {
    set_unwrapped_or_0!(
        days,
        seconds,
        nanoseconds,
        microseconds,
        milliseconds,
        minutes,
        hours,
        weeks,
    );
    let args = DurationArgs {
        days,
        seconds,
        nanoseconds,
        microseconds,
        milliseconds,
        minutes,
        hours,
        weeks,
    };
    dsl::duration(args).into()
}

#[pyfunction]
pub fn first() -> PyExpr {
    dsl::first().into()
}

#[pyfunction]
pub fn fold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    dsl::fold_exprs(acc.inner, func, exprs).into()
}

#[pyfunction]
pub fn last() -> PyExpr {
    dsl::last().into()
}

#[pyfunction]
pub fn lit(value: &PyAny, allow_object: bool) -> PyResult<PyExpr> {
    if value.is_instance_of::<PyBool>() {
        let val = value.extract::<bool>().unwrap();
        Ok(dsl::lit(val).into())
    } else if let Ok(int) = value.downcast::<PyInt>() {
        match int.extract::<i64>() {
            Ok(val) => {
                if val >= 0 && val < i32::MAX as i64 || val <= 0 && val > i32::MIN as i64 {
                    Ok(dsl::lit(val as i32).into())
                } else {
                    Ok(dsl::lit(val).into())
                }
            }
            _ => {
                let val = int.extract::<u64>().unwrap();
                Ok(dsl::lit(val).into())
            }
        }
    } else if let Ok(float) = value.downcast::<PyFloat>() {
        let val = float.extract::<f64>().unwrap();
        Ok(dsl::lit(val).into())
    } else if let Ok(pystr) = value.downcast::<PyString>() {
        Ok(dsl::lit(
            pystr
                .to_str()
                .expect("could not transform Python string to Rust Unicode"),
        )
        .into())
    } else if let Ok(series) = value.extract::<PySeries>() {
        Ok(dsl::lit(series.series).into())
    } else if value.is_none() {
        Ok(dsl::lit(Null {}).into())
    } else if let Ok(value) = value.downcast::<PyBytes>() {
        Ok(dsl::lit(value.as_bytes()).into())
    } else if allow_object {
        let s = Python::with_gil(|py| {
            PySeries::new_object("", vec![ObjectValue::from(value.into_py(py))], false).series
        });
        Ok(dsl::lit(s).into())
    } else {
        Err(PyValueError::new_err(format!(
            "could not convert value {:?} as a Literal",
            value.str()?
        )))
    }
}

#[pyfunction]
#[pyo3(signature = (pyexpr, lambda, output_type, apply_groups, returns_scalar))]
pub fn map_mul(
    py: Python,
    pyexpr: Vec<PyExpr>,
    lambda: PyObject,
    output_type: Option<Wrap<DataType>>,
    apply_groups: bool,
    returns_scalar: bool,
) -> PyExpr {
    apply::lazy::map_mul(
        &pyexpr,
        py,
        lambda,
        output_type,
        apply_groups,
        returns_scalar,
    )
}

#[pyfunction]
pub fn max_exprs(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::max_exprs(exprs).into()
}

#[pyfunction]
pub fn min_exprs(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::min_exprs(exprs).into()
}

#[pyfunction]
pub fn pearson_corr(a: PyExpr, b: PyExpr, ddof: u8) -> PyExpr {
    dsl::pearson_corr(a.inner, b.inner, ddof).into()
}

#[pyfunction]
pub fn reduce(lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    dsl::reduce_exprs(func, exprs).into()
}

#[pyfunction]
pub fn repeat(value: Wrap<AnyValue>, n: PyExpr, dtype: Option<Wrap<DataType>>) -> PyResult<PyExpr> {
    let value = value.0;
    let n = n.inner;
    let dtype = dtype.map(|wrap| wrap.0);

    let target_dtype = match dtype {
        Some(dtype) => dtype,
        None => match value.dtype() {
            // Integer inputs that fit in Int32 are parsed as such
            DataType::Int64 => {
                let int_value: i64 = value.try_extract().unwrap();
                if int_value >= i32::MIN as i64 && int_value <= i32::MAX as i64 {
                    DataType::Int32
                } else {
                    DataType::Int64
                }
            }
            DataType::Unknown => DataType::Null,
            _ => value.dtype(),
        },
    };

    let lit_value = LiteralValue::try_from(value).map_err(PyPolarsErr::from)?;
    let must_cast = lit_value.get_datatype() != target_dtype;

    let mut expr = dsl::repeat(lit_value, n);

    if must_cast {
        expr = expr.cast(target_dtype);
    }

    Ok(expr.into())
}

#[pyfunction]
pub fn spearman_rank_corr(a: PyExpr, b: PyExpr, ddof: u8, propagate_nans: bool) -> PyExpr {
    #[cfg(feature = "propagate_nans")]
    {
        dsl::spearman_rank_corr(a.inner, b.inner, ddof, propagate_nans).into()
    }
    #[cfg(not(feature = "propagate_nans"))]
    {
        panic!("activate 'propagate_nans'")
    }
}

#[pyfunction]
pub fn sum_exprs(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::sum_exprs(exprs).into()
}

#[pyfunction]
pub fn time_range_lazy(
    start: PyExpr,
    end: PyExpr,
    every: &str,
    closed: Wrap<ClosedWindow>,
) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::parse(every);
    dsl::functions::time_range(start, end, every, closed.0).into()
}

#[pyfunction]
#[cfg(feature = "sql")]
pub fn sql_expr(sql: &str) -> PyResult<PyExpr> {
    let expr = polars::sql::sql_expr(sql).map_err(PyPolarsErr::from)?;
    Ok(expr.into())
}
