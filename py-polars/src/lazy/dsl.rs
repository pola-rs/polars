use super::apply::*;
use crate::conversion::{str_to_null_behavior, Wrap};
use crate::lazy::map_single;
use crate::lazy::utils::py_exprs_to_exprs;
use crate::prelude::{parse_strategy, str_to_rankmethod};
use crate::series::PySeries;
use crate::utils::reinterpret;
use polars::lazy::dsl;
use polars::lazy::dsl::Operator;
use polars::prelude::*;
use polars_core::prelude::QuantileInterpolOptions;
use pyo3::class::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};
use std::borrow::Cow;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: dsl::Expr,
}

pub(crate) trait ToExprs {
    fn to_exprs(self) -> Vec<Expr>;
}

impl ToExprs for Vec<PyExpr> {
    fn to_exprs(self) -> Vec<Expr> {
        // Safety
        // repr is transparent
        // and has only got one inner field`
        unsafe { std::mem::transmute(self) }
    }
}

#[pymethods]
impl PyExpr {
    fn __richcmp__(&self, other: Self, op: CompareOp) -> PyExpr {
        match op {
            CompareOp::Eq => self.eq(other),
            CompareOp::Ne => self.neq(other),
            CompareOp::Gt => self.gt(other),
            CompareOp::Lt => self.lt(other),
            CompareOp::Ge => self.gt_eq(other),
            CompareOp::Le => self.lt_eq(other),
        }
    }

    fn __add__(&self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Plus, rhs.inner).into())
    }
    fn __sub__(&self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Minus, rhs.inner).into())
    }
    fn __mul__(&self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Multiply, rhs.inner).into())
    }
    fn __truediv__(&self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::TrueDivide, rhs.inner).into())
    }
    fn __mod__(&self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Modulus, rhs.inner).into())
    }
    fn __floordiv__(&self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Divide, rhs.inner).into())
    }

    pub fn to_str(&self) -> String {
        format!("{:?}", self.inner)
    }
    pub fn eq(&self, other: PyExpr) -> PyExpr {
        self.clone().inner.eq(other.inner).into()
    }
    pub fn neq(&self, other: PyExpr) -> PyExpr {
        self.clone().inner.neq(other.inner).into()
    }
    pub fn gt(&self, other: PyExpr) -> PyExpr {
        self.clone().inner.gt(other.inner).into()
    }
    pub fn gt_eq(&self, other: PyExpr) -> PyExpr {
        self.clone().inner.gt_eq(other.inner).into()
    }
    pub fn lt_eq(&self, other: PyExpr) -> PyExpr {
        self.clone().inner.lt_eq(other.inner).into()
    }
    pub fn lt(&self, other: PyExpr) -> PyExpr {
        self.clone().inner.lt(other.inner).into()
    }
    pub fn alias(&self, name: &str) -> PyExpr {
        self.clone().inner.alias(name).into()
    }
    pub fn is_not(&self) -> PyExpr {
        self.clone().inner.not().into()
    }
    pub fn is_null(&self) -> PyExpr {
        self.clone().inner.is_null().into()
    }
    pub fn is_not_null(&self) -> PyExpr {
        self.clone().inner.is_not_null().into()
    }

    pub fn is_infinite(&self) -> PyExpr {
        self.clone().inner.is_infinite().into()
    }

    pub fn is_finite(&self) -> PyExpr {
        self.clone().inner.is_finite().into()
    }

    pub fn is_nan(&self) -> PyExpr {
        self.clone().inner.is_nan().into()
    }

    pub fn is_not_nan(&self) -> PyExpr {
        self.clone().inner.is_not_nan().into()
    }

    pub fn min(&self) -> PyExpr {
        self.clone().inner.min().into()
    }
    pub fn max(&self) -> PyExpr {
        self.clone().inner.max().into()
    }
    pub fn mean(&self) -> PyExpr {
        self.clone().inner.mean().into()
    }
    pub fn median(&self) -> PyExpr {
        self.clone().inner.median().into()
    }
    pub fn sum(&self) -> PyExpr {
        self.clone().inner.sum().into()
    }
    pub fn n_unique(&self) -> PyExpr {
        self.clone().inner.n_unique().into()
    }
    pub fn arg_unique(&self) -> PyExpr {
        self.clone().inner.arg_unique().into()
    }
    pub fn unique(&self) -> PyExpr {
        self.clone().inner.unique().into()
    }
    pub fn unique_stable(&self) -> PyExpr {
        self.clone().inner.unique_stable().into()
    }
    pub fn first(&self) -> PyExpr {
        self.clone().inner.first().into()
    }
    pub fn last(&self) -> PyExpr {
        self.clone().inner.last().into()
    }
    pub fn list(&self) -> PyExpr {
        self.clone().inner.list().into()
    }
    pub fn quantile(&self, quantile: f64, interpolation: &str) -> PyExpr {
        let interpol = match interpolation {
            "nearest" => QuantileInterpolOptions::Nearest,
            "lower" => QuantileInterpolOptions::Lower,
            "higher" => QuantileInterpolOptions::Higher,
            "midpoint" => QuantileInterpolOptions::Midpoint,
            "linear" => QuantileInterpolOptions::Linear,
            _ => panic!("not supported"),
        };
        self.clone().inner.quantile(quantile, interpol).into()
    }
    pub fn agg_groups(&self) -> PyExpr {
        self.clone().inner.agg_groups().into()
    }
    pub fn count(&self) -> PyExpr {
        self.clone().inner.count().into()
    }
    pub fn value_counts(&self, multithreaded: bool) -> PyExpr {
        self.inner.clone().value_counts(multithreaded).into()
    }
    pub fn unique_counts(&self) -> PyExpr {
        self.inner.clone().unique_counts().into()
    }
    pub fn cast(&self, data_type: Wrap<DataType>, strict: bool) -> PyExpr {
        let dt = data_type.0;
        let expr = if strict {
            self.inner.clone().strict_cast(dt)
        } else {
            self.inner.clone().cast(dt)
        };
        expr.into()
    }
    pub fn sort_with(&self, descending: bool, nulls_last: bool) -> PyExpr {
        self.clone()
            .inner
            .sort_with(SortOptions {
                descending,
                nulls_last,
            })
            .into()
    }

    pub fn arg_sort(&self, reverse: bool) -> PyExpr {
        self.clone().inner.arg_sort(reverse).into()
    }
    pub fn arg_max(&self) -> PyExpr {
        self.clone().inner.arg_max().into()
    }
    pub fn arg_min(&self) -> PyExpr {
        self.clone().inner.arg_min().into()
    }
    pub fn take(&self, idx: PyExpr) -> PyExpr {
        self.clone().inner.take(idx.inner).into()
    }

    pub fn sort_by(&self, by: Vec<PyExpr>, reverse: Vec<bool>) -> PyExpr {
        let by = by.into_iter().map(|e| e.inner).collect::<Vec<_>>();
        self.clone().inner.sort_by(by, reverse).into()
    }

    pub fn backward_fill(&self) -> PyExpr {
        self.clone().inner.backward_fill().into()
    }

    pub fn forward_fill(&self) -> PyExpr {
        self.clone().inner.forward_fill().into()
    }

    pub fn shift(&self, periods: i64) -> PyExpr {
        self.clone().inner.shift(periods).into()
    }
    pub fn shift_and_fill(&self, periods: i64, fill_value: PyExpr) -> PyExpr {
        self.clone()
            .inner
            .shift_and_fill(periods, fill_value.inner)
            .into()
    }

    pub fn fill_null(&self, expr: PyExpr) -> PyExpr {
        self.clone().inner.fill_null(expr.inner).into()
    }

    pub fn fill_null_with_strategy(&self, strategy: &str) -> PyExpr {
        let strat = parse_strategy(strategy);
        self.inner
            .clone()
            .apply(move |s| s.fill_null(strat), GetOutput::same_type())
            .with_fmt("fill_null")
            .into()
    }

    pub fn fill_nan(&self, expr: PyExpr) -> PyExpr {
        self.inner.clone().fill_nan(expr.inner).into()
    }

    pub fn drop_nulls(&self) -> PyExpr {
        self.inner.clone().drop_nulls().into()
    }

    pub fn drop_nans(&self) -> PyExpr {
        self.inner.clone().drop_nans().into()
    }

    pub fn filter(&self, predicate: PyExpr) -> PyExpr {
        self.clone().inner.filter(predicate.inner).into()
    }
    pub fn reverse(&self) -> PyExpr {
        self.clone().inner.reverse().into()
    }
    pub fn std(&self) -> PyExpr {
        self.clone().inner.std().into()
    }
    pub fn var(&self) -> PyExpr {
        self.clone().inner.var().into()
    }
    pub fn is_unique(&self) -> PyExpr {
        self.clone().inner.is_unique().into()
    }

    pub fn is_first(&self) -> PyExpr {
        self.clone().inner.is_first().into()
    }

    pub fn explode(&self) -> PyExpr {
        self.clone().inner.explode().into()
    }

    pub fn take_every(&self, n: usize) -> PyExpr {
        self.clone()
            .inner
            .map(move |s: Series| Ok(s.take_every(n)), GetOutput::same_type())
            .with_fmt("take_every")
            .into()
    }
    pub fn tail(&self, n: Option<usize>) -> PyExpr {
        self.clone().inner.tail(n).into()
    }

    pub fn head(&self, n: Option<usize>) -> PyExpr {
        self.clone().inner.head(n).into()
    }

    pub fn slice(&self, offset: PyExpr, length: PyExpr) -> PyExpr {
        self.inner.clone().slice(offset.inner, length.inner).into()
    }

    pub fn round(&self, decimals: u32) -> PyExpr {
        self.clone().inner.round(decimals).into()
    }

    pub fn floor(&self) -> PyExpr {
        self.clone().inner.floor().into()
    }

    pub fn ceil(&self) -> PyExpr {
        self.clone().inner.ceil().into()
    }

    pub fn clip(&self, min: f64, max: f64) -> PyExpr {
        self.clone().inner.clip(min, max).into()
    }

    pub fn abs(&self) -> PyExpr {
        self.clone().inner.abs().into()
    }

    pub fn is_duplicated(&self) -> PyExpr {
        self.clone().inner.is_duplicated().into()
    }

    pub fn over(&self, partition_by: Vec<PyExpr>) -> PyExpr {
        let partition_by = partition_by
            .into_iter()
            .map(|e| e.inner)
            .collect::<Vec<Expr>>();
        self.clone().inner.over(partition_by).into()
    }

    pub fn _and(&self, expr: PyExpr) -> PyExpr {
        self.clone().inner.and(expr.inner).into()
    }

    pub fn _xor(&self, expr: PyExpr) -> PyExpr {
        self.clone().inner.xor(expr.inner).into()
    }

    pub fn _or(&self, expr: PyExpr) -> PyExpr {
        self.clone().inner.or(expr.inner).into()
    }
    #[cfg(feature = "is_in")]
    pub fn is_in(&self, expr: PyExpr) -> PyExpr {
        self.clone().inner.is_in(expr.inner).into()
    }

    pub fn repeat_by(&self, by: PyExpr) -> PyExpr {
        self.clone().inner.repeat_by(by.inner).into()
    }

    pub fn pow(&self, exponent: f64) -> PyExpr {
        self.clone().inner.pow(exponent).into()
    }

    pub fn cumsum(&self, reverse: bool) -> PyExpr {
        self.clone().inner.cumsum(reverse).into()
    }
    pub fn cummax(&self, reverse: bool) -> PyExpr {
        self.clone().inner.cummax(reverse).into()
    }
    pub fn cummin(&self, reverse: bool) -> PyExpr {
        self.clone().inner.cummin(reverse).into()
    }

    pub fn cumprod(&self, reverse: bool) -> PyExpr {
        self.clone().inner.cumprod(reverse).into()
    }

    pub fn product(&self) -> PyExpr {
        self.clone().inner.product().into()
    }

    pub fn str_parse_date(&self, fmt: Option<String>, strict: bool, exact: bool) -> PyExpr {
        self.inner
            .clone()
            .str()
            .strptime(StrpTimeOptions {
                date_dtype: DataType::Date,
                fmt,
                strict,
                exact,
            })
            .into()
    }

    pub fn str_parse_datetime(&self, fmt: Option<String>, strict: bool, exact: bool) -> PyExpr {
        self.inner
            .clone()
            .str()
            .strptime(StrpTimeOptions {
                date_dtype: DataType::Datetime(TimeUnit::Microseconds, None),
                fmt,
                strict,
                exact,
            })
            .into()
    }

    pub fn str_strip(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;

            Ok(ca.apply(|s| Cow::Borrowed(s.trim())).into_series())
        };
        self.clone()
            .inner
            .map(function, GetOutput::same_type())
            .with_fmt("str.strip")
            .into()
    }

    pub fn str_rstrip(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;

            Ok(ca.apply(|s| Cow::Borrowed(s.trim_end())).into_series())
        };
        self.clone()
            .inner
            .map(function, GetOutput::same_type())
            .with_fmt("str.rstrip")
            .into()
    }

    pub fn str_lstrip(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;

            Ok(ca.apply(|s| Cow::Borrowed(s.trim_start())).into_series())
        };
        self.clone()
            .inner
            .map(function, GetOutput::same_type())
            .with_fmt("str.lstrip")
            .into()
    }

    pub fn str_to_uppercase(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;
            Ok(ca.to_uppercase().into_series())
        };
        self.clone()
            .inner
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("str.to_uppercase")
            .into()
    }

    pub fn str_slice(&self, start: i64, length: Option<u64>) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            Ok(ca.str_slice(start, length)?.into_series())
        };
        self.clone()
            .inner
            .map(function, GetOutput::from_type(DataType::Utf8))
            .with_fmt("str.slice")
            .into()
    }

    pub fn str_to_lowercase(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;
            Ok(ca.to_lowercase().into_series())
        };
        self.clone()
            .inner
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("str.to_lowercase")
            .into()
    }

    pub fn str_lengths(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;
            Ok(ca.str_lengths().into_series())
        };
        self.clone()
            .inner
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("str.len")
            .into()
    }

    pub fn str_replace(&self, pat: String, val: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.replace(&pat, &val) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
            }
        };
        self.clone()
            .inner
            .map(function, GetOutput::same_type())
            .with_fmt("str.replace")
            .into()
    }

    pub fn str_replace_all(&self, pat: String, val: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.replace_all(&pat, &val) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
            }
        };
        self.clone()
            .inner
            .map(function, GetOutput::same_type())
            .with_fmt("str.replace_all")
            .into()
    }

    pub fn str_contains(&self, pat: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.contains(&pat) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
            }
        };
        self.clone()
            .inner
            .map(function, GetOutput::from_type(DataType::Boolean))
            .with_fmt("str.contains")
            .into()
    }
    pub fn str_hex_encode(&self) -> PyExpr {
        self.clone()
            .inner
            .map(
                move |s| s.utf8().map(|s| s.hex_encode().into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("str.hex_encode")
            .into()
    }
    pub fn str_hex_decode(&self, strict: Option<bool>) -> PyExpr {
        self.clone()
            .inner
            .map(
                move |s| s.utf8()?.hex_decode(strict).map(|s| s.into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("str.hex_decode")
            .into()
    }
    pub fn str_base64_encode(&self) -> PyExpr {
        self.clone()
            .inner
            .map(
                move |s| s.utf8().map(|s| s.base64_encode().into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("str.base64_encode")
            .into()
    }

    pub fn str_base64_decode(&self, strict: Option<bool>) -> PyExpr {
        self.clone()
            .inner
            .map(
                move |s| s.utf8()?.base64_decode(strict).map(|s| s.into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("str.base64_decode")
            .into()
    }
    pub fn str_json_path_match(&self, pat: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.json_path_match(&pat) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
            }
        };
        self.clone()
            .inner
            .map(function, GetOutput::from_type(DataType::Boolean))
            .with_fmt("str.json_path_match")
            .into()
    }

    pub fn str_extract(&self, pat: &str, group_index: usize) -> PyExpr {
        self.inner.clone().str().extract(pat, group_index).into()
    }

    pub fn strftime(&self, fmt: &str) -> PyExpr {
        self.inner.clone().dt().strftime(fmt).into()
    }
    pub fn str_split(&self, by: &str) -> PyExpr {
        self.inner.clone().str().split(by).into()
    }
    pub fn str_split_inclusive(&self, by: &str) -> PyExpr {
        self.inner.clone().str().split_inclusive(by).into()
    }

    pub fn str_split_exact(&self, by: &str, n: usize) -> PyExpr {
        self.inner.clone().str().split_exact(by, n).into()
    }
    pub fn str_split_exact_inclusive(&self, by: &str, n: usize) -> PyExpr {
        self.inner.clone().str().split_exact_inclusive(by, n).into()
    }

    pub fn arr_lengths(&self) -> PyExpr {
        self.inner.clone().arr().lengths().into()
    }

    pub fn year(&self) -> PyExpr {
        self.clone().inner.dt().year().into()
    }
    pub fn month(&self) -> PyExpr {
        self.clone().inner.dt().month().into()
    }
    pub fn week(&self) -> PyExpr {
        self.clone().inner.dt().week().into()
    }
    pub fn weekday(&self) -> PyExpr {
        self.clone().inner.dt().weekday().into()
    }
    pub fn day(&self) -> PyExpr {
        self.clone().inner.dt().day().into()
    }
    pub fn ordinal_day(&self) -> PyExpr {
        self.clone().inner.dt().ordinal_day().into()
    }
    pub fn hour(&self) -> PyExpr {
        self.clone().inner.dt().hour().into()
    }
    pub fn minute(&self) -> PyExpr {
        self.clone().inner.dt().minute().into()
    }
    pub fn second(&self) -> PyExpr {
        self.clone().inner.dt().second().into()
    }
    pub fn nanosecond(&self) -> PyExpr {
        self.clone().inner.dt().nanosecond().into()
    }
    pub fn duration_days(&self) -> PyExpr {
        self.inner
            .clone()
            .map(
                |s| Ok(s.duration()?.days().into_series()),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
    pub fn duration_hours(&self) -> PyExpr {
        self.inner
            .clone()
            .map(
                |s| Ok(s.duration()?.hours().into_series()),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
    pub fn duration_seconds(&self) -> PyExpr {
        self.inner
            .clone()
            .map(
                |s| Ok(s.duration()?.seconds().into_series()),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
    pub fn duration_nanoseconds(&self) -> PyExpr {
        self.inner
            .clone()
            .map(
                |s| Ok(s.duration()?.nanoseconds().into_series()),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
    pub fn duration_milliseconds(&self) -> PyExpr {
        self.inner
            .clone()
            .map(
                |s| Ok(s.duration()?.milliseconds().into_series()),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
    pub fn timestamp(&self, tu: Wrap<TimeUnit>) -> PyExpr {
        self.inner.clone().dt().timestamp(tu.0).into()
    }
    pub fn dt_epoch_seconds(&self) -> PyExpr {
        self.clone()
            .inner
            .map(
                |s| {
                    s.timestamp(TimeUnit::Milliseconds)
                        .map(|ca| (ca / 1000).into_series())
                },
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }

    pub fn dt_with_time_unit(&self, tu: Wrap<TimeUnit>) -> PyExpr {
        self.inner.clone().dt().with_time_unit(tu.0).into()
    }

    pub fn dt_cast_time_unit(&self, tu: Wrap<TimeUnit>) -> PyExpr {
        self.inner.clone().dt().cast_time_unit(tu.0).into()
    }

    pub fn rolling_apply(
        &self,
        py: Python,
        lambda: PyObject,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> PyExpr {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };
        // get the pypolars module
        // do the import outside of the function.
        let pypolars = PyModule::import(py, "polars").unwrap().to_object(py);

        let function = move |s: &Series| {
            let gil = Python::acquire_gil();
            let py = gil.python();

            let out = call_lambda_with_series(py, s.clone(), &lambda, &pypolars);
            match out.getattr(py, "_s") {
                Ok(pyseries) => {
                    let pyseries = pyseries.extract::<PySeries>(py).unwrap();
                    pyseries.series
                }
                Err(_) => {
                    let obj = out;
                    let is_float = obj.as_ref(py).is_instance_of::<PyFloat>().unwrap();

                    let dtype = s.dtype();

                    use DataType::*;
                    let result = match dtype {
                        UInt8 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(UInt8Chunked::from_slice("", &[v as u8]).into_series())
                            } else {
                                obj.extract::<u8>(py)
                                    .map(|v| UInt8Chunked::from_slice("", &[v]).into_series())
                            }
                        }
                        UInt16 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(UInt16Chunked::from_slice("", &[v as u16]).into_series())
                            } else {
                                obj.extract::<u16>(py)
                                    .map(|v| UInt16Chunked::from_slice("", &[v]).into_series())
                            }
                        }
                        UInt32 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(UInt32Chunked::from_slice("", &[v as u32]).into_series())
                            } else {
                                obj.extract::<u32>(py)
                                    .map(|v| UInt32Chunked::from_slice("", &[v]).into_series())
                            }
                        }
                        UInt64 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(UInt64Chunked::from_slice("", &[v as u64]).into_series())
                            } else {
                                obj.extract::<u64>(py)
                                    .map(|v| UInt64Chunked::from_slice("", &[v]).into_series())
                            }
                        }
                        Int8 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(Int8Chunked::from_slice("", &[v as i8]).into_series())
                            } else {
                                obj.extract::<i8>(py)
                                    .map(|v| Int8Chunked::from_slice("", &[v]).into_series())
                            }
                        }
                        Int16 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(Int16Chunked::from_slice("", &[v as i16]).into_series())
                            } else {
                                obj.extract::<i16>(py)
                                    .map(|v| Int16Chunked::from_slice("", &[v]).into_series())
                            }
                        }
                        Int32 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(Int32Chunked::from_slice("", &[v as i32]).into_series())
                            } else {
                                obj.extract::<i32>(py)
                                    .map(|v| Int32Chunked::from_slice("", &[v]).into_series())
                            }
                        }
                        Int64 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(Int64Chunked::from_slice("", &[v as i64]).into_series())
                            } else {
                                obj.extract::<i64>(py)
                                    .map(|v| Int64Chunked::from_slice("", &[v]).into_series())
                            }
                        }
                        Float32 => obj
                            .extract::<f32>(py)
                            .map(|v| Float32Chunked::from_slice("", &[v]).into_series()),
                        Float64 => obj
                            .extract::<f64>(py)
                            .map(|v| Float64Chunked::from_slice("", &[v]).into_series()),
                        dt => panic!("{:?} not implemented", dt),
                    };

                    match result {
                        Ok(s) => s,
                        Err(e) => {
                            panic!("{:?}", e)
                        }
                    }
                }
            }
        };
        self.clone()
            .inner
            .rolling_apply(Arc::new(function), GetOutput::same_type(), options)
            .with_fmt("rolling_apply")
            .into()
    }

    pub fn map(&self, py: Python, lambda: PyObject, output_type: &PyAny, agg_list: bool) -> PyExpr {
        map_single(self, py, lambda, output_type, agg_list)
    }

    pub fn dot(&self, other: PyExpr) -> PyExpr {
        self.inner.clone().dot(other.inner).into()
    }
    pub fn hash(&self, k0: u64, k1: u64, k2: u64, k3: u64) -> PyExpr {
        let function = move |s: Series| {
            let hb = ahash::RandomState::with_seeds(k0, k1, k2, k3);
            Ok(s.hash(hb).into_series())
        };
        self.clone()
            .inner
            .map(function, GetOutput::from_type(DataType::UInt64))
            .into()
    }

    pub fn reinterpret(&self, signed: bool) -> PyExpr {
        let function = move |s: Series| reinterpret(&s, signed);
        let dt = if signed {
            DataType::Int64
        } else {
            DataType::UInt64
        };
        self.clone()
            .inner
            .map(function, GetOutput::from_type(dt))
            .into()
    }
    pub fn mode(&self) -> PyExpr {
        self.inner.clone().mode().into()
    }
    pub fn keep_name(&self) -> PyExpr {
        self.inner.clone().keep_name().into()
    }
    pub fn prefix(&self, prefix: &str) -> PyExpr {
        self.inner.clone().prefix(prefix).into()
    }
    pub fn suffix(&self, suffix: &str) -> PyExpr {
        self.inner.clone().suffix(suffix).into()
    }
    pub fn map_alias(&self, lambda: PyObject) -> PyExpr {
        self.inner
            .clone()
            .map_alias(move |name| {
                let gil = Python::acquire_gil();
                let py = gil.python();
                let out = lambda.call1(py, (name,)).unwrap();
                out.to_string()
            })
            .into()
    }
    pub fn exclude(&self, columns: Vec<String>) -> PyExpr {
        self.inner.clone().exclude(&columns).into()
    }
    pub fn exclude_dtype(&self, dtypes: Vec<Wrap<DataType>>) -> PyExpr {
        // Safety:
        // Wrap is transparent.
        let dtypes: Vec<DataType> = unsafe { std::mem::transmute(dtypes) };
        self.inner.clone().exclude_dtype(&dtypes).into()
    }
    pub fn interpolate(&self) -> PyExpr {
        self.inner.clone().interpolate().into()
    }

    pub fn rolling_sum(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> PyExpr {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };
        self.inner.clone().rolling_sum(options).into()
    }
    pub fn rolling_min(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> Self {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };
        self.inner.clone().rolling_min(options).into()
    }
    pub fn rolling_max(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> Self {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };
        self.inner.clone().rolling_max(options).into()
    }
    pub fn rolling_mean(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> Self {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };

        self.inner.clone().rolling_mean(options).into()
    }

    pub fn rolling_std(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> Self {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };

        self.inner.clone().rolling_std(options).into()
    }

    pub fn rolling_var(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> Self {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };

        self.inner.clone().rolling_var(options).into()
    }

    pub fn rolling_median(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> Self {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };
        self.inner.clone().rolling_median(options).into()
    }

    pub fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: &str,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> Self {
        let interpol = match interpolation {
            "nearest" => QuantileInterpolOptions::Nearest,
            "lower" => QuantileInterpolOptions::Lower,
            "higher" => QuantileInterpolOptions::Higher,
            "midpoint" => QuantileInterpolOptions::Midpoint,
            "linear" => QuantileInterpolOptions::Linear,
            _ => panic!("not supported"),
        };

        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };

        self.inner
            .clone()
            .rolling_quantile(quantile, interpol, options)
            .into()
    }

    pub fn rolling_skew(&self, window_size: usize, bias: bool) -> Self {
        self.inner
            .clone()
            .rolling_apply_float(window_size, move |ca| {
                ca.clone().into_series().skew(bias).unwrap()
            })
            .into()
    }

    pub fn lower_bound(&self) -> Self {
        self.inner.clone().lower_bound().into()
    }

    pub fn upper_bound(&self) -> Self {
        self.inner.clone().upper_bound().into()
    }

    fn lst_max(&self) -> Self {
        self.inner.clone().arr().max().into()
    }

    fn lst_min(&self) -> Self {
        self.inner.clone().arr().min().into()
    }

    fn lst_sum(&self) -> Self {
        self.inner.clone().arr().sum().with_fmt("arr.sum").into()
    }

    fn lst_mean(&self) -> Self {
        self.inner.clone().arr().mean().with_fmt("arr.mean").into()
    }

    fn lst_sort(&self, reverse: bool) -> Self {
        self.inner
            .clone()
            .arr()
            .sort(reverse)
            .with_fmt("arr.sort")
            .into()
    }

    fn lst_reverse(&self) -> Self {
        self.inner
            .clone()
            .arr()
            .reverse()
            .with_fmt("arr.reverse")
            .into()
    }

    fn lst_unique(&self) -> Self {
        self.inner
            .clone()
            .arr()
            .unique()
            .with_fmt("arr.unique")
            .into()
    }

    fn lst_get(&self, index: i64) -> Self {
        self.inner.clone().arr().get(index).into()
    }

    fn lst_join(&self, separator: &str) -> Self {
        self.inner.clone().arr().join(separator).into()
    }

    fn lst_arg_min(&self) -> Self {
        self.inner.clone().arr().arg_min().into()
    }

    fn lst_arg_max(&self) -> Self {
        self.inner.clone().arr().arg_max().into()
    }

    fn lst_diff(&self, n: usize, null_behavior: &str) -> PyResult<Self> {
        let null_behavior = str_to_null_behavior(null_behavior)?;
        Ok(self.inner.clone().arr().diff(n, null_behavior).into())
    }

    fn lst_shift(&self, periods: i64) -> Self {
        self.inner.clone().arr().shift(periods).into()
    }

    fn lst_slice(&self, offset: i64, length: usize) -> Self {
        self.inner.clone().arr().slice(offset, length).into()
    }

    fn rank(&self, method: &str, reverse: bool) -> Self {
        let method = str_to_rankmethod(method).unwrap();
        let options = RankOptions {
            method,
            descending: reverse,
        };
        self.inner.clone().rank(options).into()
    }

    fn diff(&self, n: usize, null_behavior: &str) -> Self {
        let null_behavior = str_to_null_behavior(null_behavior).unwrap();
        self.inner.clone().diff(n, null_behavior).into()
    }

    fn pct_change(&self, n: usize) -> Self {
        self.inner.clone().pct_change(n).into()
    }

    fn skew(&self, bias: bool) -> Self {
        self.inner.clone().skew(bias).into()
    }
    fn kurtosis(&self, fisher: bool, bias: bool) -> Self {
        self.inner.clone().kurtosis(fisher, bias).into()
    }
    fn str_concat(&self, delimiter: &str) -> Self {
        self.inner.clone().str().concat(delimiter).into()
    }

    fn cat_set_ordering(&self, ordering: &str) -> Self {
        let ordering = match ordering {
            "physical" => CategoricalOrdering::Physical,
            "lexical" => CategoricalOrdering::Lexical,
            _ => panic!("expected one of {{'physical', 'lexical'}}"),
        };

        self.inner.clone().cat().set_ordering(ordering).into()
    }

    fn date_truncate(&self, every: &str, offset: &str) -> Self {
        let every = Duration::parse(every);
        let offset = Duration::parse(offset);
        self.inner
            .clone()
            .apply(
                move |s| match s.dtype() {
                    DataType::Datetime(_, _) => {
                        Ok(s.datetime().unwrap().truncate(every, offset).into_series())
                    }
                    DataType::Date => Ok(s.date().unwrap().truncate(every, offset).into_series()),
                    dt => Err(PolarsError::ComputeError(
                        format!("expected date/datetime got {:?}", dt).into(),
                    )),
                },
                GetOutput::same_type(),
            )
            .with_fmt("dt.truncate")
            .into()
    }

    pub fn reshape(&self, dims: Vec<i64>) -> Self {
        self.inner.clone().reshape(&dims).into()
    }

    pub fn cumcount(&self, reverse: bool) -> Self {
        self.inner.clone().cumcount(reverse).into()
    }

    pub fn to_physical(&self) -> Self {
        self.inner
            .clone()
            .map(
                |s| Ok(s.to_physical_repr().into_owned()),
                GetOutput::map_dtype(|dt| dt.to_physical()),
            )
            .with_fmt("to_physical")
            .into()
    }

    pub fn shuffle(&self, seed: u64) -> Self {
        self.inner.clone().shuffle(seed).into()
    }

    pub fn sample_frac(&self, frac: f64, with_replacement: bool, seed: u64) -> Self {
        self.inner
            .clone()
            .sample_frac(frac, with_replacement, seed)
            .into()
    }

    pub fn ewm_mean(&self, alpha: f64, adjust: bool, min_periods: usize) -> Self {
        let options = EWMOptions {
            alpha,
            adjust,
            min_periods,
        };
        self.inner.clone().ewm_mean(options).into()
    }
    pub fn ewm_std(&self, alpha: f64, adjust: bool, min_periods: usize) -> Self {
        let options = EWMOptions {
            alpha,
            adjust,
            min_periods,
        };
        self.inner.clone().ewm_std(options).into()
    }
    pub fn ewm_var(&self, alpha: f64, adjust: bool, min_periods: usize) -> Self {
        let options = EWMOptions {
            alpha,
            adjust,
            min_periods,
        };
        self.inner.clone().ewm_var(options).into()
    }
    pub fn extend_constant(&self, py: Python, value: Wrap<AnyValue>, n: usize) -> Self {
        let value = value.into_py(py);
        self.inner
            .clone()
            .apply(
                move |s| {
                    let gil = Python::acquire_gil();
                    let py = gil.python();
                    let value = value.extract::<Wrap<AnyValue>>(py).unwrap().0;
                    s.extend_constant(value, n)
                },
                GetOutput::same_type(),
            )
            .with_fmt("extend")
            .into()
    }
    pub fn any(&self) -> Self {
        self.inner.clone().any().into()
    }

    pub fn all(&self) -> Self {
        self.inner.clone().all().into()
    }

    pub fn struct_field_by_name(&self, name: &str) -> PyExpr {
        self.inner.clone().struct_().field_by_name(name).into()
    }

    pub fn struct_rename_fields(&self, names: Vec<String>) -> PyExpr {
        self.inner.clone().struct_().rename_fields(names).into()
    }

    pub fn log(&self, base: f64) -> Self {
        self.inner.clone().log(base).into()
    }

    pub fn entropy(&self, base: f64) -> Self {
        self.inner.clone().entropy(base).into()
    }
}

impl From<dsl::Expr> for PyExpr {
    fn from(expr: dsl::Expr) -> Self {
        PyExpr { inner: expr }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct When {
    predicate: PyExpr,
}

#[pyclass]
#[derive(Clone)]
pub struct WhenThen {
    predicate: PyExpr,
    then: PyExpr,
}

#[pyclass]
#[derive(Clone)]
pub struct WhenThenThen {
    inner: dsl::WhenThenThen,
}

#[pymethods]
impl When {
    pub fn then(&self, expr: PyExpr) -> WhenThen {
        WhenThen {
            predicate: self.predicate.clone(),
            then: expr,
        }
    }
}

#[pymethods]
impl WhenThen {
    pub fn when(&self, predicate: PyExpr) -> WhenThenThen {
        let e = dsl::when(self.predicate.inner.clone())
            .then(self.then.inner.clone())
            .when(predicate.inner);
        WhenThenThen { inner: e }
    }

    pub fn otherwise(&self, expr: PyExpr) -> PyExpr {
        dsl::ternary_expr(
            self.predicate.inner.clone(),
            self.then.inner.clone(),
            expr.inner,
        )
        .into()
    }
}

#[pymethods]
impl WhenThenThen {
    pub fn when(&self, predicate: PyExpr) -> WhenThenThen {
        Self {
            inner: self.inner.clone().when(predicate.inner),
        }
    }
    pub fn then(&self, expr: PyExpr) -> WhenThenThen {
        Self {
            inner: self.inner.clone().then(expr.inner),
        }
    }
    pub fn otherwise(&self, expr: PyExpr) -> PyExpr {
        self.inner.clone().otherwise(expr.inner).into()
    }
}

pub fn when(predicate: PyExpr) -> When {
    When { predicate }
}

pub fn col(name: &str) -> PyExpr {
    dsl::col(name).into()
}

pub fn count() -> PyExpr {
    dsl::count().into()
}

pub fn first() -> PyExpr {
    dsl::first().into()
}

pub fn last() -> PyExpr {
    dsl::last().into()
}

pub fn cols(names: Vec<String>) -> PyExpr {
    dsl::cols(names).into()
}

pub fn dtype_cols(dtypes: Vec<DataType>) -> PyExpr {
    dsl::dtype_cols(dtypes).into()
}

pub fn binary_expr(l: PyExpr, op: u8, r: PyExpr) -> PyExpr {
    let left = l.inner;
    let right = r.inner;

    let op = match op {
        0 => Operator::Eq,
        1 => Operator::NotEq,
        2 => Operator::Lt,
        3 => Operator::LtEq,
        4 => Operator::Gt,
        5 => Operator::GtEq,
        6 => Operator::Plus,
        7 => Operator::Minus,
        8 => Operator::Multiply,
        9 => Operator::Divide,
        10 => Operator::Modulus,
        11 => Operator::And,
        12 => Operator::Or,
        _ => panic!("not an operator"),
    };

    dsl::binary_expr(left, op, right).into()
}

pub fn fold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = py_exprs_to_exprs(exprs);

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    polars::lazy::dsl::fold_exprs(acc.inner, func, exprs).into()
}

pub fn lit(value: &PyAny) -> PyExpr {
    if let Ok(true) = value.is_instance_of::<PyBool>() {
        let val = value.extract::<bool>().unwrap();
        dsl::lit(val).into()
    } else if let Ok(int) = value.downcast::<PyInt>() {
        let val = int.extract::<i64>().unwrap();

        if val > 0 && val < i32::MAX as i64 || val < 0 && val > i32::MIN as i64 {
            dsl::lit(val as i32).into()
        } else {
            dsl::lit(val).into()
        }
    } else if let Ok(float) = value.downcast::<PyFloat>() {
        let val = float.extract::<f64>().unwrap();
        dsl::lit(val).into()
    } else if let Ok(pystr) = value.downcast::<PyString>() {
        dsl::lit(
            pystr
                .to_str()
                .expect("could not transform Python string to Rust Unicode"),
        )
        .into()
    } else if let Ok(series) = value.extract::<PySeries>() {
        dsl::lit(series.series).into()
    } else if value.is_none() {
        dsl::lit(Null {}).into()
    } else {
        panic!("could not convert value {:?} as a Literal", value)
    }
}

pub fn range(low: i64, high: i64, dtype: Wrap<DataType>) -> PyExpr {
    match dtype.0 {
        DataType::Int32 => dsl::range(low as i32, high as i32).into(),
        DataType::UInt32 => dsl::range(low as u32, high as u32).into(),
        _ => dsl::range(low, high).into(),
    }
}
