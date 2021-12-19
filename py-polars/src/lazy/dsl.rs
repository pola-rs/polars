use super::apply::*;
use crate::conversion::{str_to_null_behavior, Wrap};
use crate::lazy::map_single;
use crate::lazy::utils::py_exprs_to_exprs;
use crate::prelude::{parse_strategy, str_to_rankmethod};
use crate::series::PySeries;
use crate::utils::{reinterpret, str_to_polarstype};
use polars::lazy::dsl;
use polars::lazy::dsl::Operator;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};
use pyo3::{class::basic::CompareOp, PyNumberProtocol, PyObjectProtocol};

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: dsl::Expr,
}

#[pyproto]
impl PyNumberProtocol for PyExpr {
    fn __add__(lhs: Self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(lhs.inner, Operator::Plus, rhs.inner).into())
    }
    fn __sub__(lhs: Self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(lhs.inner, Operator::Minus, rhs.inner).into())
    }
    fn __mul__(lhs: Self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(lhs.inner, Operator::Multiply, rhs.inner).into())
    }
    fn __truediv__(lhs: Self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(lhs.inner, Operator::TrueDivide, rhs.inner).into())
    }
    fn __mod__(lhs: Self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(lhs.inner, Operator::Modulus, rhs.inner).into())
    }
    fn __floordiv__(lhs: Self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(lhs.inner, Operator::Divide, rhs.inner).into())
    }
}

#[pyproto]
impl<'p> PyObjectProtocol<'p> for PyExpr {
    fn __richcmp__(&'p self, other: PyExpr, op: CompareOp) -> PyExpr {
        match op {
            CompareOp::Eq => self.eq(other),
            CompareOp::Ne => self.neq(other),
            CompareOp::Gt => self.gt(other),
            CompareOp::Lt => self.lt(other),
            CompareOp::Ge => self.gt_eq(other),
            CompareOp::Le => self.lt_eq(other),
        }
    }
}

#[pymethods]
impl PyExpr {
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
    pub fn cast(&self, data_type: &PyAny, strict: bool) -> PyExpr {
        let str_repr = data_type.str().unwrap().to_str().unwrap();
        let dt = str_to_polarstype(str_repr);
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
        self.clone()
            .inner
            .apply(
                |s| Ok(Series::new(s.name(), &[s.arg_max().map(|idx| idx as u32)])),
                GetOutput::from_type(DataType::UInt32),
            )
            .into()
    }
    pub fn arg_min(&self) -> PyExpr {
        self.clone()
            .inner
            .apply(
                |s| Ok(Series::new(s.name(), &[s.arg_min().map(|idx| idx as u32)])),
                GetOutput::from_type(DataType::UInt32),
            )
            .into()
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
        self.clone()
            .inner
            .apply(move |s| s.fill_null(strat), GetOutput::same_type())
            .into()
    }

    pub fn fill_nan(&self, expr: PyExpr) -> PyExpr {
        self.clone().inner.fill_nan(expr.inner).into()
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
            .into()
    }
    pub fn tail(&self, n: Option<usize>) -> PyExpr {
        self.clone().inner.tail(n).into()
    }

    pub fn head(&self, n: Option<usize>) -> PyExpr {
        self.clone().inner.head(n).into()
    }

    pub fn slice(&self, offset: i64, length: usize) -> PyExpr {
        self.clone().inner.slice(offset, length).into()
    }

    pub fn round(&self, decimals: u32) -> PyExpr {
        self.clone().inner.round(decimals).into()
    }

    pub fn floor(&self) -> PyExpr {
        self.clone().inner.floor().into()
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

    pub fn str_parse_date(&self, fmt: Option<String>) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            ca.as_date(fmt.as_deref()).map(|ca| ca.into_series())
        };
        self.clone()
            .inner
            .map(function, GetOutput::from_type(DataType::Date))
            .into()
    }

    pub fn str_parse_datetime(&self, fmt: Option<String>) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            ca.as_datetime(fmt.as_deref()).map(|ca| ca.into_series())
        };
        self.clone()
            .inner
            .map(function, GetOutput::from_type(DataType::Datetime))
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
            .into()
    }

    pub fn str_extract(&self, pat: String, group_index: usize) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.extract(&pat, group_index) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
            }
        };
        self.clone()
            .inner
            .map(function, GetOutput::from_type(DataType::Boolean))
            .into()
    }

    pub fn strftime(&self, fmt: String) -> PyExpr {
        let function = move |s: Series| s.strftime(&fmt);
        self.clone()
            .inner
            .map(function, GetOutput::from_type(DataType::Utf8))
            .into()
    }

    pub fn arr_lengths(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.list()?;
            Ok(ca.lst_lengths().into_series())
        };
        self.clone()
            .inner
            .map(function, GetOutput::from_type(DataType::UInt32))
            .into()
    }

    pub fn year(&self) -> PyExpr {
        self.clone().inner.year().into()
    }
    pub fn month(&self) -> PyExpr {
        self.clone().inner.month().into()
    }
    pub fn week(&self) -> PyExpr {
        self.clone().inner.week().into()
    }
    pub fn weekday(&self) -> PyExpr {
        self.clone().inner.weekday().into()
    }
    pub fn day(&self) -> PyExpr {
        self.clone().inner.day().into()
    }
    pub fn ordinal_day(&self) -> PyExpr {
        self.clone().inner.ordinal_day().into()
    }
    pub fn hour(&self) -> PyExpr {
        self.clone().inner.hour().into()
    }
    pub fn minute(&self) -> PyExpr {
        self.clone().inner.minute().into()
    }
    pub fn second(&self) -> PyExpr {
        self.clone().inner.second().into()
    }
    pub fn nanosecond(&self) -> PyExpr {
        self.clone().inner.nanosecond().into()
    }
    pub fn timestamp(&self) -> PyExpr {
        self.clone()
            .inner
            .map(
                |s| s.timestamp().map(|ca| ca.into_series()),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
    pub fn dt_epoch_seconds(&self) -> PyExpr {
        self.clone()
            .inner
            .map(
                |s| s.timestamp().map(|ca| (ca / 1000).into_series()),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }

    pub fn rolling_apply(&self, py: Python, window_size: usize, lambda: PyObject) -> PyExpr {
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
                    let is_float = obj.as_ref(py).is_instance::<PyFloat>().unwrap();

                    let dtype = s.dtype();

                    use DataType::*;
                    let result = match dtype {
                        UInt8 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(UInt8Chunked::new_from_slice("", &[v as u8]).into_series())
                            } else {
                                obj.extract::<u8>(py)
                                    .map(|v| UInt8Chunked::new_from_slice("", &[v]).into_series())
                            }
                        }
                        UInt16 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(UInt16Chunked::new_from_slice("", &[v as u16]).into_series())
                            } else {
                                obj.extract::<u16>(py)
                                    .map(|v| UInt16Chunked::new_from_slice("", &[v]).into_series())
                            }
                        }
                        UInt32 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(UInt32Chunked::new_from_slice("", &[v as u32]).into_series())
                            } else {
                                obj.extract::<u32>(py)
                                    .map(|v| UInt32Chunked::new_from_slice("", &[v]).into_series())
                            }
                        }
                        UInt64 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(UInt64Chunked::new_from_slice("", &[v as u64]).into_series())
                            } else {
                                obj.extract::<u64>(py)
                                    .map(|v| UInt64Chunked::new_from_slice("", &[v]).into_series())
                            }
                        }
                        Int8 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(Int8Chunked::new_from_slice("", &[v as i8]).into_series())
                            } else {
                                obj.extract::<i8>(py)
                                    .map(|v| Int8Chunked::new_from_slice("", &[v]).into_series())
                            }
                        }
                        Int16 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(Int16Chunked::new_from_slice("", &[v as i16]).into_series())
                            } else {
                                obj.extract::<i16>(py)
                                    .map(|v| Int16Chunked::new_from_slice("", &[v]).into_series())
                            }
                        }
                        Int32 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(Int32Chunked::new_from_slice("", &[v as i32]).into_series())
                            } else {
                                obj.extract::<i32>(py)
                                    .map(|v| Int32Chunked::new_from_slice("", &[v]).into_series())
                            }
                        }
                        Int64 => {
                            if is_float {
                                let v = obj.extract::<f64>(py).unwrap();
                                Ok(Int64Chunked::new_from_slice("", &[v as i64]).into_series())
                            } else {
                                obj.extract::<i64>(py)
                                    .map(|v| Int64Chunked::new_from_slice("", &[v]).into_series())
                            }
                        }
                        Float32 => obj
                            .extract::<f32>(py)
                            .map(|v| Float32Chunked::new_from_slice("", &[v]).into_series()),
                        Float64 => obj
                            .extract::<f64>(py)
                            .map(|v| Float64Chunked::new_from_slice("", &[v]).into_series()),
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
            .rolling_apply(window_size, Arc::new(function), GetOutput::same_type())
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

    pub fn rolling_median(&self, window_size: usize) -> Self {
        self.inner
            .clone()
            .rolling_apply_float(window_size, |ca| ChunkAgg::median(ca))
            .into()
    }

    pub fn rolling_quantile(&self, window_size: usize, quantile: f64, interpolation: &str) -> Self {
        let interpol = match interpolation {
            "nearest" => QuantileInterpolOptions::Nearest,
            "lower" => QuantileInterpolOptions::Lower,
            "higher" => QuantileInterpolOptions::Higher,
            "midpoint" => QuantileInterpolOptions::Midpoint,
            "linear" => QuantileInterpolOptions::Linear,
            _ => panic!("not supported"),
        };

        self.inner
            .clone()
            .rolling_apply_float(window_size, move |ca| {
                ChunkAgg::quantile(ca, quantile, interpol).unwrap()
            })
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
        self.inner
            .clone()
            .map(
                |s| Ok(s.list()?.lst_max()),
                GetOutput::map_field(|f| {
                    if let DataType::List(adt) = f.data_type() {
                        Field::new(f.name(), *adt.clone())
                    } else {
                        // inner type
                        f.clone()
                    }
                }),
            )
            .into()
    }

    fn lst_min(&self) -> Self {
        self.inner
            .clone()
            .map(
                |s| Ok(s.list()?.lst_min()),
                GetOutput::map_field(|f| {
                    if let DataType::List(adt) = f.data_type() {
                        Field::new(f.name(), *adt.clone())
                    } else {
                        // inner type
                        f.clone()
                    }
                }),
            )
            .into()
    }

    fn lst_sum(&self) -> Self {
        self.inner
            .clone()
            .map(
                |s| Ok(s.list()?.lst_sum()),
                GetOutput::map_field(|f| {
                    if let DataType::List(adt) = f.data_type() {
                        Field::new(f.name(), *adt.clone())
                    } else {
                        // inner type
                        f.clone()
                    }
                }),
            )
            .into()
    }

    fn lst_mean(&self) -> Self {
        self.inner
            .clone()
            .map(
                |s| Ok(s.list()?.lst_mean().into_series()),
                GetOutput::from_type(DataType::Float64),
            )
            .into()
    }

    fn lst_sort(&self, reverse: bool) -> Self {
        self.inner
            .clone()
            .map(
                move |s| Ok(s.list()?.lst_sort(reverse).into_series()),
                GetOutput::same_type(),
            )
            .into()
    }

    fn lst_reverse(&self) -> Self {
        self.inner
            .clone()
            .map(
                move |s| Ok(s.list()?.lst_reverse().into_series()),
                GetOutput::same_type(),
            )
            .into()
    }

    fn lst_unique(&self) -> Self {
        self.inner
            .clone()
            .map(
                move |s| Ok(s.list()?.lst_unique()?.into_series()),
                GetOutput::same_type(),
            )
            .into()
    }

    fn lst_get(&self, index: i64) -> Self {
        self.inner
            .clone()
            .map(
                move |s| s.list()?.lst_get(index),
                GetOutput::map_field(|field| match field.data_type() {
                    DataType::List(inner) => Field::new(field.name(), *inner.clone()),
                    _ => panic!("should be a list type"),
                }),
            )
            .into()
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

    fn skew(&self, bias: bool) -> Self {
        self.inner.clone().skew(bias).into()
    }
    fn kurtosis(&self, fisher: bool, bias: bool) -> Self {
        self.inner.clone().kurtosis(fisher, bias).into()
    }
    fn str_concat(&self, delimiter: &str) -> Self {
        self.inner.clone().str_concat(delimiter).into()
    }

    fn date_truncate(&self, every: &str, offset: &str) -> Self {
        let every = Duration::parse(every);
        let offset = Duration::parse(offset);
        self.inner
            .clone()
            .apply(
                move |s| match s.dtype() {
                    DataType::Datetime => {
                        Ok(s.datetime().unwrap().truncate(every, offset).into_series())
                    }
                    DataType::Date => Ok(s.date().unwrap().truncate(every, offset).into_series()),
                    dt => Err(PolarsError::ComputeError(
                        format!("expected date/datetime got {:?}", dt).into(),
                    )),
                },
                GetOutput::same_type(),
            )
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
            .into()
    }

    pub fn shuffle(&self, seed: u64) -> Self {
        self.inner.clone().shuffle(seed).into()
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
    if let Ok(true) = value.is_instance::<PyBool>() {
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

pub fn range(low: i64, high: i64, dtype: &PyAny) -> PyExpr {
    let str_repr = dtype.str().unwrap().to_str().unwrap();
    let dtype = str_to_polarstype(str_repr);
    match dtype {
        DataType::Int32 => dsl::range(low as i32, high as i32).into(),
        DataType::UInt32 => dsl::range(low as u32, high as u32).into(),
        _ => dsl::range(low, high).into(),
    }
}
