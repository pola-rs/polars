use std::any::Any;

use polars::lazy::dsl;
use polars::prelude::*;
use polars::series::ops::NullBehavior;
use polars_core::prelude::QuantileInterpolOptions;
use polars_core::series::IsSorted;
use pyo3::class::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFloat};

use crate::apply::lazy::{call_lambda_with_series, map_single};
use crate::conversion::{parse_fill_null_strategy, Wrap};
use crate::error::PyPolarsErr;
use crate::series::PySeries;
use crate::utils::reinterpret;
use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn __richcmp__(&self, other: Self, op: CompareOp) -> Self {
        match op {
            CompareOp::Eq => self.eq(other),
            CompareOp::Ne => self.neq(other),
            CompareOp::Gt => self.gt(other),
            CompareOp::Lt => self.lt(other),
            CompareOp::Ge => self.gt_eq(other),
            CompareOp::Le => self.lt_eq(other),
        }
    }

    fn __add__(&self, rhs: Self) -> PyResult<Self> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Plus, rhs.inner).into())
    }
    fn __sub__(&self, rhs: Self) -> PyResult<Self> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Minus, rhs.inner).into())
    }
    fn __mul__(&self, rhs: Self) -> PyResult<Self> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Multiply, rhs.inner).into())
    }
    fn __truediv__(&self, rhs: Self) -> PyResult<Self> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::TrueDivide, rhs.inner).into())
    }
    fn __mod__(&self, rhs: Self) -> PyResult<Self> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Modulus, rhs.inner).into())
    }
    fn __floordiv__(&self, rhs: Self) -> PyResult<Self> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::FloorDivide, rhs.inner).into())
    }

    fn to_str(&self) -> String {
        format!("{:?}", self.inner)
    }
    fn eq(&self, other: Self) -> Self {
        self.clone().inner.eq(other.inner).into()
    }

    fn eq_missing(&self, other: Self) -> Self {
        self.clone().inner.eq_missing(other.inner).into()
    }
    fn neq(&self, other: Self) -> Self {
        self.clone().inner.neq(other.inner).into()
    }
    fn neq_missing(&self, other: Self) -> Self {
        self.clone().inner.neq_missing(other.inner).into()
    }
    fn gt(&self, other: Self) -> Self {
        self.clone().inner.gt(other.inner).into()
    }
    fn gt_eq(&self, other: Self) -> Self {
        self.clone().inner.gt_eq(other.inner).into()
    }
    fn lt_eq(&self, other: Self) -> Self {
        self.clone().inner.lt_eq(other.inner).into()
    }
    fn lt(&self, other: Self) -> Self {
        self.clone().inner.lt(other.inner).into()
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        // Used in pickle/pickling
        let mut writer: Vec<u8> = vec![];
        ciborium::ser::into_writer(&self.inner, &mut writer)
            .map_err(|e| PyPolarsErr::Other(format!("{}", e)))?;

        Ok(PyBytes::new(py, &writer).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Used in pickle/pickling
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.inner = ciborium::de::from_reader(s.as_bytes())
                    .map_err(|e| PyPolarsErr::Other(format!("{}", e)))?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn alias(&self, name: &str) -> Self {
        self.clone().inner.alias(name).into()
    }
    fn is_not(&self) -> Self {
        self.clone().inner.not().into()
    }
    fn is_null(&self) -> Self {
        self.clone().inner.is_null().into()
    }
    fn is_not_null(&self) -> Self {
        self.clone().inner.is_not_null().into()
    }

    fn is_infinite(&self) -> Self {
        self.clone().inner.is_infinite().into()
    }

    fn is_finite(&self) -> Self {
        self.clone().inner.is_finite().into()
    }

    fn is_nan(&self) -> Self {
        self.clone().inner.is_nan().into()
    }

    fn is_not_nan(&self) -> Self {
        self.clone().inner.is_not_nan().into()
    }

    fn min(&self) -> Self {
        self.clone().inner.min().into()
    }
    fn max(&self) -> Self {
        self.clone().inner.max().into()
    }
    #[cfg(feature = "propagate_nans")]
    fn nan_max(&self) -> Self {
        self.clone().inner.nan_max().into()
    }
    #[cfg(feature = "propagate_nans")]
    fn nan_min(&self) -> Self {
        self.clone().inner.nan_min().into()
    }
    fn mean(&self) -> Self {
        self.clone().inner.mean().into()
    }
    fn median(&self) -> Self {
        self.clone().inner.median().into()
    }
    fn sum(&self) -> Self {
        self.clone().inner.sum().into()
    }
    fn n_unique(&self) -> Self {
        self.clone().inner.n_unique().into()
    }
    fn arg_unique(&self) -> Self {
        self.clone().inner.arg_unique().into()
    }
    fn unique(&self) -> Self {
        self.clone().inner.unique().into()
    }
    fn unique_stable(&self) -> Self {
        self.clone().inner.unique_stable().into()
    }
    fn first(&self) -> Self {
        self.clone().inner.first().into()
    }
    fn last(&self) -> Self {
        self.clone().inner.last().into()
    }
    fn implode(&self) -> Self {
        self.clone().inner.implode().into()
    }
    fn quantile(&self, quantile: Self, interpolation: Wrap<QuantileInterpolOptions>) -> Self {
        self.clone()
            .inner
            .quantile(quantile.inner, interpolation.0)
            .into()
    }
    fn agg_groups(&self) -> Self {
        self.clone().inner.agg_groups().into()
    }
    fn count(&self) -> Self {
        self.clone().inner.count().into()
    }
    fn value_counts(&self, multithreaded: bool, sorted: bool) -> Self {
        self.inner
            .clone()
            .value_counts(multithreaded, sorted)
            .into()
    }
    fn unique_counts(&self) -> Self {
        self.inner.clone().unique_counts().into()
    }
    fn null_count(&self) -> Self {
        self.inner.clone().null_count().into()
    }
    fn cast(&self, data_type: Wrap<DataType>, strict: bool) -> Self {
        let dt = data_type.0;
        let expr = if strict {
            self.inner.clone().strict_cast(dt)
        } else {
            self.inner.clone().cast(dt)
        };
        expr.into()
    }
    fn sort_with(&self, descending: bool, nulls_last: bool) -> Self {
        self.clone()
            .inner
            .sort_with(SortOptions {
                descending,
                nulls_last,
                multithreaded: true,
            })
            .into()
    }

    fn arg_sort(&self, descending: bool, nulls_last: bool) -> Self {
        self.clone()
            .inner
            .arg_sort(SortOptions {
                descending,
                nulls_last,
                multithreaded: true,
            })
            .into()
    }

    #[cfg(feature = "top_k")]
    fn top_k(&self, k: usize) -> Self {
        self.inner.clone().top_k(k).into()
    }

    #[cfg(feature = "top_k")]
    fn bottom_k(&self, k: usize) -> Self {
        self.inner.clone().bottom_k(k).into()
    }

    fn arg_max(&self) -> Self {
        self.clone().inner.arg_max().into()
    }
    fn arg_min(&self) -> Self {
        self.clone().inner.arg_min().into()
    }

    #[cfg(feature = "search_sorted")]
    fn search_sorted(&self, element: Self, side: Wrap<SearchSortedSide>) -> Self {
        self.inner
            .clone()
            .search_sorted(element.inner, side.0)
            .into()
    }
    fn take(&self, idx: Self) -> Self {
        self.clone().inner.take(idx.inner).into()
    }

    fn sort_by(&self, by: Vec<Self>, descending: Vec<bool>) -> Self {
        let by = by.into_iter().map(|e| e.inner).collect::<Vec<_>>();
        self.clone().inner.sort_by(by, descending).into()
    }

    fn backward_fill(&self, limit: FillNullLimit) -> Self {
        self.clone().inner.backward_fill(limit).into()
    }

    fn forward_fill(&self, limit: FillNullLimit) -> Self {
        self.clone().inner.forward_fill(limit).into()
    }

    fn shift(&self, periods: i64) -> Self {
        self.clone().inner.shift(periods).into()
    }
    fn shift_and_fill(&self, periods: i64, fill_value: Self) -> Self {
        self.clone()
            .inner
            .shift_and_fill(periods, fill_value.inner)
            .into()
    }

    fn fill_null(&self, expr: Self) -> Self {
        self.clone().inner.fill_null(expr.inner).into()
    }

    fn fill_null_with_strategy(&self, strategy: &str, limit: FillNullLimit) -> PyResult<Self> {
        let strat = parse_fill_null_strategy(strategy, limit)?;
        Ok(self
            .inner
            .clone()
            .apply(
                move |s| s.fill_null(strat).map(Some),
                GetOutput::same_type(),
            )
            .with_fmt("fill_null_with_strategy")
            .into())
    }

    fn fill_nan(&self, expr: Self) -> Self {
        self.inner.clone().fill_nan(expr.inner).into()
    }

    fn drop_nulls(&self) -> Self {
        self.inner.clone().drop_nulls().into()
    }

    fn drop_nans(&self) -> Self {
        self.inner.clone().drop_nans().into()
    }

    fn filter(&self, predicate: Self) -> Self {
        self.clone().inner.filter(predicate.inner).into()
    }
    fn reverse(&self) -> Self {
        self.clone().inner.reverse().into()
    }
    fn std(&self, ddof: u8) -> Self {
        self.clone().inner.std(ddof).into()
    }
    fn var(&self, ddof: u8) -> Self {
        self.clone().inner.var(ddof).into()
    }
    fn is_unique(&self) -> Self {
        self.clone().inner.is_unique().into()
    }

    fn approx_unique(&self) -> Self {
        self.clone().inner.approx_unique().into()
    }

    fn is_first(&self) -> Self {
        self.clone().inner.is_first().into()
    }

    fn explode(&self) -> Self {
        self.clone().inner.explode().into()
    }

    fn take_every(&self, n: usize) -> Self {
        self.clone()
            .inner
            .map(
                move |s: Series| Ok(Some(s.take_every(n))),
                GetOutput::same_type(),
            )
            .with_fmt("take_every")
            .into()
    }
    fn tail(&self, n: usize) -> Self {
        self.clone().inner.tail(Some(n)).into()
    }

    fn head(&self, n: usize) -> Self {
        self.clone().inner.head(Some(n)).into()
    }

    fn slice(&self, offset: Self, length: Self) -> Self {
        self.inner.clone().slice(offset.inner, length.inner).into()
    }

    fn append(&self, other: Self, upcast: bool) -> Self {
        self.inner.clone().append(other.inner, upcast).into()
    }

    fn rechunk(&self) -> Self {
        self.inner
            .clone()
            .map(|s| Ok(Some(s.rechunk())), GetOutput::same_type())
            .into()
    }

    fn round(&self, decimals: u32) -> Self {
        self.clone().inner.round(decimals).into()
    }

    fn floor(&self) -> Self {
        self.clone().inner.floor().into()
    }

    fn ceil(&self) -> Self {
        self.clone().inner.ceil().into()
    }

    fn clip(&self, py: Python, min: PyObject, max: PyObject) -> Self {
        let min = min.extract::<Wrap<AnyValue>>(py).unwrap().0;
        let max = max.extract::<Wrap<AnyValue>>(py).unwrap().0;
        self.clone().inner.clip(min, max).into()
    }

    fn clip_min(&self, py: Python, min: PyObject) -> Self {
        let min = min.extract::<Wrap<AnyValue>>(py).unwrap().0;
        self.clone().inner.clip_min(min).into()
    }

    fn clip_max(&self, py: Python, max: PyObject) -> Self {
        let max = max.extract::<Wrap<AnyValue>>(py).unwrap().0;
        self.clone().inner.clip_max(max).into()
    }

    fn abs(&self) -> Self {
        self.clone().inner.abs().into()
    }

    #[cfg(feature = "trigonometry")]
    fn sin(&self) -> Self {
        self.clone().inner.sin().into()
    }

    #[cfg(feature = "trigonometry")]
    fn cos(&self) -> Self {
        self.clone().inner.cos().into()
    }

    #[cfg(feature = "trigonometry")]
    fn tan(&self) -> Self {
        self.clone().inner.tan().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arcsin(&self) -> Self {
        self.clone().inner.arcsin().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arccos(&self) -> Self {
        self.clone().inner.arccos().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arctan(&self) -> Self {
        self.clone().inner.arctan().into()
    }

    #[cfg(feature = "trigonometry")]
    fn sinh(&self) -> Self {
        self.clone().inner.sinh().into()
    }

    #[cfg(feature = "trigonometry")]
    fn cosh(&self) -> Self {
        self.clone().inner.cosh().into()
    }

    #[cfg(feature = "trigonometry")]
    fn tanh(&self) -> Self {
        self.clone().inner.tanh().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arcsinh(&self) -> Self {
        self.clone().inner.arcsinh().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arccosh(&self) -> Self {
        self.clone().inner.arccosh().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arctanh(&self) -> Self {
        self.clone().inner.arctanh().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn degrees(&self) -> Self {
        self.clone().inner.degrees().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn radians(&self) -> Self {
        self.clone().inner.radians().into()
    }

    #[cfg(feature = "sign")]
    fn sign(&self) -> Self {
        self.clone().inner.sign().into()
    }

    fn is_duplicated(&self) -> Self {
        self.clone().inner.is_duplicated().into()
    }

    fn over(&self, partition_by: Vec<Self>, mapping: Wrap<WindowMapping>) -> Self {
        let partition_by = partition_by
            .into_iter()
            .map(|e| e.inner)
            .collect::<Vec<Expr>>();
        self.clone()
            .inner
            .over_with_options(partition_by, WindowOptions { mapping: mapping.0 })
            .into()
    }

    fn _and(&self, expr: Self) -> Self {
        self.clone().inner.and(expr.inner).into()
    }

    fn _xor(&self, expr: Self) -> Self {
        self.clone().inner.xor(expr.inner).into()
    }

    fn _or(&self, expr: Self) -> Self {
        self.clone().inner.or(expr.inner).into()
    }
    #[cfg(feature = "is_in")]
    fn is_in(&self, expr: Self) -> Self {
        self.clone().inner.is_in(expr.inner).into()
    }

    #[cfg(feature = "repeat_by")]
    fn repeat_by(&self, by: Self) -> Self {
        self.clone().inner.repeat_by(by.inner).into()
    }

    fn pow(&self, exponent: Self) -> Self {
        self.clone().inner.pow(exponent.inner).into()
    }

    fn cumsum(&self, reverse: bool) -> Self {
        self.clone().inner.cumsum(reverse).into()
    }
    fn cummax(&self, reverse: bool) -> Self {
        self.clone().inner.cummax(reverse).into()
    }
    fn cummin(&self, reverse: bool) -> Self {
        self.clone().inner.cummin(reverse).into()
    }

    fn cumprod(&self, reverse: bool) -> Self {
        self.clone().inner.cumprod(reverse).into()
    }

    fn product(&self) -> Self {
        self.clone().inner.product().into()
    }

    fn shrink_dtype(&self) -> Self {
        self.inner.clone().shrink_dtype().into()
    }

    #[pyo3(signature = (lambda, window_size, weights, min_periods, center))]
    fn rolling_apply(
        &self,
        lambda: PyObject,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> Self {
        let options = RollingOptionsFixedWindow {
            window_size,
            weights,
            min_periods,
            center,
            ..Default::default()
        };
        let function = move |s: &Series| {
            Python::with_gil(|py| {
                let out = call_lambda_with_series(py, s.clone(), &lambda)
                    .expect("python function failed");
                match out.getattr(py, "_s") {
                    Ok(pyseries) => {
                        let pyseries = pyseries.extract::<PySeries>(py).unwrap();
                        pyseries.series
                    }
                    Err(_) => {
                        let obj = out;
                        let is_float = obj.as_ref(py).is_instance_of::<PyFloat>();

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
                            dt => panic!("{dt:?} not implemented"),
                        };

                        match result {
                            Ok(s) => s,
                            Err(e) => {
                                panic!("{e:?}")
                            }
                        }
                    }
                }
            })
        };
        self.clone()
            .inner
            .rolling_apply(Arc::new(function), GetOutput::same_type(), options)
            .with_fmt("rolling_apply")
            .into()
    }

    #[pyo3(signature = (lambda, output_type, agg_list))]
    fn map(&self, lambda: PyObject, output_type: Option<Wrap<DataType>>, agg_list: bool) -> Self {
        map_single(self, lambda, output_type, agg_list)
    }

    fn dot(&self, other: Self) -> Self {
        self.inner.clone().dot(other.inner).into()
    }

    fn reinterpret(&self, signed: bool) -> Self {
        let function = move |s: Series| reinterpret(&s, signed).map(Some);
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
    fn mode(&self) -> Self {
        self.inner.clone().mode().into()
    }
    fn keep_name(&self) -> Self {
        self.inner.clone().keep_name().into()
    }
    fn prefix(&self, prefix: &str) -> Self {
        self.inner.clone().prefix(prefix).into()
    }
    fn suffix(&self, suffix: &str) -> Self {
        self.inner.clone().suffix(suffix).into()
    }
    fn map_alias(&self, lambda: PyObject) -> Self {
        self.inner
            .clone()
            .map_alias(move |name| {
                let out = Python::with_gil(|py| lambda.call1(py, (name,)));
                match out {
                    Ok(out) => Ok(out.to_string()),
                    Err(e) => Err(PolarsError::ComputeError(
                        format!("Python function in 'map_alias' produced an error: {e}.").into(),
                    )),
                }
            })
            .into()
    }
    fn exclude(&self, columns: Vec<String>) -> Self {
        self.inner.clone().exclude(columns).into()
    }
    fn exclude_dtype(&self, dtypes: Vec<Wrap<DataType>>) -> Self {
        // Safety:
        // Wrap is transparent.
        let dtypes: Vec<DataType> = unsafe { std::mem::transmute(dtypes) };
        self.inner.clone().exclude_dtype(&dtypes).into()
    }
    fn interpolate(&self, method: Wrap<InterpolationMethod>) -> Self {
        self.inner.clone().interpolate(method.0).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed))]
    fn rolling_sum(
        &self,
        window_size: &str,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
        by: Option<String>,
        closed: Option<Wrap<ClosedWindow>>,
    ) -> Self {
        let options = RollingOptions {
            window_size: Duration::parse(window_size),
            weights,
            min_periods,
            center,
            by,
            closed_window: closed.map(|c| c.0),
            ..Default::default()
        };
        self.inner.clone().rolling_sum(options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed))]
    fn rolling_min(
        &self,
        window_size: &str,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
        by: Option<String>,
        closed: Option<Wrap<ClosedWindow>>,
    ) -> Self {
        let options = RollingOptions {
            window_size: Duration::parse(window_size),
            weights,
            min_periods,
            center,
            by,
            closed_window: closed.map(|c| c.0),
            ..Default::default()
        };
        self.inner.clone().rolling_min(options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed))]
    fn rolling_max(
        &self,
        window_size: &str,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
        by: Option<String>,
        closed: Option<Wrap<ClosedWindow>>,
    ) -> Self {
        let options = RollingOptions {
            window_size: Duration::parse(window_size),
            weights,
            min_periods,
            center,
            by,
            closed_window: closed.map(|c| c.0),
            ..Default::default()
        };
        self.inner.clone().rolling_max(options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed))]
    fn rolling_mean(
        &self,
        window_size: &str,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
        by: Option<String>,
        closed: Option<Wrap<ClosedWindow>>,
    ) -> Self {
        let options = RollingOptions {
            window_size: Duration::parse(window_size),
            weights,
            min_periods,
            center,
            by,
            closed_window: closed.map(|c| c.0),
            ..Default::default()
        };

        self.inner.clone().rolling_mean(options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed, ddof))]
    #[allow(clippy::too_many_arguments)]
    fn rolling_std(
        &self,
        window_size: &str,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
        by: Option<String>,
        closed: Option<Wrap<ClosedWindow>>,
        ddof: u8,
    ) -> Self {
        let options = RollingOptions {
            window_size: Duration::parse(window_size),
            weights,
            min_periods,
            center,
            by,
            closed_window: closed.map(|c| c.0),
            fn_params: Some(Arc::new(RollingVarParams { ddof }) as Arc<dyn Any + Send + Sync>),
        };

        self.inner.clone().rolling_std(options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed, ddof))]
    #[allow(clippy::too_many_arguments)]
    fn rolling_var(
        &self,
        window_size: &str,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
        by: Option<String>,
        closed: Option<Wrap<ClosedWindow>>,
        ddof: u8,
    ) -> Self {
        let options = RollingOptions {
            window_size: Duration::parse(window_size),
            weights,
            min_periods,
            center,
            by,
            closed_window: closed.map(|c| c.0),
            fn_params: Some(Arc::new(RollingVarParams { ddof }) as Arc<dyn Any + Send + Sync>),
        };

        self.inner.clone().rolling_var(options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed))]
    fn rolling_median(
        &self,
        window_size: &str,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
        by: Option<String>,
        closed: Option<Wrap<ClosedWindow>>,
    ) -> Self {
        let options = RollingOptions {
            window_size: Duration::parse(window_size),
            weights,
            min_periods,
            center,
            by,
            closed_window: closed.map(|c| c.0),
            ..Default::default()
        };
        self.inner.clone().rolling_median(options).into()
    }

    #[pyo3(signature = (quantile, interpolation, window_size, weights, min_periods, center, by, closed))]
    #[allow(clippy::too_many_arguments)]
    fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: Wrap<QuantileInterpolOptions>,
        window_size: &str,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
        by: Option<String>,
        closed: Option<Wrap<ClosedWindow>>,
    ) -> Self {
        let options = RollingOptions {
            window_size: Duration::parse(window_size),
            weights,
            min_periods,
            center,
            by,
            closed_window: closed.map(|c| c.0),
            ..Default::default()
        };

        self.inner
            .clone()
            .rolling_quantile(quantile, interpolation.0, options)
            .into()
    }

    fn rolling_skew(&self, window_size: usize, bias: bool) -> Self {
        self.inner.clone().rolling_skew(window_size, bias).into()
    }

    fn lower_bound(&self) -> Self {
        self.inner.clone().lower_bound().into()
    }

    fn upper_bound(&self) -> Self {
        self.inner.clone().upper_bound().into()
    }

    fn cumulative_eval(&self, expr: Self, min_periods: usize, parallel: bool) -> Self {
        self.inner
            .clone()
            .cumulative_eval(expr.inner, min_periods, parallel)
            .into()
    }

    fn rank(&self, method: Wrap<RankMethod>, descending: bool, seed: Option<u64>) -> Self {
        let options = RankOptions {
            method: method.0,
            descending,
        };
        self.inner.clone().rank(options, seed).into()
    }

    fn diff(&self, n: i64, null_behavior: Wrap<NullBehavior>) -> Self {
        self.inner.clone().diff(n, null_behavior.0).into()
    }

    #[cfg(feature = "pct_change")]
    fn pct_change(&self, n: i64) -> Self {
        self.inner.clone().pct_change(n).into()
    }

    fn skew(&self, bias: bool) -> Self {
        self.inner.clone().skew(bias).into()
    }
    fn kurtosis(&self, fisher: bool, bias: bool) -> Self {
        self.inner.clone().kurtosis(fisher, bias).into()
    }

    fn reshape(&self, dims: Vec<i64>) -> Self {
        self.inner.clone().reshape(&dims).into()
    }

    fn cumcount(&self, reverse: bool) -> Self {
        self.inner.clone().cumcount(reverse).into()
    }

    fn to_physical(&self) -> Self {
        self.inner.clone().to_physical().into()
    }

    fn shuffle(&self, seed: Option<u64>) -> Self {
        self.inner.clone().shuffle(seed).into()
    }

    fn sample_n(&self, n: usize, with_replacement: bool, shuffle: bool, seed: Option<u64>) -> Self {
        self.inner
            .clone()
            .sample_n(n, with_replacement, shuffle, seed)
            .into()
    }

    fn sample_frac(
        &self,
        frac: f64,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        self.inner
            .clone()
            .sample_frac(frac, with_replacement, shuffle, seed)
            .into()
    }

    fn ewm_mean(&self, alpha: f64, adjust: bool, min_periods: usize, ignore_nulls: bool) -> Self {
        let options = EWMOptions {
            alpha,
            adjust,
            bias: false,
            min_periods,
            ignore_nulls,
        };
        self.inner.clone().ewm_mean(options).into()
    }
    fn ewm_std(
        &self,
        alpha: f64,
        adjust: bool,
        bias: bool,
        min_periods: usize,
        ignore_nulls: bool,
    ) -> Self {
        let options = EWMOptions {
            alpha,
            adjust,
            bias,
            min_periods,
            ignore_nulls,
        };
        self.inner.clone().ewm_std(options).into()
    }
    fn ewm_var(
        &self,
        alpha: f64,
        adjust: bool,
        bias: bool,
        min_periods: usize,
        ignore_nulls: bool,
    ) -> Self {
        let options = EWMOptions {
            alpha,
            adjust,
            bias,
            min_periods,
            ignore_nulls,
        };
        self.inner.clone().ewm_var(options).into()
    }
    fn extend_constant(&self, py: Python, value: Wrap<AnyValue>, n: usize) -> Self {
        let value = value.into_py(py);
        self.inner
            .clone()
            .apply(
                move |s| {
                    Python::with_gil(|py| {
                        let value = value.extract::<Wrap<AnyValue>>(py).unwrap().0;
                        s.extend_constant(value, n).map(Some)
                    })
                },
                GetOutput::same_type(),
            )
            .with_fmt("extend")
            .into()
    }
    fn any(&self) -> Self {
        self.inner.clone().any().into()
    }

    fn all(&self) -> Self {
        self.inner.clone().all().into()
    }

    fn log(&self, base: f64) -> Self {
        self.inner.clone().log(base).into()
    }

    fn log1p(&self) -> Self {
        self.inner.clone().log1p().into()
    }

    fn exp(&self) -> Self {
        self.inner.clone().exp().into()
    }

    fn entropy(&self, base: f64, normalize: bool) -> Self {
        self.inner.clone().entropy(base, normalize).into()
    }
    fn hash(&self, seed: u64, seed_1: u64, seed_2: u64, seed_3: u64) -> Self {
        self.inner.clone().hash(seed, seed_1, seed_2, seed_3).into()
    }
    fn set_sorted_flag(&self, descending: bool) -> Self {
        let is_sorted = if descending {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        };
        self.inner.clone().set_sorted_flag(is_sorted).into()
    }

    fn cache(&self) -> Self {
        self.inner.clone().cache().into()
    }
}
