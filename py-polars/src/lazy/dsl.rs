mod binary;
mod categorical;
mod datetime;
mod list;
#[cfg(feature = "meta")]
mod meta;
mod string;
mod r#struct;

use polars::lazy::dsl;
use polars::lazy::dsl::Operator;
use polars::prelude::*;
use polars::series::ops::NullBehavior;
use polars_core::prelude::QuantileInterpolOptions;
use polars_core::series::IsSorted;
use pyo3::class::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyString};

use super::apply::*;
use crate::conversion::{parse_fill_null_strategy, Wrap};
use crate::lazy::map_single;
use crate::lazy::utils::py_exprs_to_exprs;
use crate::prelude::ObjectValue;
use crate::series::PySeries;
use crate::utils::reinterpret;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: dsl::Expr,
}

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

    pub fn to_str(&self) -> String {
        format!("{:?}", self.inner)
    }
    pub fn eq(&self, other: Self) -> Self {
        self.clone().inner.eq(other.inner).into()
    }
    pub fn neq(&self, other: Self) -> Self {
        self.clone().inner.neq(other.inner).into()
    }
    pub fn gt(&self, other: Self) -> Self {
        self.clone().inner.gt(other.inner).into()
    }
    pub fn gt_eq(&self, other: Self) -> Self {
        self.clone().inner.gt_eq(other.inner).into()
    }
    pub fn lt_eq(&self, other: Self) -> Self {
        self.clone().inner.lt_eq(other.inner).into()
    }
    pub fn lt(&self, other: Self) -> Self {
        self.clone().inner.lt(other.inner).into()
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        // Used in pickle/pickling
        #[cfg(feature = "json")]
        {
            let s = serde_json::to_string(&self.inner).unwrap();
            Ok(PyBytes::new(py, s.as_bytes()).to_object(py))
        }
        #[cfg(not(feature = "json"))]
        {
            panic!("activate 'json' feature")
        }
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Used in pickle/pickling
        #[cfg(feature = "json")]
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                // Safety
                // we skipped the serializing/deserializing of the static in lifetime in `DataType`
                // so we actually don't have a lifetime at all when serializing.

                // PyBytes still has a lifetime. Bit its ok, because we drop it immediately
                // in this scope
                let s = unsafe { std::mem::transmute::<&'_ PyBytes, &'static PyBytes>(s) };
                self.inner = serde_json::from_slice(s.as_bytes()).unwrap();

                Ok(())
            }
            Err(e) => Err(e),
        }

        #[cfg(not(feature = "json"))]
        {
            panic!("activate 'json' feature")
        }
    }

    pub fn alias(&self, name: &str) -> Self {
        self.clone().inner.alias(name).into()
    }
    pub fn is_not(&self) -> Self {
        self.clone().inner.not().into()
    }
    pub fn is_null(&self) -> Self {
        self.clone().inner.is_null().into()
    }
    pub fn is_not_null(&self) -> Self {
        self.clone().inner.is_not_null().into()
    }

    pub fn is_infinite(&self) -> Self {
        self.clone().inner.is_infinite().into()
    }

    pub fn is_finite(&self) -> Self {
        self.clone().inner.is_finite().into()
    }

    pub fn is_nan(&self) -> Self {
        self.clone().inner.is_nan().into()
    }

    pub fn is_not_nan(&self) -> Self {
        self.clone().inner.is_not_nan().into()
    }

    pub fn min(&self) -> Self {
        self.clone().inner.min().into()
    }
    pub fn max(&self) -> Self {
        self.clone().inner.max().into()
    }
    #[cfg(feature = "propagate_nans")]
    pub fn nan_max(&self) -> Self {
        self.clone().inner.nan_max().into()
    }
    #[cfg(feature = "propagate_nans")]
    pub fn nan_min(&self) -> Self {
        self.clone().inner.nan_min().into()
    }
    pub fn mean(&self) -> Self {
        self.clone().inner.mean().into()
    }
    pub fn median(&self) -> Self {
        self.clone().inner.median().into()
    }
    pub fn sum(&self) -> Self {
        self.clone().inner.sum().into()
    }
    pub fn n_unique(&self) -> Self {
        self.clone().inner.n_unique().into()
    }
    pub fn arg_unique(&self) -> Self {
        self.clone().inner.arg_unique().into()
    }
    pub fn unique(&self) -> Self {
        self.clone().inner.unique().into()
    }
    pub fn unique_stable(&self) -> Self {
        self.clone().inner.unique_stable().into()
    }
    pub fn first(&self) -> Self {
        self.clone().inner.first().into()
    }
    pub fn last(&self) -> Self {
        self.clone().inner.last().into()
    }
    pub fn implode(&self) -> Self {
        self.clone().inner.implode().into()
    }
    pub fn quantile(&self, quantile: Self, interpolation: Wrap<QuantileInterpolOptions>) -> Self {
        self.clone()
            .inner
            .quantile(quantile.inner, interpolation.0)
            .into()
    }
    pub fn agg_groups(&self) -> Self {
        self.clone().inner.agg_groups().into()
    }
    pub fn count(&self) -> Self {
        self.clone().inner.count().into()
    }
    pub fn value_counts(&self, multithreaded: bool, sorted: bool) -> Self {
        self.inner
            .clone()
            .value_counts(multithreaded, sorted)
            .into()
    }
    pub fn unique_counts(&self) -> Self {
        self.inner.clone().unique_counts().into()
    }
    pub fn null_count(&self) -> Self {
        self.inner.clone().null_count().into()
    }
    pub fn cast(&self, data_type: Wrap<DataType>, strict: bool) -> Self {
        let dt = data_type.0;
        let expr = if strict {
            self.inner.clone().strict_cast(dt)
        } else {
            self.inner.clone().cast(dt)
        };
        expr.into()
    }
    pub fn sort_with(&self, descending: bool, nulls_last: bool) -> Self {
        self.clone()
            .inner
            .sort_with(SortOptions {
                descending,
                nulls_last,
                multithreaded: true,
            })
            .into()
    }

    pub fn arg_sort(&self, descending: bool, nulls_last: bool) -> Self {
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
    pub fn top_k(&self, k: usize) -> Self {
        self.inner.clone().top_k(k).into()
    }

    #[cfg(feature = "top_k")]
    pub fn bottom_k(&self, k: usize) -> Self {
        self.inner.clone().bottom_k(k).into()
    }

    pub fn arg_max(&self) -> Self {
        self.clone().inner.arg_max().into()
    }
    pub fn arg_min(&self) -> Self {
        self.clone().inner.arg_min().into()
    }

    #[cfg(feature = "search_sorted")]
    pub fn search_sorted(&self, element: Self, side: Wrap<SearchSortedSide>) -> Self {
        self.inner
            .clone()
            .search_sorted(element.inner, side.0)
            .into()
    }
    pub fn take(&self, idx: Self) -> Self {
        self.clone().inner.take(idx.inner).into()
    }

    pub fn sort_by(&self, by: Vec<Self>, descending: Vec<bool>) -> Self {
        let by = by.into_iter().map(|e| e.inner).collect::<Vec<_>>();
        self.clone().inner.sort_by(by, descending).into()
    }

    pub fn backward_fill(&self, limit: FillNullLimit) -> Self {
        self.clone().inner.backward_fill(limit).into()
    }

    pub fn forward_fill(&self, limit: FillNullLimit) -> Self {
        self.clone().inner.forward_fill(limit).into()
    }

    pub fn shift(&self, periods: i64) -> Self {
        self.clone().inner.shift(periods).into()
    }
    pub fn shift_and_fill(&self, periods: i64, fill_value: Self) -> Self {
        self.clone()
            .inner
            .shift_and_fill(periods, fill_value.inner)
            .into()
    }

    pub fn fill_null(&self, expr: Self) -> Self {
        self.clone().inner.fill_null(expr.inner).into()
    }

    pub fn fill_null_with_strategy(&self, strategy: &str, limit: FillNullLimit) -> PyResult<Self> {
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

    pub fn fill_nan(&self, expr: Self) -> Self {
        self.inner.clone().fill_nan(expr.inner).into()
    }

    pub fn drop_nulls(&self) -> Self {
        self.inner.clone().drop_nulls().into()
    }

    pub fn drop_nans(&self) -> Self {
        self.inner.clone().drop_nans().into()
    }

    pub fn filter(&self, predicate: Self) -> Self {
        self.clone().inner.filter(predicate.inner).into()
    }
    pub fn reverse(&self) -> Self {
        self.clone().inner.reverse().into()
    }
    pub fn std(&self, ddof: u8) -> Self {
        self.clone().inner.std(ddof).into()
    }
    pub fn var(&self, ddof: u8) -> Self {
        self.clone().inner.var(ddof).into()
    }
    pub fn is_unique(&self) -> Self {
        self.clone().inner.is_unique().into()
    }

    pub fn approx_unique(&self) -> Self {
        self.clone().inner.approx_unique().into()
    }

    pub fn is_first(&self) -> Self {
        self.clone().inner.is_first().into()
    }

    pub fn explode(&self) -> Self {
        self.clone().inner.explode().into()
    }

    pub fn take_every(&self, n: usize) -> Self {
        self.clone()
            .inner
            .map(
                move |s: Series| Ok(Some(s.take_every(n))),
                GetOutput::same_type(),
            )
            .with_fmt("take_every")
            .into()
    }
    pub fn tail(&self, n: usize) -> Self {
        self.clone().inner.tail(Some(n)).into()
    }

    pub fn head(&self, n: usize) -> Self {
        self.clone().inner.head(Some(n)).into()
    }

    pub fn slice(&self, offset: Self, length: Self) -> Self {
        self.inner.clone().slice(offset.inner, length.inner).into()
    }

    pub fn append(&self, other: Self, upcast: bool) -> Self {
        self.inner.clone().append(other.inner, upcast).into()
    }

    pub fn rechunk(&self) -> Self {
        self.inner
            .clone()
            .map(|s| Ok(Some(s.rechunk())), GetOutput::same_type())
            .into()
    }

    pub fn round(&self, decimals: u32) -> Self {
        self.clone().inner.round(decimals).into()
    }

    pub fn floor(&self) -> Self {
        self.clone().inner.floor().into()
    }

    pub fn ceil(&self) -> Self {
        self.clone().inner.ceil().into()
    }

    pub fn clip(&self, py: Python, min: PyObject, max: PyObject) -> Self {
        let min = min.extract::<Wrap<AnyValue>>(py).unwrap().0;
        let max = max.extract::<Wrap<AnyValue>>(py).unwrap().0;
        self.clone().inner.clip(min, max).into()
    }

    pub fn clip_min(&self, py: Python, min: PyObject) -> Self {
        let min = min.extract::<Wrap<AnyValue>>(py).unwrap().0;
        self.clone().inner.clip_min(min).into()
    }

    pub fn clip_max(&self, py: Python, max: PyObject) -> Self {
        let max = max.extract::<Wrap<AnyValue>>(py).unwrap().0;
        self.clone().inner.clip_max(max).into()
    }

    pub fn abs(&self) -> Self {
        self.clone().inner.abs().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn sin(&self) -> Self {
        self.clone().inner.sin().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn cos(&self) -> Self {
        self.clone().inner.cos().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn tan(&self) -> Self {
        self.clone().inner.tan().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn arcsin(&self) -> Self {
        self.clone().inner.arcsin().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn arccos(&self) -> Self {
        self.clone().inner.arccos().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn arctan(&self) -> Self {
        self.clone().inner.arctan().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn sinh(&self) -> Self {
        self.clone().inner.sinh().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn cosh(&self) -> Self {
        self.clone().inner.cosh().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn tanh(&self) -> Self {
        self.clone().inner.tanh().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn arcsinh(&self) -> Self {
        self.clone().inner.arcsinh().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn arccosh(&self) -> Self {
        self.clone().inner.arccosh().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn arctanh(&self) -> Self {
        self.clone().inner.arctanh().into()
    }

    #[cfg(feature = "sign")]
    pub fn sign(&self) -> Self {
        self.clone().inner.sign().into()
    }

    pub fn is_duplicated(&self) -> Self {
        self.clone().inner.is_duplicated().into()
    }

    pub fn over(&self, partition_by: Vec<Self>) -> Self {
        let partition_by = partition_by
            .into_iter()
            .map(|e| e.inner)
            .collect::<Vec<Expr>>();
        self.clone().inner.over(partition_by).into()
    }

    pub fn _and(&self, expr: Self) -> Self {
        self.clone().inner.and(expr.inner).into()
    }

    pub fn _xor(&self, expr: Self) -> Self {
        self.clone().inner.xor(expr.inner).into()
    }

    pub fn _or(&self, expr: Self) -> Self {
        self.clone().inner.or(expr.inner).into()
    }
    #[cfg(feature = "is_in")]
    pub fn is_in(&self, expr: Self) -> Self {
        self.clone().inner.is_in(expr.inner).into()
    }

    #[cfg(feature = "repeat_by")]
    pub fn repeat_by(&self, by: Self) -> Self {
        self.clone().inner.repeat_by(by.inner).into()
    }

    pub fn pow(&self, exponent: Self) -> Self {
        self.clone().inner.pow(exponent.inner).into()
    }

    pub fn cumsum(&self, reverse: bool) -> Self {
        self.clone().inner.cumsum(reverse).into()
    }
    pub fn cummax(&self, reverse: bool) -> Self {
        self.clone().inner.cummax(reverse).into()
    }
    pub fn cummin(&self, reverse: bool) -> Self {
        self.clone().inner.cummin(reverse).into()
    }

    pub fn cumprod(&self, reverse: bool) -> Self {
        self.clone().inner.cumprod(reverse).into()
    }

    pub fn product(&self) -> Self {
        self.clone().inner.product().into()
    }

    pub fn shrink_dtype(&self) -> Self {
        self.inner.clone().shrink_dtype().into()
    }

    #[pyo3(signature = (lambda, window_size, weights, min_periods, center))]
    pub fn rolling_apply(
        &self,
        py: Python,
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
        };
        // get the pypolars module
        // do the import outside of the function.
        let pypolars = PyModule::import(py, "polars").unwrap().to_object(py);

        let function = move |s: &Series| {
            Python::with_gil(|py| {
                let out = call_lambda_with_series(py, s.clone(), &lambda, &pypolars)
                    .expect("python function failed");
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
    pub fn map(
        &self,
        lambda: PyObject,
        output_type: Option<Wrap<DataType>>,
        agg_list: bool,
    ) -> Self {
        map_single(self, lambda, output_type, agg_list)
    }

    pub fn dot(&self, other: Self) -> Self {
        self.inner.clone().dot(other.inner).into()
    }

    pub fn reinterpret(&self, signed: bool) -> Self {
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
    pub fn mode(&self) -> Self {
        self.inner.clone().mode().into()
    }
    pub fn keep_name(&self) -> Self {
        self.inner.clone().keep_name().into()
    }
    pub fn prefix(&self, prefix: &str) -> Self {
        self.inner.clone().prefix(prefix).into()
    }
    pub fn suffix(&self, suffix: &str) -> Self {
        self.inner.clone().suffix(suffix).into()
    }
    pub fn map_alias(&self, lambda: PyObject) -> Self {
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
    pub fn exclude(&self, columns: Vec<String>) -> Self {
        self.inner.clone().exclude(columns).into()
    }
    pub fn exclude_dtype(&self, dtypes: Vec<Wrap<DataType>>) -> Self {
        // Safety:
        // Wrap is transparent.
        let dtypes: Vec<DataType> = unsafe { std::mem::transmute(dtypes) };
        self.inner.clone().exclude_dtype(&dtypes).into()
    }
    pub fn interpolate(&self, method: Wrap<InterpolationMethod>) -> Self {
        self.inner.clone().interpolate(method.0).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed))]
    pub fn rolling_sum(
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
        };
        self.inner.clone().rolling_sum(options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed))]
    pub fn rolling_min(
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
        };
        self.inner.clone().rolling_min(options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed))]
    pub fn rolling_max(
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
        };
        self.inner.clone().rolling_max(options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed))]
    pub fn rolling_mean(
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
        };

        self.inner.clone().rolling_mean(options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed))]
    pub fn rolling_std(
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
        };

        self.inner.clone().rolling_std(options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed))]
    pub fn rolling_var(
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
        };

        self.inner.clone().rolling_var(options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, by, closed))]
    pub fn rolling_median(
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
        };
        self.inner.clone().rolling_median(options).into()
    }

    #[pyo3(signature = (quantile, interpolation, window_size, weights, min_periods, center, by, closed))]
    #[allow(clippy::too_many_arguments)]
    pub fn rolling_quantile(
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
        };

        self.inner
            .clone()
            .rolling_quantile(quantile, interpolation.0, options)
            .into()
    }

    pub fn rolling_skew(&self, window_size: usize, bias: bool) -> Self {
        self.inner.clone().rolling_skew(window_size, bias).into()
    }

    pub fn lower_bound(&self) -> Self {
        self.inner.clone().lower_bound().into()
    }

    pub fn upper_bound(&self) -> Self {
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
                |s| Ok(Some(s.to_physical_repr().into_owned())),
                GetOutput::map_dtype(|dt| dt.to_physical()),
            )
            .with_fmt("to_physical")
            .into()
    }

    pub fn shuffle(&self, seed: Option<u64>) -> Self {
        self.inner.clone().shuffle(seed).into()
    }

    pub fn sample_n(
        &self,
        n: usize,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        self.inner
            .clone()
            .sample_n(n, with_replacement, shuffle, seed)
            .into()
    }

    pub fn sample_frac(
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

    pub fn ewm_mean(
        &self,
        alpha: f64,
        adjust: bool,
        min_periods: usize,
        ignore_nulls: bool,
    ) -> Self {
        let options = EWMOptions {
            alpha,
            adjust,
            bias: false,
            min_periods,
            ignore_nulls,
        };
        self.inner.clone().ewm_mean(options).into()
    }
    pub fn ewm_std(
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
    pub fn ewm_var(
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
    pub fn extend_constant(&self, py: Python, value: Wrap<AnyValue>, n: usize) -> Self {
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
    pub fn any(&self) -> Self {
        self.inner.clone().any().into()
    }

    pub fn all(&self) -> Self {
        self.inner.clone().all().into()
    }

    pub fn log(&self, base: f64) -> Self {
        self.inner.clone().log(base).into()
    }

    pub fn log1p(&self) -> Self {
        self.inner.clone().log1p().into()
    }

    pub fn exp(&self) -> Self {
        self.inner.clone().exp().into()
    }

    pub fn entropy(&self, base: f64, normalize: bool) -> Self {
        self.inner.clone().entropy(base, normalize).into()
    }
    pub fn hash(&self, seed: u64, seed_1: u64, seed_2: u64, seed_3: u64) -> Self {
        self.inner.clone().hash(seed, seed_1, seed_2, seed_3).into()
    }
    pub fn set_sorted_flag(&self, descending: bool) -> Self {
        let is_sorted = if descending {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        };
        self.inner.clone().set_sorted_flag(is_sorted).into()
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

pub fn reduce(lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = py_exprs_to_exprs(exprs);

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    polars::lazy::dsl::reduce_exprs(func, exprs).into()
}

pub fn cumfold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>, include_init: bool) -> PyExpr {
    let exprs = py_exprs_to_exprs(exprs);

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    cumfold_exprs(acc.inner, func, exprs, include_init).into()
}

pub fn cumreduce(lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = py_exprs_to_exprs(exprs);

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    cumreduce_exprs(func, exprs).into()
}

pub fn lit(value: &PyAny, allow_object: bool) -> PyResult<PyExpr> {
    if let Ok(true) = value.is_instance_of::<PyBool>() {
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

pub fn range(low: i64, high: i64, dtype: Wrap<DataType>) -> PyExpr {
    match dtype.0 {
        DataType::Int32 => dsl::range(low as i32, high as i32).into(),
        DataType::UInt32 => dsl::range(low as u32, high as u32).into(),
        _ => dsl::range(low, high).into(),
    }
}
