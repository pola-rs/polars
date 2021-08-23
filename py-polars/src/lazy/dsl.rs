use crate::lazy::utils::py_exprs_to_exprs;
use crate::series::PySeries;
use crate::utils::{reinterpret, str_to_polarstype};
use polars::lazy::dsl;
use polars::lazy::dsl::Operator;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};
use pyo3::{class::basic::CompareOp, PyNumberProtocol, PyObjectProtocol};

fn call_lambda_with_series(
    py: Python,
    s: Series,
    lambda: &PyObject,
    polars_module: &PyObject,
) -> PyObject {
    let pypolars = polars_module.cast_as::<PyModule>(py).unwrap();

    // create a PySeries struct/object for Python
    let pyseries = PySeries::new(s);
    // Wrap this PySeries object in the python side Series wrapper
    let python_series_wrapper = pypolars
        .getattr("wrap_s")
        .unwrap()
        .call1((pyseries,))
        .unwrap();
    // call the lambda and get a python side Series wrapper
    match lambda.call1(py, (python_series_wrapper,)) {
        Ok(pyobj) => pyobj,
        Err(e) => panic!("python apply failed: {}", e.pvalue(py).to_string()),
    }
}

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
    pub fn quantile(&self, quantile: f64) -> PyExpr {
        self.clone().inner.quantile(quantile).into()
    }
    pub fn agg_groups(&self) -> PyExpr {
        self.clone().inner.agg_groups().into()
    }
    pub fn count(&self) -> PyExpr {
        self.clone().inner.count().into()
    }
    pub fn cast(&self, data_type: &PyAny) -> PyExpr {
        let str_repr = data_type.str().unwrap().to_str().unwrap();
        let dt = str_to_polarstype(str_repr);
        let expr = self.inner.clone().cast(dt);
        expr.into()
    }
    pub fn sort(&self, reverse: bool) -> PyExpr {
        self.clone().inner.sort(reverse).into()
    }

    pub fn arg_sort(&self, reverse: bool) -> PyExpr {
        self.clone().inner.arg_sort(reverse).into()
    }

    pub fn take(&self, idx: PyExpr) -> PyExpr {
        self.clone().inner.take(idx.inner).into()
    }

    pub fn sort_by(&self, by: PyExpr, reverse: bool) -> PyExpr {
        self.clone().inner.sort_by(by.inner, reverse).into()
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

    pub fn fill_none(&self, expr: PyExpr) -> PyExpr {
        self.clone().inner.fill_none(expr.inner).into()
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
            .map(move |s: Series| Ok(s.take_every(n)), None)
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

    pub fn is_duplicated(&self) -> PyExpr {
        self.clone().inner.is_duplicated().into()
    }

    pub fn over(&self, partition_by: Vec<PyExpr>) -> PyExpr {
        let partition_by = partition_by.into_iter().map(|e| e.inner).collect();
        self.clone().inner.over(partition_by).into()
    }

    pub fn _and(&self, expr: PyExpr) -> PyExpr {
        self.clone().inner.and(expr.inner).into()
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

    pub fn cum_sum(&self, reverse: bool) -> PyExpr {
        self.clone().inner.cum_sum(reverse).into()
    }
    pub fn cum_max(&self, reverse: bool) -> PyExpr {
        self.clone().inner.cum_max(reverse).into()
    }
    pub fn cum_min(&self, reverse: bool) -> PyExpr {
        self.clone().inner.cum_min(reverse).into()
    }

    pub fn str_parse_date32(&self, fmt: Option<String>) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            ca.as_date32(fmt.as_deref()).map(|ca| ca.into_series())
        };
        self.clone()
            .inner
            .map(function, Some(DataType::Date32))
            .into()
    }

    pub fn str_parse_date64(&self, fmt: Option<String>) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            ca.as_date64(fmt.as_deref()).map(|ca| ca.into_series())
        };
        self.clone()
            .inner
            .map(function, Some(DataType::Date64))
            .into()
    }

    pub fn str_to_uppercase(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;
            Ok(ca.to_uppercase().into_series())
        };
        self.clone()
            .inner
            .map(function, Some(DataType::UInt32))
            .into()
    }

    pub fn str_slice(&self, start: i64, length: Option<u64>) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            Ok(ca.str_slice(start, length)?.into_series())
        };
        self.clone()
            .inner
            .map(function, Some(DataType::Utf8))
            .into()
    }

    pub fn str_to_lowercase(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;
            Ok(ca.to_lowercase().into_series())
        };
        self.clone()
            .inner
            .map(function, Some(DataType::UInt32))
            .into()
    }

    pub fn str_lengths(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;
            Ok(ca.str_lengths().into_series())
        };
        self.clone()
            .inner
            .map(function, Some(DataType::UInt32))
            .into()
    }

    pub fn str_replace(&self, pat: String, val: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.replace(&pat, &val) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::Other(format!("{:?}", e).into())),
            }
        };
        self.clone().inner.map(function, None).into()
    }

    pub fn str_replace_all(&self, pat: String, val: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.replace_all(&pat, &val) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::Other(format!("{:?}", e).into())),
            }
        };
        self.clone().inner.map(function, None).into()
    }

    pub fn str_contains(&self, pat: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.contains(&pat) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::Other(format!("{:?}", e).into())),
            }
        };
        self.clone()
            .inner
            .map(function, Some(DataType::Boolean))
            .into()
    }

    pub fn str_json_path_match(&self, pat: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.json_path_match(&pat) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::Other(format!("{:?}", e).into())),
            }
        };
        self.clone()
            .inner
            .map(function, Some(DataType::Boolean))
            .into()
    }

    pub fn strftime(&self, fmt: String) -> PyExpr {
        let function = move |s: Series| s.strftime(&fmt);
        self.clone()
            .inner
            .map(function, Some(DataType::Utf8))
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
                Some(DataType::Int64),
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
            .rolling_apply(window_size, Arc::new(function))
            .into()
    }

    pub fn map(&self, py: Python, lambda: PyObject, output_type: &PyAny, agg_list: bool) -> PyExpr {
        let output_type = match output_type.is_none() {
            true => None,
            false => {
                let str_repr = output_type.str().unwrap().to_str().unwrap();
                Some(str_to_polarstype(str_repr))
            }
        };
        // get the pypolars module
        // do the import outside of the function to prevent import side effects in a hot loop.
        let pypolars = PyModule::import(py, "polars").unwrap().to_object(py);

        let function = move |s: Series| {
            let gil = Python::acquire_gil();
            let py = gil.python();

            // this is a python Series
            let out = call_lambda_with_series(py, s, &lambda, &pypolars);
            // unpack the wrapper in a PySeries
            let py_pyseries = out.getattr(py, "_s").expect(
                "Could net get series attribute '_s'. Make sure that you return a Series object.",
            );
            // Downcast to Rust
            let pyseries = py_pyseries.extract::<PySeries>(py).unwrap();
            // Finally get the actual Series
            Ok(pyseries.series)
        };

        if agg_list {
            self.clone().inner.map_list(function, output_type).into()
        } else {
            self.clone().inner.map(function, output_type).into()
        }
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
            .map(function, Some(DataType::UInt64))
            .into()
    }

    pub fn reinterpret(&self, signed: bool) -> PyExpr {
        let function = move |s: Series| reinterpret(&s, signed);
        let dt = if signed {
            DataType::Int64
        } else {
            DataType::UInt64
        };
        self.clone().inner.map(function, Some(dt)).into()
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
    pub fn interpolate(&self) -> PyExpr {
        self.inner.clone().interpolate().into()
    }

    pub fn rolling_sum(
        &self,
        window_size: u32,
        weight: Option<Vec<f64>>,
        ignore_null: bool,
        min_periods: u32,
    ) -> PyExpr {
        self.inner
            .clone()
            .rolling_sum(window_size, weight.as_deref(), ignore_null, min_periods)
            .into()
    }
    pub fn rolling_min(
        &self,
        window_size: u32,
        weight: Option<Vec<f64>>,
        ignore_null: bool,
        min_periods: u32,
    ) -> Self {
        self.inner
            .clone()
            .rolling_min(window_size, weight.as_deref(), ignore_null, min_periods)
            .into()
    }
    pub fn rolling_max(
        &self,
        window_size: u32,
        weight: Option<Vec<f64>>,
        ignore_null: bool,
        min_periods: u32,
    ) -> Self {
        self.inner
            .clone()
            .rolling_max(window_size, weight.as_deref(), ignore_null, min_periods)
            .into()
    }
    pub fn rolling_mean(
        &self,
        window_size: u32,
        weight: Option<Vec<f64>>,
        ignore_null: bool,
        min_periods: u32,
    ) -> Self {
        self.inner
            .clone()
            .rolling_mean(window_size, weight.as_deref(), ignore_null, min_periods)
            .into()
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

/// A python lambda taking two Series
fn binary_lambda(lambda: &PyObject, a: Series, b: Series) -> Result<Series> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    // get the pypolars module
    let pypolars = PyModule::import(py, "polars").unwrap();
    // create a PySeries struct/object for Python
    let pyseries_a = PySeries::new(a);
    let pyseries_b = PySeries::new(b);

    // Wrap this PySeries object in the python side Series wrapper
    let python_series_wrapper_a = pypolars
        .getattr("wrap_s")
        .unwrap()
        .call1((pyseries_a,))
        .unwrap();
    let python_series_wrapper_b = pypolars
        .getattr("wrap_s")
        .unwrap()
        .call1((pyseries_b,))
        .unwrap();

    // call the lambda and get a python side Series wrapper
    let result_series_wrapper =
        match lambda.call1(py, (python_series_wrapper_a, python_series_wrapper_b)) {
            Ok(pyobj) => pyobj,
            Err(e) => panic!("UDF failed: {}", e.pvalue(py).to_string()),
        };
    // unpack the wrapper in a PySeries
    let py_pyseries = result_series_wrapper
        .getattr(py, "_s")
        .expect("Could net get series attribute '_s'. Make sure that you return a Series object.");
    // Downcast to Rust
    let pyseries = py_pyseries.extract::<PySeries>(py).unwrap();
    // Finally get the actual Series
    Ok(pyseries.series)
}

pub fn binary_function(
    input_a: PyExpr,
    input_b: PyExpr,
    lambda: PyObject,
    output_type: &PyAny,
) -> PyExpr {
    let input_a = input_a.inner;
    let input_b = input_b.inner;

    let output_field = match output_type.is_none() {
        true => Field::new("binary_function", DataType::Null),
        false => {
            let str_repr = output_type.str().unwrap().to_str().unwrap();
            let data_type = str_to_polarstype(str_repr);
            Field::new("binary_function", data_type)
        }
    };

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);

    polars::lazy::dsl::map_binary(input_a, input_b, func, Some(output_field)).into()
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
        }  else {
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
