use std::any::Any;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyFloat;

use crate::conversion::Wrap;
use crate::map::lazy::call_lambda_with_series;
use crate::{PyExpr, PySeries};

#[pymethods]
impl PyExpr {
    #[pyo3(signature = (window_size, weights, min_periods, center))]
    fn rolling_sum(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: Option<usize>,
        center: bool,
    ) -> Self {
        let min_periods = min_periods.unwrap_or(window_size);
        let options = RollingOptionsFixedWindow {
            window_size,
            weights,
            min_periods,
            center,
            ..Default::default()
        };
        self.inner.clone().rolling_sum(options).into()
    }

    #[pyo3(signature = (by, window_size, min_periods, closed))]
    fn rolling_sum_by(
        &self,
        by: PyExpr,
        window_size: &str,
        min_periods: usize,
        closed: Wrap<ClosedWindow>,
    ) -> Self {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::parse(window_size),
            min_periods,
            closed_window: closed.0,
            fn_params: None,
        };
        self.inner.clone().rolling_sum_by(by.inner, options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center))]
    fn rolling_min(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: Option<usize>,
        center: bool,
    ) -> Self {
        let min_periods = min_periods.unwrap_or(window_size);
        let options = RollingOptionsFixedWindow {
            window_size,
            weights,
            min_periods,
            center,
            ..Default::default()
        };
        self.inner.clone().rolling_min(options).into()
    }

    #[pyo3(signature = (by, window_size, min_periods, closed))]
    fn rolling_min_by(
        &self,
        by: PyExpr,
        window_size: &str,
        min_periods: usize,
        closed: Wrap<ClosedWindow>,
    ) -> Self {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::parse(window_size),
            min_periods,
            closed_window: closed.0,
            fn_params: None,
        };
        self.inner.clone().rolling_min_by(by.inner, options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center))]
    fn rolling_max(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: Option<usize>,
        center: bool,
    ) -> Self {
        let min_periods = min_periods.unwrap_or(window_size);
        let options = RollingOptionsFixedWindow {
            window_size,
            weights,
            min_periods,
            center,
            ..Default::default()
        };
        self.inner.clone().rolling_max(options).into()
    }
    #[pyo3(signature = (by, window_size, min_periods, closed))]
    fn rolling_max_by(
        &self,
        by: PyExpr,
        window_size: &str,
        min_periods: usize,
        closed: Wrap<ClosedWindow>,
    ) -> Self {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::parse(window_size),
            min_periods,
            closed_window: closed.0,
            fn_params: None,
        };
        self.inner.clone().rolling_max_by(by.inner, options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center))]
    fn rolling_mean(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: Option<usize>,
        center: bool,
    ) -> Self {
        let min_periods = min_periods.unwrap_or(window_size);
        let options = RollingOptionsFixedWindow {
            window_size,
            weights,
            min_periods,
            center,
            ..Default::default()
        };

        self.inner.clone().rolling_mean(options).into()
    }

    #[pyo3(signature = (by, window_size, min_periods, closed))]
    fn rolling_mean_by(
        &self,
        by: PyExpr,
        window_size: &str,
        min_periods: usize,
        closed: Wrap<ClosedWindow>,
    ) -> Self {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::parse(window_size),
            min_periods,
            closed_window: closed.0,
            fn_params: None,
        };

        self.inner.clone().rolling_mean_by(by.inner, options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, ddof))]
    fn rolling_std(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: Option<usize>,
        center: bool,
        ddof: u8,
    ) -> Self {
        let min_periods = min_periods.unwrap_or(window_size);
        let options = RollingOptionsFixedWindow {
            window_size,
            weights,
            min_periods,
            center,
            fn_params: Some(Arc::new(RollingVarParams { ddof }) as Arc<dyn Any + Send + Sync>),
        };

        self.inner.clone().rolling_std(options).into()
    }

    #[pyo3(signature = (by, window_size, min_periods, closed, ddof))]
    fn rolling_std_by(
        &self,
        by: PyExpr,
        window_size: &str,
        min_periods: usize,
        closed: Wrap<ClosedWindow>,
        ddof: u8,
    ) -> Self {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::parse(window_size),
            min_periods,
            closed_window: closed.0,
            fn_params: Some(Arc::new(RollingVarParams { ddof }) as Arc<dyn Any + Send + Sync>),
        };

        self.inner.clone().rolling_std_by(by.inner, options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center, ddof))]
    fn rolling_var(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: Option<usize>,
        center: bool,
        ddof: u8,
    ) -> Self {
        let min_periods = min_periods.unwrap_or(window_size);
        let options = RollingOptionsFixedWindow {
            window_size,
            weights,
            min_periods,
            center,
            fn_params: Some(Arc::new(RollingVarParams { ddof }) as Arc<dyn Any + Send + Sync>),
        };

        self.inner.clone().rolling_var(options).into()
    }

    #[pyo3(signature = (by, window_size, min_periods, closed, ddof))]
    fn rolling_var_by(
        &self,
        by: PyExpr,
        window_size: &str,
        min_periods: usize,
        closed: Wrap<ClosedWindow>,
        ddof: u8,
    ) -> Self {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::parse(window_size),
            min_periods,
            closed_window: closed.0,
            fn_params: Some(Arc::new(RollingVarParams { ddof }) as Arc<dyn Any + Send + Sync>),
        };

        self.inner.clone().rolling_var_by(by.inner, options).into()
    }

    #[pyo3(signature = (window_size, weights, min_periods, center))]
    fn rolling_median(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: Option<usize>,
        center: bool,
    ) -> Self {
        let min_periods = min_periods.unwrap_or(window_size);
        let options = RollingOptionsFixedWindow {
            window_size,
            min_periods,
            weights,
            center,
            fn_params: None,
        };
        self.inner.clone().rolling_median(options).into()
    }

    #[pyo3(signature = (by, window_size, min_periods, closed))]
    fn rolling_median_by(
        &self,
        by: PyExpr,
        window_size: &str,
        min_periods: usize,
        closed: Wrap<ClosedWindow>,
    ) -> Self {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::parse(window_size),
            min_periods,
            closed_window: closed.0,
            fn_params: None,
        };
        self.inner
            .clone()
            .rolling_median_by(by.inner, options)
            .into()
    }

    #[pyo3(signature = (quantile, interpolation, window_size, weights, min_periods, center))]
    fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: Wrap<QuantileInterpolOptions>,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: Option<usize>,
        center: bool,
    ) -> Self {
        let min_periods = min_periods.unwrap_or(window_size);
        let options = RollingOptionsFixedWindow {
            window_size,
            weights,
            min_periods,
            center,
            fn_params: None,
        };

        self.inner
            .clone()
            .rolling_quantile(interpolation.0, quantile, options)
            .into()
    }

    #[pyo3(signature = (by, quantile, interpolation, window_size, min_periods, closed))]
    fn rolling_quantile_by(
        &self,
        by: PyExpr,
        quantile: f64,
        interpolation: Wrap<QuantileInterpolOptions>,
        window_size: &str,
        min_periods: usize,
        closed: Wrap<ClosedWindow>,
    ) -> Self {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::parse(window_size),
            min_periods,
            closed_window: closed.0,
            fn_params: None,
        };

        self.inner
            .clone()
            .rolling_quantile_by(by.inner, interpolation.0, quantile, options)
            .into()
    }

    fn rolling_skew(&self, window_size: usize, bias: bool) -> Self {
        self.inner.clone().rolling_skew(window_size, bias).into()
    }

    #[pyo3(signature = (lambda, window_size, weights, min_periods, center))]
    fn rolling_map(
        &self,
        lambda: PyObject,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: Option<usize>,
        center: bool,
    ) -> Self {
        let min_periods = min_periods.unwrap_or(window_size);
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
                    },
                    Err(_) => {
                        let obj = out;
                        let is_float = obj.bind(py).is_instance_of::<PyFloat>();

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
                            },
                            UInt16 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(UInt16Chunked::from_slice("", &[v as u16]).into_series())
                                } else {
                                    obj.extract::<u16>(py)
                                        .map(|v| UInt16Chunked::from_slice("", &[v]).into_series())
                                }
                            },
                            UInt32 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(UInt32Chunked::from_slice("", &[v as u32]).into_series())
                                } else {
                                    obj.extract::<u32>(py)
                                        .map(|v| UInt32Chunked::from_slice("", &[v]).into_series())
                                }
                            },
                            UInt64 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(UInt64Chunked::from_slice("", &[v as u64]).into_series())
                                } else {
                                    obj.extract::<u64>(py)
                                        .map(|v| UInt64Chunked::from_slice("", &[v]).into_series())
                                }
                            },
                            Int8 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(Int8Chunked::from_slice("", &[v as i8]).into_series())
                                } else {
                                    obj.extract::<i8>(py)
                                        .map(|v| Int8Chunked::from_slice("", &[v]).into_series())
                                }
                            },
                            Int16 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(Int16Chunked::from_slice("", &[v as i16]).into_series())
                                } else {
                                    obj.extract::<i16>(py)
                                        .map(|v| Int16Chunked::from_slice("", &[v]).into_series())
                                }
                            },
                            Int32 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(Int32Chunked::from_slice("", &[v as i32]).into_series())
                                } else {
                                    obj.extract::<i32>(py)
                                        .map(|v| Int32Chunked::from_slice("", &[v]).into_series())
                                }
                            },
                            Int64 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(Int64Chunked::from_slice("", &[v as i64]).into_series())
                                } else {
                                    obj.extract::<i64>(py)
                                        .map(|v| Int64Chunked::from_slice("", &[v]).into_series())
                                }
                            },
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
                            },
                        }
                    },
                }
            })
        };
        self.inner
            .clone()
            .rolling_map(Arc::new(function), GetOutput::same_type(), options)
            .with_fmt("rolling_map")
            .into()
    }
}
