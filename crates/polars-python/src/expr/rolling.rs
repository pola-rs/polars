use polars::prelude::*;
use polars_utils::python_function::PythonObject;
use pyo3::prelude::*;

use crate::PyExpr;
use crate::conversion::Wrap;
use crate::error::PyPolarsErr;

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
    ) -> PyResult<Self> {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::try_parse(window_size).map_err(PyPolarsErr::from)?,
            min_periods,
            closed_window: closed.0,
            fn_params: None,
        };
        Ok(self.inner.clone().rolling_sum_by(by.inner, options).into())
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
    ) -> PyResult<Self> {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::try_parse(window_size).map_err(PyPolarsErr::from)?,
            min_periods,
            closed_window: closed.0,
            fn_params: None,
        };
        Ok(self.inner.clone().rolling_min_by(by.inner, options).into())
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
    ) -> PyResult<Self> {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::try_parse(window_size).map_err(PyPolarsErr::from)?,
            min_periods,
            closed_window: closed.0,
            fn_params: None,
        };
        Ok(self.inner.clone().rolling_max_by(by.inner, options).into())
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
    ) -> PyResult<Self> {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::try_parse(window_size).map_err(PyPolarsErr::from)?,
            min_periods,
            closed_window: closed.0,
            fn_params: None,
        };

        Ok(self.inner.clone().rolling_mean_by(by.inner, options).into())
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
            fn_params: Some(RollingFnParams::Var(RollingVarParams { ddof })),
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
    ) -> PyResult<Self> {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::try_parse(window_size).map_err(PyPolarsErr::from)?,
            min_periods,
            closed_window: closed.0,
            fn_params: Some(RollingFnParams::Var(RollingVarParams { ddof })),
        };

        Ok(self.inner.clone().rolling_std_by(by.inner, options).into())
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
            fn_params: Some(RollingFnParams::Var(RollingVarParams { ddof })),
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
    ) -> PyResult<Self> {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::try_parse(window_size).map_err(PyPolarsErr::from)?,
            min_periods,
            closed_window: closed.0,
            fn_params: Some(RollingFnParams::Var(RollingVarParams { ddof })),
        };

        Ok(self.inner.clone().rolling_var_by(by.inner, options).into())
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
    ) -> PyResult<Self> {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::try_parse(window_size).map_err(PyPolarsErr::from)?,
            min_periods,
            closed_window: closed.0,
            fn_params: None,
        };
        Ok(self
            .inner
            .clone()
            .rolling_median_by(by.inner, options)
            .into())
    }

    #[pyo3(signature = (quantile, interpolation, window_size, weights, min_periods, center))]
    fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: Wrap<QuantileMethod>,
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
        interpolation: Wrap<QuantileMethod>,
        window_size: &str,
        min_periods: usize,
        closed: Wrap<ClosedWindow>,
    ) -> PyResult<Self> {
        let options = RollingOptionsDynamicWindow {
            window_size: Duration::try_parse(window_size).map_err(PyPolarsErr::from)?,
            min_periods,
            closed_window: closed.0,
            fn_params: None,
        };

        Ok(self
            .inner
            .clone()
            .rolling_quantile_by(by.inner, interpolation.0, quantile, options)
            .into())
    }

    #[pyo3(signature = (window_size, bias, min_periods, center))]
    fn rolling_skew(
        &self,
        window_size: usize,
        bias: bool,
        min_periods: Option<usize>,
        center: bool,
    ) -> Self {
        let min_periods = min_periods.unwrap_or(window_size);
        let options = RollingOptionsFixedWindow {
            window_size,
            weights: None,
            min_periods,
            center,
            fn_params: Some(RollingFnParams::Skew { bias }),
        };

        self.inner.clone().rolling_skew(options).into()
    }

    #[pyo3(signature = (window_size, fisher, bias, min_periods, center))]
    fn rolling_kurtosis(
        &self,
        window_size: usize,
        fisher: bool,
        bias: bool,
        min_periods: Option<usize>,
        center: bool,
    ) -> Self {
        let min_periods = min_periods.unwrap_or(window_size);
        let options = RollingOptionsFixedWindow {
            window_size,
            weights: None,
            min_periods,
            center,
            fn_params: Some(RollingFnParams::Kurtosis { fisher, bias }),
        };

        self.inner.clone().rolling_kurtosis(options).into()
    }

    #[pyo3(signature = (lambda, window_size, weights, min_periods, center))]
    fn rolling_map(
        &self,
        lambda: Py<PyAny>,
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
        let function = PlanCallback::new_python(PythonObject(lambda));

        self.inner.clone().rolling_map(function, options).into()
    }
}
