use std::panic::AssertUnwindSafe;

use polars::frame::DataFrame;
use polars::series::IntoSeries;
use polars_error::signals::{catch_keyboard_interrupt, KeyboardInterrupt};
use polars_error::PolarsResult;
use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::marker::Ungil;
use pyo3::{PyErr, PyResult, Python};

use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::series::PySeries;

// was redefined because I could not get feature flags activated?
#[macro_export]
macro_rules! apply_method_all_arrow_series2 {
    ($self:expr, $method:ident, $($args:expr),*) => {
        match $self.dtype() {
            DataType::Boolean => $self.bool().unwrap().$method($($args),*),
            DataType::String => $self.str().unwrap().$method($($args),*),
            DataType::UInt8 => $self.u8().unwrap().$method($($args),*),
            DataType::UInt16 => $self.u16().unwrap().$method($($args),*),
            DataType::UInt32 => $self.u32().unwrap().$method($($args),*),
            DataType::UInt64 => $self.u64().unwrap().$method($($args),*),
            DataType::Int8 => $self.i8().unwrap().$method($($args),*),
            DataType::Int16 => $self.i16().unwrap().$method($($args),*),
            DataType::Int32 => $self.i32().unwrap().$method($($args),*),
            DataType::Int64 => $self.i64().unwrap().$method($($args),*),
            DataType::Int128 => $self.i128().unwrap().$method($($args),*),
            DataType::Float32 => $self.f32().unwrap().$method($($args),*),
            DataType::Float64 => $self.f64().unwrap().$method($($args),*),
            DataType::Date => $self.date().unwrap().$method($($args),*),
            DataType::Datetime(_, _) => $self.datetime().unwrap().$method($($args),*),
            DataType::List(_) => $self.list().unwrap().$method($($args),*),
            DataType::Struct(_) => $self.struct_().unwrap().$method($($args),*),
            dt => panic!("dtype {:?} not supported", dt)
        }
    }
}

/// Boilerplate for `|e| PyPolarsErr::from(e).into()`
#[allow(unused)]
pub(crate) fn to_py_err<E: Into<PyPolarsErr>>(e: E) -> PyErr {
    e.into().into()
}

pub trait EnterPolarsExt {
    /// Whenever you have a block of code in the public Python API that
    /// (potentially) takes a long time, wrap it in enter_polars. This will
    /// ensure we release the GIL and catch KeyboardInterrupts.
    ///
    /// This not only can increase performance and usability, it can avoid
    /// deadlocks on the GIL for Python UDFs.
    fn enter_polars<T, E, F>(self, f: F) -> PyResult<T>
    where
        F: Ungil + Send + FnOnce() -> Result<T, E>,
        T: Ungil + Send,
        E: Ungil + Send + Into<PyPolarsErr>;

    /// Same as enter_polars, but wraps the result in PyResult::Ok, useful
    /// shorthand for infallible functions.
    #[inline(always)]
    fn enter_polars_ok<T, F>(self, f: F) -> PyResult<T>
    where
        Self: Sized,
        F: Ungil + Send + FnOnce() -> T,
        T: Ungil + Send,
    {
        self.enter_polars(move || PyResult::Ok(f()))
    }

    /// Same as enter_polars, but expects a PolarsResult<DataFrame> as return
    /// which is converted to a PyDataFrame.
    #[inline(always)]
    fn enter_polars_df<F>(self, f: F) -> PyResult<PyDataFrame>
    where
        Self: Sized,
        F: Ungil + Send + FnOnce() -> PolarsResult<DataFrame>,
    {
        self.enter_polars(f).map(PyDataFrame::new)
    }

    /// Same as enter_polars, but expects a PolarsResult<S> as return which
    /// is converted to a PySeries through S: IntoSeries.
    #[inline(always)]
    fn enter_polars_series<T, F>(self, f: F) -> PyResult<PySeries>
    where
        Self: Sized,
        T: Ungil + Send + IntoSeries,
        F: Ungil + Send + FnOnce() -> PolarsResult<T>,
    {
        self.enter_polars(f).map(|s| PySeries::new(s.into_series()))
    }
}

impl EnterPolarsExt for Python<'_> {
    fn enter_polars<T, E, F>(self, f: F) -> PyResult<T>
    where
        F: Ungil + Send + FnOnce() -> Result<T, E>,
        T: Ungil + Send,
        E: Ungil + Send + Into<PyPolarsErr>,
    {
        self.allow_threads(|| match catch_keyboard_interrupt(AssertUnwindSafe(f)) {
            Ok(Ok(ret)) => Ok(ret),
            Ok(Err(err)) => Err(PyErr::from(err.into())),
            Err(KeyboardInterrupt) => Err(PyKeyboardInterrupt::new_err("")),
        })
    }
}
