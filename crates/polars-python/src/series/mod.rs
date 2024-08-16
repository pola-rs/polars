mod aggregation;
mod arithmetic;
mod buffers;
mod c_interface;
mod comparison;
mod construction;
mod export;
mod general;
mod import;
mod numpy_ufunc;
mod scatter;

use std::io::Cursor;

use polars_core::chunked_array::cast::CastOptions;
use polars_core::series::IsSorted;
use polars_core::utils::flatten::flatten_series;
use polars_core::{with_match_physical_numeric_polars_type, with_match_physical_numeric_type};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::Python;

use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::map::series::{call_lambda_and_extract, ApplyLambda};
use crate::prelude::*;
use crate::py_modules::POLARS;
use crate::raise_err;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PySeries {
    pub series: Series,
}

impl From<Series> for PySeries {
    fn from(series: Series) -> Self {
        PySeries { series }
    }
}

impl PySeries {
    pub(crate) fn new(series: Series) -> Self {
        PySeries { series }
    }
}

pub(crate) trait ToSeries {
    fn to_series(self) -> Vec<Series>;
}

impl ToSeries for Vec<PySeries> {
    fn to_series(self) -> Vec<Series> {
        // SAFETY: repr is transparent.
        unsafe { std::mem::transmute(self) }
    }
}

pub(crate) trait ToPySeries {
    fn to_pyseries(self) -> Vec<PySeries>;
}

impl ToPySeries for Vec<Series> {
    fn to_pyseries(self) -> Vec<PySeries> {
        // SAFETY: repr is transparent.
        unsafe { std::mem::transmute(self) }
    }
}
