use polars::prelude::*;
use pyo3::{exceptions::RuntimeError, prelude::*};

use crate::datatypes::DataType;
use crate::{
    error::PyPolarsEr,
    file::{get_either_file, get_file_like, EitherRustPythonFile},
    series::{to_pyseries_collection, to_series_collection, PySeries},
};

#[pyclass]
#[repr(transparent)]
pub struct PyDataFrame {
    pub df: DataFrame,
}

impl PyDataFrame {
    fn new(df: DataFrame) -> Self {
        PyDataFrame { df }
    }
}

#[pymethods]
impl PyDataFrame {
    #[new]
    pub fn __init__(columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = to_series_collection(columns);
        let df = DataFrame::new(columns).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    pub fn read_csv(
        py_f: PyObject,
        infer_schema_length: usize,
        batch_size: usize,
        has_header: bool,
        ignore_errors: bool,
    ) -> PyResult<Self> {
        let file = get_file_like(py_f, false)?;
        let reader = CsvReader::new(file)
            .infer_schema(Some(infer_schema_length))
            .has_header(has_header)
            .with_batch_size(batch_size);

        let reader = if ignore_errors {
            reader.with_ignore_parser_error()
        } else {
            reader
        };
        let df = reader.finish().map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    pub fn read_parquet(py_f: PyObject, batch_size: usize) -> PyResult<Self> {
        use EitherRustPythonFile::*;
        let result = match get_either_file(py_f, false)? {
            Py(f) => ParquetReader::new(f).with_batch_size(batch_size).finish(),
            Rust(f) => ParquetReader::new(f).with_batch_size(batch_size).finish(),
        };
        let df = result.map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    pub fn read_ipc(py_f: PyObject) -> PyResult<Self> {
        let file = get_file_like(py_f, false)?;
        let df = IPCReader::new(file).finish().map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn to_csv(
        &mut self,
        py_f: PyObject,
        batch_size: usize,
        has_headers: bool,
        delimiter: u8,
    ) -> PyResult<()> {
        let mut buf = get_file_like(py_f, true)?;
        CsvWriter::new(&mut buf)
            .has_headers(has_headers)
            .with_delimiter(delimiter)
            .with_batch_size(batch_size)
            .finish(&mut self.df)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn to_ipc(&mut self, py_f: PyObject, batch_size: usize) -> PyResult<()> {
        let mut buf = get_file_like(py_f, true)?;
        IPCWriter::new(&mut buf)
            .with_batch_size(batch_size)
            .finish(&mut self.df)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    /// Format `DataFrame` as String
    pub fn as_str(&self) -> String {
        format!("{:?}", self.df)
    }

    pub fn with_parallel(&mut self, parallel: bool) {
        self.df.with_parallel(parallel);
    }

    pub fn fill_none(&self, strategy: &str) -> PyResult<Self> {
        let strat = match strategy {
            "backward" => FillNoneStrategy::Backward,
            "forward" => FillNoneStrategy::Forward,
            "min" => FillNoneStrategy::Min,
            "max" => FillNoneStrategy::Max,
            "mean" => FillNoneStrategy::Mean,
            s => return Err(PyPolarsEr::Other(format!("Strategy {} not supported", s)).into()),
        };
        let df = self.df.fill_none(strat).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn inner_join(&self, other: &PyDataFrame, left_on: &str, right_on: &str) -> PyResult<Self> {
        let df = self
            .df
            .inner_join(&other.df, left_on, right_on)
            .map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn left_join(&self, other: &PyDataFrame, left_on: &str, right_on: &str) -> PyResult<Self> {
        let df = self
            .df
            .left_join(&other.df, left_on, right_on)
            .map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn outer_join(&self, other: &PyDataFrame, left_on: &str, right_on: &str) -> PyResult<Self> {
        let df = self
            .df
            .outer_join(&other.df, left_on, right_on)
            .map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn get_columns(&self) -> Vec<PySeries> {
        let cols = self.df.get_columns().clone();
        to_pyseries_collection(cols)
    }

    /// Get column names
    pub fn columns(&self) -> Vec<&str> {
        self.df.columns()
    }

    /// Get datatypes
    pub fn dtypes(&self) -> Vec<u8> {
        self.df
            .dtypes()
            .iter()
            .map(|arrow_dtype| {
                let dt: DataType = arrow_dtype.into();
                dt as u8
            })
            .collect()
    }

    pub fn n_chunks(&self) -> PyResult<usize> {
        let n = self.df.n_chunks().map_err(PyPolarsEr::from)?;
        Ok(n)
    }

    pub fn shape(&self) -> (usize, usize) {
        self.df.shape()
    }

    pub fn height(&self) -> usize {
        self.df.height()
    }

    pub fn width(&self) -> usize {
        self.df.width()
    }

    pub fn hstack(&mut self, columns: Vec<PySeries>) -> PyResult<()> {
        let columns = to_series_collection(columns);
        self.df.hstack(&columns).map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn vstack(&mut self, df: &PyDataFrame) -> PyResult<()> {
        self.df.vstack(&df.df).map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn drop_in_place(&mut self, name: &str) -> PyResult<PySeries> {
        let s = self.df.drop_in_place(name).map_err(PyPolarsEr::from)?;
        Ok(PySeries { series: s })
    }

    pub fn drop(&self, name: &str) -> PyResult<Self> {
        let df = self.df.drop(name).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn select_at_idx(&self, idx: usize) -> Option<PySeries> {
        self.df.select_at_idx(idx).map(|s| PySeries::new(s.clone()))
    }

    pub fn find_idx_by_name(&self, name: &str) -> Option<usize> {
        self.df.find_idx_by_name(name)
    }

    pub fn column(&self, name: &str) -> PyResult<PySeries> {
        let series = self
            .df
            .column(name)
            .map(|s| PySeries::new(s.clone()))
            .map_err(PyPolarsEr::from)?;
        Ok(series)
    }

    pub fn select(&self, selection: Vec<&str>) -> PyResult<Self> {
        let df = self.df.select(&selection).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn filter(&self, mask: &PySeries) -> PyResult<Self> {
        let filter_series = &mask.series;
        if let Series::Bool(ca) = filter_series {
            let df = self.df.filter(ca).map_err(PyPolarsEr::from)?;
            Ok(PyDataFrame::new(df))
        } else {
            Err(RuntimeError::py_err("Expected a boolean mask"))
        }
    }

    pub fn take(&self, indices: Vec<usize>) -> PyResult<Self> {
        let df = self.df.take(&indices).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn take_with_series(&self, indices: &PySeries) -> PyResult<Self> {
        let idx = indices.series.u32().map_err(PyPolarsEr::from)?;
        let df = self.df.take(&idx).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn sort(&self, by_column: &str, reverse: bool) -> PyResult<Self> {
        let df = self.df.sort(by_column, reverse).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn sort_in_place(&mut self, by_column: &str, reverse: bool) -> PyResult<()> {
        self.df
            .sort_in_place(by_column, reverse)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn replace(&mut self, column: &str, new_col: PySeries) -> PyResult<()> {
        self.df
            .replace(column, new_col.series)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn replace_at_idx(&mut self, index: usize, new_col: PySeries) -> PyResult<()> {
        self.df
            .replace_at_idx(index, new_col.series)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn slice(&self, offset: usize, length: usize) -> PyResult<Self> {
        let df = self.df.slice(offset, length).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn head(&self, length: Option<usize>) -> Self {
        let df = self.df.head(length);
        PyDataFrame::new(df)
    }

    pub fn tail(&self, length: Option<usize>) -> Self {
        let df = self.df.tail(length);
        PyDataFrame::new(df)
    }

    pub fn frame_equal(&self, other: &PyDataFrame) -> bool {
        self.df.frame_equal(&other.df)
    }

    pub fn groupby(&self, by: Vec<&str>, select: Option<Vec<String>>, agg: &str) -> PyResult<Self> {
        let gb = self.df.groupby(&by).map_err(PyPolarsEr::from)?;
        let selection = match select.as_ref() {
            Some(s) => gb.select(s),
            None => gb,
        };
        let df = match agg {
            "min" => selection.min(),
            "max" => selection.max(),
            "mean" => selection.mean(),
            "first" => selection.first(),
            "last" => selection.last(),
            "sum" => selection.sum(),
            "count" => selection.count(),
            "n_unique" => selection.n_unique(),
            "median" => selection.median(),
            "agg_list" => selection.agg_list(),
            "groups" => selection.groups(),
            a => Err(PolarsError::Other(
                format!("agg fn {} does not exists", a).into(),
            )),
        };
        let df = df.map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn groupby_agg(
        &self,
        by: Vec<&str>,
        column_to_agg: Vec<(&str, Vec<&str>)>,
    ) -> PyResult<Self> {
        let gb = self.df.groupby(&by).map_err(PyPolarsEr::from)?;
        let df = gb.agg(&column_to_agg).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn groupby_quantile(
        &self,
        by: Vec<&str>,
        select: Vec<String>,
        quantile: f64,
    ) -> PyResult<Self> {
        let gb = self.df.groupby(&by).map_err(PyPolarsEr::from)?;
        let selection = gb.select(&select);
        let df = selection.quantile(quantile);
        let df = df.map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn pivot(
        &self,
        by: Vec<String>,
        pivot_column: &str,
        values_column: &str,
        agg: &str,
    ) -> PyResult<Self> {
        let mut gb = self.df.groupby(&by).map_err(PyPolarsEr::from)?;
        let pivot = gb.pivot(pivot_column, values_column);
        let df = match agg {
            "first" => pivot.first(),
            "min" => pivot.min(),
            "max" => pivot.max(),
            "mean" => pivot.mean(),
            "median" => pivot.median(),
            "sum" => pivot.sum(),
            a => Err(PolarsError::Other(
                format!("agg fn {} does not exists", a).into(),
            )),
        };
        let df = df.map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn clone(&self) -> Self {
        PyDataFrame::new(self.df.clone())
    }
}
