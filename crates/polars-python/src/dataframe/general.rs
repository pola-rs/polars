use std::mem::ManuallyDrop;

use either::Either;
use polars::export::arrow::bitmap::MutableBitmap;
use polars::prelude::*;
use polars_core::frame::*;
#[cfg(feature = "pivot")]
use polars_lazy::frame::pivot::{pivot, pivot_stable};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyList;

use super::PyDataFrame;
use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::map::dataframe::{
    apply_lambda_unknown, apply_lambda_with_bool_out_type, apply_lambda_with_primitive_out_type,
    apply_lambda_with_string_out_type,
};
use crate::prelude::strings_to_smartstrings;
use crate::series::{PySeries, ToPySeries, ToSeries};
use crate::{PyExpr, PyLazyFrame};

#[pymethods]
impl PyDataFrame {
    #[new]
    pub fn __init__(columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = columns.to_series();
        let df = DataFrame::new(columns).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn estimated_size(&self) -> usize {
        self.df.estimated_size()
    }

    pub fn dtype_strings(&self) -> Vec<String> {
        self.df
            .get_columns()
            .iter()
            .map(|s| format!("{}", s.dtype()))
            .collect()
    }

    pub fn add(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df + &s.series).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn sub(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df - &s.series).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn div(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df / &s.series).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn mul(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df * &s.series).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn rem(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df % &s.series).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn add_df(&self, s: &Self) -> PyResult<Self> {
        let df = (&self.df + &s.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn sub_df(&self, s: &Self) -> PyResult<Self> {
        let df = (&self.df - &s.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn div_df(&self, s: &Self) -> PyResult<Self> {
        let df = (&self.df / &s.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn mul_df(&self, s: &Self) -> PyResult<Self> {
        let df = (&self.df * &s.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn rem_df(&self, s: &Self) -> PyResult<Self> {
        let df = (&self.df % &s.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn sample_n(
        &self,
        n: &PySeries,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let df = self
            .df
            .sample_n(&n.series, with_replacement, shuffle, seed)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn sample_frac(
        &self,
        frac: &PySeries,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let df = self
            .df
            .sample_frac(&frac.series, with_replacement, shuffle, seed)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn rechunk(&self, py: Python) -> Self {
        let mut df = self.df.clone();
        py.allow_threads(|| df.as_single_chunk_par());
        df.into()
    }

    /// Format `DataFrame` as String
    pub fn as_str(&self) -> String {
        format!("{:?}", self.df)
    }

    pub fn get_columns(&self) -> Vec<PySeries> {
        let cols = self.df.get_columns().to_vec();
        cols.to_pyseries()
    }

    /// Get column names
    pub fn columns(&self) -> Vec<&str> {
        self.df.get_column_names()
    }

    /// set column names
    pub fn set_column_names(&mut self, names: Vec<PyBackedStr>) -> PyResult<()> {
        self.df
            .set_column_names(&names)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    /// Get datatypes
    pub fn dtypes(&self, py: Python) -> PyObject {
        let iter = self
            .df
            .iter()
            .map(|s| Wrap(s.dtype().clone()).to_object(py));
        PyList::new_bound(py, iter).to_object(py)
    }

    pub fn n_chunks(&self) -> usize {
        self.df.n_chunks()
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

    pub fn is_empty(&self) -> bool {
        self.df.is_empty()
    }

    pub fn hstack(&self, columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = columns.to_series();
        let df = self.df.hstack(&columns).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn hstack_mut(&mut self, columns: Vec<PySeries>) -> PyResult<()> {
        let columns = columns.to_series();
        self.df.hstack_mut(&columns).map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn vstack(&self, other: &PyDataFrame) -> PyResult<Self> {
        let df = self.df.vstack(&other.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn vstack_mut(&mut self, other: &PyDataFrame) -> PyResult<()> {
        self.df.vstack_mut(&other.df).map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn extend(&mut self, other: &PyDataFrame) -> PyResult<()> {
        self.df.extend(&other.df).map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn drop_in_place(&mut self, name: &str) -> PyResult<PySeries> {
        let s = self.df.drop_in_place(name).map_err(PyPolarsErr::from)?;
        Ok(PySeries { series: s })
    }

    pub fn to_series(&self, index: isize) -> PyResult<PySeries> {
        let df = &self.df;

        let index_adjusted = if index < 0 {
            df.width().checked_sub(index.unsigned_abs())
        } else {
            Some(usize::try_from(index).unwrap())
        };

        let s = index_adjusted.and_then(|i| df.select_at_idx(i));
        match s {
            Some(s) => Ok(PySeries::new(s.clone())),
            None => Err(PyIndexError::new_err(
                polars_err!(oob = index, df.width()).to_string(),
            )),
        }
    }

    pub fn get_column_index(&self, name: &str) -> PyResult<usize> {
        Ok(self
            .df
            .try_get_column_index(name)
            .map_err(PyPolarsErr::from)?)
    }

    pub fn get_column(&self, name: &str) -> PyResult<PySeries> {
        let series = self
            .df
            .column(name)
            .map(|s| PySeries::new(s.clone()))
            .map_err(PyPolarsErr::from)?;
        Ok(series)
    }

    pub fn select(&self, columns: Vec<PyBackedStr>) -> PyResult<Self> {
        let df = self.df.select(columns).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn gather(&self, indices: Wrap<Vec<IdxSize>>) -> PyResult<Self> {
        let indices = indices.0;
        let indices = IdxCa::from_vec("", indices);
        let df = self.df.take(&indices).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn gather_with_series(&self, indices: &PySeries) -> PyResult<Self> {
        let indices = indices.series.idx().map_err(PyPolarsErr::from)?;
        let df = self.df.take(indices).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn replace(&mut self, column: &str, new_col: PySeries) -> PyResult<()> {
        self.df
            .replace(column, new_col.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn replace_column(&mut self, index: usize, new_column: PySeries) -> PyResult<()> {
        self.df
            .replace_column(index, new_column.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn insert_column(&mut self, index: usize, column: PySeries) -> PyResult<()> {
        self.df
            .insert_column(index, column.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn slice(&self, offset: i64, length: Option<usize>) -> Self {
        let df = self
            .df
            .slice(offset, length.unwrap_or_else(|| self.df.height()));
        df.into()
    }

    pub fn head(&self, n: usize) -> Self {
        let df = self.df.head(Some(n));
        PyDataFrame::new(df)
    }

    pub fn tail(&self, n: usize) -> Self {
        let df = self.df.tail(Some(n));
        PyDataFrame::new(df)
    }

    pub fn is_unique(&self) -> PyResult<PySeries> {
        let mask = self.df.is_unique().map_err(PyPolarsErr::from)?;
        Ok(mask.into_series().into())
    }

    pub fn is_duplicated(&self) -> PyResult<PySeries> {
        let mask = self.df.is_duplicated().map_err(PyPolarsErr::from)?;
        Ok(mask.into_series().into())
    }

    pub fn equals(&self, other: &PyDataFrame, null_equal: bool) -> bool {
        if null_equal {
            self.df.equals_missing(&other.df)
        } else {
            self.df.equals(&other.df)
        }
    }

    pub fn with_row_index(&self, name: &str, offset: Option<IdxSize>) -> PyResult<Self> {
        let df = self
            .df
            .with_row_index(name, offset)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn group_by_map_groups(
        &self,
        by: Vec<PyBackedStr>,
        lambda: PyObject,
        maintain_order: bool,
    ) -> PyResult<Self> {
        let gb = if maintain_order {
            self.df.group_by_stable(&by)
        } else {
            self.df.group_by(&by)
        }
        .map_err(PyPolarsErr::from)?;

        let function = move |df: DataFrame| {
            Python::with_gil(|py| {
                let pypolars = PyModule::import_bound(py, "polars").unwrap();
                let pydf = PyDataFrame::new(df);
                let python_df_wrapper =
                    pypolars.getattr("wrap_df").unwrap().call1((pydf,)).unwrap();

                // Call the lambda and get a python-side DataFrame wrapper.
                let result_df_wrapper = match lambda.call1(py, (python_df_wrapper,)) {
                    Ok(pyobj) => pyobj,
                    Err(e) => panic!("UDF failed: {}", e.value_bound(py)),
                };
                let py_pydf = result_df_wrapper.getattr(py, "_df").expect(
                    "Could not get DataFrame attribute '_df'. Make sure that you return a DataFrame object.",
                );

                let pydf = py_pydf.extract::<PyDataFrame>(py).unwrap();
                Ok(pydf.df)
            })
        };
        // We don't use `py.allow_threads(|| gb.par_apply(..)` because that segfaulted
        // due to code related to Pyo3 or rayon, cannot reproduce it in native polars.
        // So we lose parallelism, but it doesn't really matter because we are GIL bound anyways
        // and this function should not be used in idiomatic polars anyway.
        let df = gb.apply(function).map_err(PyPolarsErr::from)?;

        Ok(df.into())
    }

    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Self {
        PyDataFrame::new(self.df.clone())
    }

    #[cfg(feature = "pivot")]
    pub fn unpivot(
        &self,
        on: Vec<PyBackedStr>,
        index: Vec<PyBackedStr>,
        value_name: Option<&str>,
        variable_name: Option<&str>,
    ) -> PyResult<Self> {
        use polars_ops::pivot::UnpivotDF;
        let args = UnpivotArgsIR {
            on: strings_to_smartstrings(on),
            index: strings_to_smartstrings(index),
            value_name: value_name.map(|s| s.into()),
            variable_name: variable_name.map(|s| s.into()),
        };

        let df = self.df.unpivot2(args).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[cfg(feature = "pivot")]
    #[pyo3(signature = (on, index, values, maintain_order, sort_columns, aggregate_expr, separator))]
    pub fn pivot_expr(
        &self,
        on: Vec<String>,
        index: Option<Vec<String>>,
        values: Option<Vec<String>>,
        maintain_order: bool,
        sort_columns: bool,
        aggregate_expr: Option<PyExpr>,
        separator: Option<&str>,
    ) -> PyResult<Self> {
        let fun = if maintain_order { pivot_stable } else { pivot };
        let agg_expr = aggregate_expr.map(|expr| expr.inner);
        let df = fun(
            &self.df,
            on,
            index,
            values,
            sort_columns,
            agg_expr,
            separator,
        )
        .map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn partition_by(
        &self,
        by: Vec<String>,
        maintain_order: bool,
        include_key: bool,
    ) -> PyResult<Vec<Self>> {
        let out = if maintain_order {
            self.df.partition_by_stable(by, include_key)
        } else {
            self.df.partition_by(by, include_key)
        }
        .map_err(PyPolarsErr::from)?;

        // SAFETY: PyDataFrame is a repr(transparent) DataFrame.
        Ok(unsafe { std::mem::transmute::<Vec<DataFrame>, Vec<PyDataFrame>>(out) })
    }

    pub fn lazy(&self) -> PyLazyFrame {
        self.df.clone().lazy().into()
    }

    pub fn max_horizontal(&self) -> PyResult<Option<PySeries>> {
        let s = self.df.max_horizontal().map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn min_horizontal(&self) -> PyResult<Option<PySeries>> {
        let s = self.df.min_horizontal().map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn sum_horizontal(&self, ignore_nulls: bool) -> PyResult<Option<PySeries>> {
        let null_strategy = if ignore_nulls {
            NullStrategy::Ignore
        } else {
            NullStrategy::Propagate
        };
        let s = self
            .df
            .sum_horizontal(null_strategy)
            .map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn mean_horizontal(&self, ignore_nulls: bool) -> PyResult<Option<PySeries>> {
        let null_strategy = if ignore_nulls {
            NullStrategy::Ignore
        } else {
            NullStrategy::Propagate
        };
        let s = self
            .df
            .mean_horizontal(null_strategy)
            .map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    #[pyo3(signature = (columns, separator, drop_first=false))]
    pub fn to_dummies(
        &self,
        columns: Option<Vec<String>>,
        separator: Option<&str>,
        drop_first: bool,
    ) -> PyResult<Self> {
        let df = match columns {
            Some(cols) => self.df.columns_to_dummies(
                cols.iter().map(|x| x as &str).collect(),
                separator,
                drop_first,
            ),
            None => self.df.to_dummies(separator, drop_first),
        }
        .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn null_count(&self) -> Self {
        let df = self.df.null_count();
        df.into()
    }

    #[pyo3(signature = (lambda, output_type, inference_size))]
    pub fn map_rows(
        &mut self,
        lambda: Bound<PyAny>,
        output_type: Option<Wrap<DataType>>,
        inference_size: usize,
    ) -> PyResult<(PyObject, bool)> {
        Python::with_gil(|py| {
            // needed for series iter
            self.df.as_single_chunk_par();
            let df = &self.df;

            use apply_lambda_with_primitive_out_type as apply;
            #[rustfmt::skip]
            let out = match output_type.map(|dt| dt.0) {
                Some(DataType::Int32) => apply::<Int32Type>(df, py, lambda, 0, None).into_series(),
                Some(DataType::Int64) => apply::<Int64Type>(df, py, lambda, 0, None).into_series(),
                Some(DataType::UInt32) => apply::<UInt32Type>(df, py, lambda, 0, None).into_series(),
                Some(DataType::UInt64) => apply::<UInt64Type>(df, py, lambda, 0, None).into_series(),
                Some(DataType::Float32) => apply::<Float32Type>(df, py, lambda, 0, None).into_series(),
                Some(DataType::Float64) => apply::<Float64Type>(df, py, lambda, 0, None).into_series(),
                Some(DataType::Date) => apply::<Int32Type>(df, py, lambda, 0, None).into_date().into_series(),
                Some(DataType::Datetime(tu, tz)) => apply::<Int64Type>(df, py, lambda, 0, None).into_datetime(tu, tz).into_series(),
                Some(DataType::Boolean) => apply_lambda_with_bool_out_type(df, py, lambda, 0, None).into_series(),
                Some(DataType::String) => apply_lambda_with_string_out_type(df, py, lambda, 0, None).into_series(),
                _ => return apply_lambda_unknown(df, py, lambda, inference_size),
            };

            Ok((PySeries::from(out).into_py(py), false))
        })
    }

    pub fn shrink_to_fit(&mut self) {
        self.df.shrink_to_fit();
    }

    pub fn hash_rows(&mut self, k0: u64, k1: u64, k2: u64, k3: u64) -> PyResult<PySeries> {
        let hb = PlRandomState::with_seeds(k0, k1, k2, k3);
        let hash = self.df.hash_rows(Some(hb)).map_err(PyPolarsErr::from)?;
        Ok(hash.into_series().into())
    }

    #[pyo3(signature = (keep_names_as, column_names))]
    pub fn transpose(
        &mut self,
        keep_names_as: Option<&str>,
        column_names: &Bound<PyAny>,
    ) -> PyResult<Self> {
        let new_col_names = if let Ok(name) = column_names.extract::<Vec<String>>() {
            Some(Either::Right(name))
        } else if let Ok(name) = column_names.extract::<String>() {
            Some(Either::Left(name))
        } else {
            None
        };
        Ok(self
            .df
            .transpose(keep_names_as, new_col_names)
            .map_err(PyPolarsErr::from)?
            .into())
    }
    pub fn upsample(
        &self,
        by: Vec<String>,
        index_column: &str,
        every: &str,
        stable: bool,
    ) -> PyResult<Self> {
        let out = if stable {
            self.df
                .upsample_stable(by, index_column, Duration::parse(every))
        } else {
            self.df.upsample(by, index_column, Duration::parse(every))
        };
        let out = out.map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    pub fn to_struct(&self, name: &str, invalid_indices: Vec<usize>) -> PySeries {
        let ca = self.df.clone().into_struct(name);

        if !invalid_indices.is_empty() {
            let mut validity = MutableBitmap::with_capacity(ca.len());
            validity.extend_constant(ca.len(), true);
            for i in invalid_indices {
                validity.set(i, false);
            }
            let ca = ca.rechunk();
            ca.with_outer_validity(Some(validity.freeze()))
                .into_series()
                .into()
        } else {
            ca.into_series().into()
        }
    }

    pub fn unnest(&self, columns: Vec<String>) -> PyResult<Self> {
        let df = self.df.unnest(columns).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn clear(&self) -> Self {
        self.df.clear().into()
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn into_raw_parts(&mut self) -> (usize, usize, usize) {
        // Used for polars-lazy python node. This takes the dataframe from
        // underneath of you, so don't use this anywhere else.
        let mut df = std::mem::take(&mut self.df);
        let cols = unsafe { std::mem::take(df.get_columns_mut()) };
        let mut md_cols = ManuallyDrop::new(cols);
        let ptr = md_cols.as_mut_ptr();
        let len = md_cols.len();
        let cap = md_cols.capacity();
        (ptr as usize, len, cap)
    }
}
