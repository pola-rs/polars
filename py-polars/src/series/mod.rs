mod aggregation;
mod arithmetic;
mod buffers;
mod comparison;
mod construction;
mod export;
mod numpy_ufunc;
mod scatter;

use std::io::Cursor;

use polars_core::series::IsSorted;
use polars_core::utils::flatten::flatten_series;
use polars_core::with_match_physical_numeric_polars_type;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::Python;

use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::map::series::{call_lambda_and_extract, ApplyLambda};
use crate::prelude::*;
use crate::py_modules::POLARS;
use crate::{apply_method_all_arrow_series2, raise_err};

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
        // Safety
        // repr is transparent
        unsafe { std::mem::transmute(self) }
    }
}

pub(crate) trait ToPySeries {
    fn to_pyseries(self) -> Vec<PySeries>;
}

impl ToPySeries for Vec<Series> {
    fn to_pyseries(self) -> Vec<PySeries> {
        // Safety
        // repr is transparent
        unsafe { std::mem::transmute(self) }
    }
}

#[pymethods]
impl PySeries {
    fn struct_unnest(&self) -> PyResult<PyDataFrame> {
        let ca = self.series.struct_().map_err(PyPolarsErr::from)?;
        let df: DataFrame = ca.clone().into();
        Ok(df.into())
    }

    fn struct_fields(&self) -> PyResult<Vec<&str>> {
        let ca = self.series.struct_().map_err(PyPolarsErr::from)?;
        Ok(ca.fields().iter().map(|s| s.name()).collect())
    }

    fn is_sorted_ascending_flag(&self) -> bool {
        matches!(self.series.is_sorted_flag(), IsSorted::Ascending)
    }

    fn is_sorted_descending_flag(&self) -> bool {
        matches!(self.series.is_sorted_flag(), IsSorted::Descending)
    }

    fn can_fast_explode_flag(&self) -> bool {
        match self.series.list() {
            Err(_) => false,
            Ok(list) => list._can_fast_explode(),
        }
    }

    pub fn cat_uses_lexical_ordering(&self) -> PyResult<bool> {
        let ca = self.series.categorical().map_err(PyPolarsErr::from)?;
        Ok(ca.uses_lexical_ordering())
    }

    pub fn cat_is_local(&self) -> PyResult<bool> {
        let ca = self.series.categorical().map_err(PyPolarsErr::from)?;
        Ok(ca.get_rev_map().is_local())
    }

    pub fn cat_to_local(&self) -> PyResult<Self> {
        let ca = self.series.categorical().map_err(PyPolarsErr::from)?;
        Ok(ca.to_local().into_series().into())
    }

    fn estimated_size(&self) -> usize {
        self.series.estimated_size()
    }

    #[cfg(feature = "object")]
    fn get_object(&self, index: usize) -> PyObject {
        Python::with_gil(|py| {
            if matches!(self.series.dtype(), DataType::Object(_)) {
                let obj: Option<&ObjectValue> = self.series.get_object(index).map(|any| any.into());
                obj.to_object(py)
            } else {
                py.None()
            }
        })
    }

    fn get_fmt(&self, index: usize, str_lengths: usize) -> String {
        let val = format!("{}", self.series.get(index).unwrap());
        if let DataType::Utf8 | DataType::Categorical(_, _) = self.series.dtype() {
            let v_trunc = &val[..val
                .char_indices()
                .take(str_lengths)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0)];
            if val == v_trunc {
                val
            } else {
                format!("{v_trunc}…")
            }
        } else {
            val
        }
    }

    fn rechunk(&mut self, in_place: bool) -> Option<Self> {
        let series = self.series.rechunk();
        if in_place {
            self.series = series;
            None
        } else {
            Some(series.into())
        }
    }

    fn get_index(&self, py: Python, index: usize) -> PyResult<PyObject> {
        let av = match self.series.get(index) {
            Ok(v) => v,
            Err(PolarsError::OutOfBounds(err)) => {
                return Err(PyIndexError::new_err(err.to_string()))
            },
            Err(e) => return Err(PyPolarsErr::from(e).into()),
        };

        let out = match av {
            AnyValue::List(s) | AnyValue::Array(s, _) => {
                let pyseries = PySeries::new(s);
                let out = POLARS
                    .getattr(py, "wrap_s")
                    .unwrap()
                    .call1(py, (pyseries,))
                    .unwrap();
                out.into_py(py)
            },
            _ => Wrap(av).into_py(py),
        };

        Ok(out)
    }

    /// Get index but allow negative indices
    fn get_index_signed(&self, py: Python, index: i64) -> PyResult<PyObject> {
        let index = if index < 0 {
            match self.len().checked_sub(index.unsigned_abs() as usize) {
                Some(v) => v,
                None => {
                    return Err(PyIndexError::new_err(
                        polars_err!(oob = index, self.len()).to_string(),
                    ));
                },
            }
        } else {
            index as usize
        };
        self.get_index(py, index)
    }

    fn bitand(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .bitand(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn bitor(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .bitor(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }
    fn bitxor(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .bitxor(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn chunk_lengths(&self) -> Vec<usize> {
        self.series.chunk_lengths().collect()
    }

    fn name(&self) -> &str {
        self.series.name()
    }

    fn rename(&mut self, name: &str) {
        self.series.rename(name);
    }

    fn dtype(&self, py: Python) -> PyObject {
        Wrap(self.series.dtype().clone()).to_object(py)
    }

    fn set_sorted_flag(&self, descending: bool) -> Self {
        let mut out = self.series.clone();
        if descending {
            out.set_sorted_flag(IsSorted::Descending);
        } else {
            out.set_sorted_flag(IsSorted::Ascending)
        }
        out.into()
    }

    fn n_chunks(&self) -> usize {
        self.series.n_chunks()
    }

    fn append(&mut self, other: &PySeries) -> PyResult<()> {
        self.series
            .append(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    fn extend(&mut self, other: &PySeries) -> PyResult<()> {
        self.series
            .extend(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    fn new_from_index(&self, index: usize, length: usize) -> PyResult<Self> {
        if index >= self.series.len() {
            Err(PyValueError::new_err("index is out of bounds"))
        } else {
            Ok(self.series.new_from_index(index, length).into())
        }
    }

    fn filter(&self, filter: &PySeries) -> PyResult<Self> {
        let filter_series = &filter.series;
        if let Ok(ca) = filter_series.bool() {
            let series = self.series.filter(ca).map_err(PyPolarsErr::from)?;
            Ok(PySeries { series })
        } else {
            Err(PyRuntimeError::new_err("Expected a boolean mask"))
        }
    }

    fn sort(&mut self, descending: bool) -> Self {
        self.series.sort(descending).into()
    }

    fn take_with_series(&self, indices: &PySeries) -> PyResult<Self> {
        let idx = indices.series.idx().map_err(PyPolarsErr::from)?;
        let take = self.series.take(idx).map_err(PyPolarsErr::from)?;
        Ok(take.into())
    }

    fn null_count(&self) -> PyResult<usize> {
        Ok(self.series.null_count())
    }

    fn has_validity(&self) -> bool {
        self.series.has_validity()
    }

    fn equals(&self, other: &PySeries, null_equal: bool, strict: bool) -> bool {
        if strict && (self.series.dtype() != other.series.dtype()) {
            return false;
        }
        if null_equal {
            self.series.equals_missing(&other.series)
        } else {
            self.series.equals(&other.series)
        }
    }

    fn as_str(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.series))
    }

    fn len(&self) -> usize {
        self.series.len()
    }

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    fn as_single_ptr(&mut self) -> PyResult<usize> {
        let ptr = self.series.as_single_ptr().map_err(PyPolarsErr::from)?;
        Ok(ptr)
    }

    fn clone(&self) -> Self {
        self.series.clone().into()
    }

    #[pyo3(signature = (lambda, output_type, skip_nulls))]
    fn apply_lambda(
        &self,
        lambda: &PyAny,
        output_type: Option<Wrap<DataType>>,
        skip_nulls: bool,
    ) -> PyResult<PySeries> {
        let series = &self.series;

        if skip_nulls && (series.null_count() == series.len()) {
            if let Some(output_type) = output_type {
                return Ok(Series::full_null(series.name(), series.len(), &output_type.0).into());
            }
            let msg = "The output type of the 'apply' function cannot be determined.\n\
            The function was never called because 'skip_nulls=True' and all values are null.\n\
            Consider setting 'skip_nulls=False' or setting the 'return_dtype'.";
            raise_err!(msg, ComputeError)
        }

        let output_type = output_type.map(|dt| dt.0);

        macro_rules! dispatch_apply {
            ($self:expr, $method:ident, $($args:expr),*) => {
                match $self.dtype() {
                    #[cfg(feature = "object")]
                    DataType::Object(_) => {
                        let ca = $self.0.unpack::<ObjectType<ObjectValue>>().unwrap();
                        ca.$method($($args),*)
                    },
                    _ => {
                        apply_method_all_arrow_series2!(
                            $self,
                            $method,
                            $($args),*
                        )
                    }

                }
            }

        }

        Python::with_gil(|py| {
            if matches!(
                self.series.dtype(),
                DataType::Datetime(_, _)
                    | DataType::Date
                    | DataType::Duration(_)
                    | DataType::Categorical(_, _)
                    | DataType::Binary
                    | DataType::Array(_, _)
                    | DataType::Time
            ) || !skip_nulls
            {
                let mut avs = Vec::with_capacity(self.series.len());
                let iter = self.series.iter().map(|av| match (skip_nulls, av) {
                    (true, AnyValue::Null) => AnyValue::Null,
                    (_, av) => {
                        let input = Wrap(av);
                        call_lambda_and_extract::<_, Wrap<AnyValue>>(py, lambda, input)
                            .unwrap()
                            .0
                    },
                });
                avs.extend(iter);
                return Ok(Series::new(self.name(), &avs).into());
            }

            let out = match output_type {
                Some(DataType::Int8) => {
                    let ca: Int8Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Int16) => {
                    let ca: Int16Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Int32) => {
                    let ca: Int32Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Int64) => {
                    let ca: Int64Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::UInt8) => {
                    let ca: UInt8Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::UInt16) => {
                    let ca: UInt16Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::UInt32) => {
                    let ca: UInt32Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::UInt64) => {
                    let ca: UInt64Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Float32) => {
                    let ca: Float32Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Float64) => {
                    let ca: Float64Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Boolean) => {
                    let ca: BooleanChunked = dispatch_apply!(
                        series,
                        apply_lambda_with_bool_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Utf8) => {
                    let ca = dispatch_apply!(
                        series,
                        apply_lambda_with_utf8_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;

                    ca.into_series()
                },
                #[cfg(feature = "object")]
                Some(DataType::Object(_)) => {
                    let ca = dispatch_apply!(
                        series,
                        apply_lambda_with_object_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                None => return dispatch_apply!(series, apply_lambda_unknown, py, lambda),

                _ => return dispatch_apply!(series, apply_lambda_unknown, py, lambda),
            };

            Ok(out.into())
        })
    }

    fn zip_with(&self, mask: &PySeries, other: &PySeries) -> PyResult<Self> {
        let mask = mask.series.bool().map_err(PyPolarsErr::from)?;
        let s = self
            .series
            .zip_with(mask, &other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    #[pyo3(signature = (separator, drop_first=false))]
    fn to_dummies(&self, separator: Option<&str>, drop_first: bool) -> PyResult<PyDataFrame> {
        let df = self
            .series
            .to_dummies(separator, drop_first)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    fn get_list(&self, index: usize) -> Option<Self> {
        let ca = self.series.list().ok()?;
        Some(ca.get_as_series(index)?.into())
    }

    fn n_unique(&self) -> PyResult<usize> {
        let n = self.series.n_unique().map_err(PyPolarsErr::from)?;
        Ok(n)
    }

    fn floor(&self) -> PyResult<Self> {
        let s = self.series.floor().map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    fn shrink_to_fit(&mut self) {
        self.series.shrink_to_fit();
    }

    fn dot(&self, other: &PySeries) -> Option<f64> {
        self.series.dot(&other.series)
    }

    #[cfg(feature = "ipc_streaming")]
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        // Used in pickle/pickling
        let mut buf: Vec<u8> = vec![];
        // IPC only support DataFrames so we need to convert it
        let mut df = self.series.clone().into_frame();
        IpcStreamWriter::new(&mut buf)
            .finish(&mut df)
            .expect("ipc writer");
        Ok(PyBytes::new(py, &buf).to_object(py))
    }

    #[cfg(feature = "ipc_streaming")]
    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Used in pickle/pickling
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                let c = Cursor::new(s.as_bytes());
                let reader = IpcStreamReader::new(c);
                let mut df = reader.finish().map_err(PyPolarsErr::from)?;

                df.pop()
                    .map(|s| {
                        self.series = s;
                    })
                    .ok_or_else(|| {
                        PyPolarsErr::from(PolarsError::NoData(
                            "No columns found in IPC byte stream".into(),
                        ))
                        .into()
                    })
            },
            Err(e) => Err(e),
        }
    }

    fn skew(&self, bias: bool) -> PyResult<Option<f64>> {
        let out = self.series.skew(bias).map_err(PyPolarsErr::from)?;
        Ok(out)
    }

    fn kurtosis(&self, fisher: bool, bias: bool) -> PyResult<Option<f64>> {
        let out = self
            .series
            .kurtosis(fisher, bias)
            .map_err(PyPolarsErr::from)?;
        Ok(out)
    }

    fn cast(&self, dtype: Wrap<DataType>, strict: bool) -> PyResult<Self> {
        let dtype = dtype.0;
        let out = if strict {
            self.series.strict_cast(&dtype)
        } else {
            self.series.cast(&dtype)
        };
        let out = out.map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn get_chunks(&self) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let wrap_s = py_modules::POLARS.getattr(py, "wrap_s").unwrap();
            flatten_series(&self.series)
                .into_iter()
                .map(|s| wrap_s.call1(py, (Self::new(s),)))
                .collect()
        })
    }

    fn is_sorted(&self, descending: bool) -> PyResult<bool> {
        let options = SortOptions {
            descending,
            nulls_last: descending,
            multithreaded: true,
            maintain_order: false,
        };
        Ok(self.series.is_sorted(options).map_err(PyPolarsErr::from)?)
    }

    fn clear(&self) -> Self {
        self.series.clear().into()
    }

    fn hist(&self, bins: Option<Self>, bin_count: Option<usize>) -> PyResult<PyDataFrame> {
        let bins = bins.map(|s| s.series);
        let out = hist(&self.series, bins.as_ref(), bin_count).map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn head(&self, n: usize) -> Self {
        self.series.head(Some(n)).into()
    }

    fn tail(&self, n: usize) -> Self {
        self.series.tail(Some(n)).into()
    }

    fn value_counts(&self, sort: bool, parallel: bool) -> PyResult<PyDataFrame> {
        let out = self
            .series
            .value_counts(sort, parallel)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn slice(&self, offset: i64, length: Option<usize>) -> Self {
        let length = length.unwrap_or_else(|| self.series.len());
        self.series.slice(offset, length).into()
    }
}

macro_rules! impl_set_with_mask {
    ($name:ident, $native:ty, $cast:ident, $variant:ident) => {
        fn $name(
            series: &Series,
            filter: &PySeries,
            value: Option<$native>,
        ) -> PolarsResult<Series> {
            let mask = filter.series.bool()?;
            let ca = series.$cast()?;
            let new = ca.set(mask, value)?;
            Ok(new.into_series())
        }

        #[pymethods]
        impl PySeries {
            fn $name(&self, filter: &PySeries, value: Option<$native>) -> PyResult<Self> {
                let series = $name(&self.series, filter, value).map_err(PyPolarsErr::from)?;
                Ok(Self::new(series))
            }
        }
    };
}

impl_set_with_mask!(set_with_mask_str, &str, utf8, Utf8);
impl_set_with_mask!(set_with_mask_f64, f64, f64, Float64);
impl_set_with_mask!(set_with_mask_f32, f32, f32, Float32);
impl_set_with_mask!(set_with_mask_u8, u8, u8, UInt8);
impl_set_with_mask!(set_with_mask_u16, u16, u16, UInt16);
impl_set_with_mask!(set_with_mask_u32, u32, u32, UInt32);
impl_set_with_mask!(set_with_mask_u64, u64, u64, UInt64);
impl_set_with_mask!(set_with_mask_i8, i8, i8, Int8);
impl_set_with_mask!(set_with_mask_i16, i16, i16, Int16);
impl_set_with_mask!(set_with_mask_i32, i32, i32, Int32);
impl_set_with_mask!(set_with_mask_i64, i64, i64, Int64);
impl_set_with_mask!(set_with_mask_bool, bool, bool, Boolean);

macro_rules! impl_get {
    ($name:ident, $series_variant:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            fn $name(&self, index: i64) -> Option<$type> {
                if let Ok(ca) = self.series.$series_variant() {
                    let index = if index < 0 {
                        (ca.len() as i64 + index) as usize
                    } else {
                        index as usize
                    };
                    ca.get(index)
                } else {
                    None
                }
            }
        }
    };
}

impl_get!(get_f32, f32, f32);
impl_get!(get_f64, f64, f64);
impl_get!(get_u8, u8, u8);
impl_get!(get_u16, u16, u16);
impl_get!(get_u32, u32, u32);
impl_get!(get_u64, u64, u64);
impl_get!(get_i8, i8, i8);
impl_get!(get_i16, i16, i16);
impl_get!(get_i32, i32, i32);
impl_get!(get_i64, i64, i64);
impl_get!(get_str, utf8, &str);
impl_get!(get_date, date, i32);
impl_get!(get_datetime, datetime, i64);
impl_get!(get_duration, duration, i64);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn transmute_to_series() {
        // NOTE: This is only possible because PySeries is #[repr(transparent)]
        // https://doc.rust-lang.org/reference/type-layout.html
        let ps = PySeries {
            series: [1i32, 2, 3].iter().collect(),
        };

        let s = unsafe { std::mem::transmute::<PySeries, Series>(ps.clone()) };

        assert_eq!(s.sum::<i32>(), Some(6));
        let collection = vec![ps];
        let s = collection.to_series();
        assert_eq!(
            s.iter().map(|s| s.sum::<i32>()).collect::<Vec<_>>(),
            vec![Some(6)]
        );
    }
}
