use polars_core::chunked_array::cast::CastOptions;
use polars_core::series::IsSorted;
use polars_core::utils::flatten::flatten_series;
use polars_row::RowEncodingOptions;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{IntoPyObjectExt, Python};

use self::row_encode::get_row_encoding_context;
use super::PySeries;
use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::prelude::*;
use crate::py_modules::polars;

#[pymethods]
impl PySeries {
    fn struct_unnest(&self, py: Python) -> PyResult<PyDataFrame> {
        let ca = self.series.struct_().map_err(PyPolarsErr::from)?;
        let df: DataFrame = py.allow_threads(|| ca.clone().unnest());
        Ok(df.into())
    }

    fn struct_fields(&self) -> PyResult<Vec<&str>> {
        let ca = self.series.struct_().map_err(PyPolarsErr::from)?;
        Ok(ca
            .struct_fields()
            .iter()
            .map(|s| s.name().as_str())
            .collect())
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

    pub fn cat_to_local(&self, py: Python) -> PyResult<Self> {
        let ca = self.series.categorical().map_err(PyPolarsErr::from)?;
        Ok(py.allow_threads(|| ca.to_local().into_series().into()))
    }

    fn estimated_size(&self) -> usize {
        self.series.estimated_size()
    }

    #[cfg(feature = "object")]
    fn get_object<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        if matches!(self.series.dtype(), DataType::Object(_, _)) {
            let obj: Option<&ObjectValue> = self.series.get_object(index).map(|any| any.into());
            Ok(obj.into_pyobject(py)?)
        } else {
            Ok(py.None().into_bound(py))
        }
    }

    #[cfg(feature = "dtype-array")]
    fn reshape(&self, py: Python, dims: Vec<i64>) -> PyResult<Self> {
        let dims = dims
            .into_iter()
            .map(ReshapeDimension::new)
            .collect::<Vec<_>>();

        let out = py
            .allow_threads(|| self.series.reshape_array(&dims))
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    /// Returns the string format of a single element of the Series.
    fn get_fmt(&self, index: usize, str_len_limit: usize) -> String {
        let v = format!("{}", self.series.get(index).unwrap());
        if let DataType::String | DataType::Categorical(_, _) | DataType::Enum(_, _) =
            self.series.dtype()
        {
            let v_no_quotes = &v[1..v.len() - 1];
            let v_trunc = &v_no_quotes[..v_no_quotes
                .char_indices()
                .take(str_len_limit)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0)];
            if v_no_quotes == v_trunc {
                v
            } else {
                format!("\"{v_trunc}…")
            }
        } else {
            v
        }
    }

    pub fn rechunk(&mut self, py: Python, in_place: bool) -> Option<Self> {
        let series = py.allow_threads(|| self.series.rechunk());
        if in_place {
            self.series = series;
            None
        } else {
            Some(series.into())
        }
    }

    /// Get a value by index.
    fn get_index(&self, py: Python, index: usize) -> PyResult<PyObject> {
        let av = match self.series.get(index) {
            Ok(v) => v,
            Err(PolarsError::OutOfBounds(err)) => {
                return Err(PyIndexError::new_err(err.to_string()))
            },
            Err(e) => return Err(PyPolarsErr::from(e).into()),
        };

        match av {
            AnyValue::List(s) | AnyValue::Array(s, _) => {
                let pyseries = PySeries::new(s);
                polars(py).getattr(py, "wrap_s")?.call1(py, (pyseries,))
            },
            _ => Wrap(av).into_py_any(py),
        }
    }

    /// Get a value by index, allowing negative indices.
    fn get_index_signed(&self, py: Python, index: isize) -> PyResult<PyObject> {
        let index = if index < 0 {
            match self.len().checked_sub(index.unsigned_abs()) {
                Some(v) => v,
                None => {
                    return Err(PyIndexError::new_err(
                        polars_err!(oob = index, self.len()).to_string(),
                    ));
                },
            }
        } else {
            usize::try_from(index).unwrap()
        };
        self.get_index(py, index)
    }

    fn bitand(&self, py: Python, other: &PySeries) -> PyResult<Self> {
        let out = py
            .allow_threads(|| &self.series & &other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn bitor(&self, py: Python, other: &PySeries) -> PyResult<Self> {
        let out = py
            .allow_threads(|| &self.series | &other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }
    fn bitxor(&self, py: Python, other: &PySeries) -> PyResult<Self> {
        let out = py
            .allow_threads(|| &self.series ^ &other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn chunk_lengths(&self) -> Vec<usize> {
        self.series.chunk_lengths().collect()
    }

    pub fn name(&self) -> &str {
        self.series.name().as_str()
    }

    fn rename(&mut self, name: &str) {
        self.series.rename(name.into());
    }

    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Wrap(self.series.dtype().clone()).into_pyobject(py)
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

    fn extend(&mut self, py: Python, other: &PySeries) -> PyResult<()> {
        py.allow_threads(|| self.series.extend(&other.series))
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    fn new_from_index(&self, py: Python, index: usize, length: usize) -> PyResult<Self> {
        if index >= self.series.len() {
            Err(PyValueError::new_err("index is out of bounds"))
        } else {
            Ok(py.allow_threads(|| self.series.new_from_index(index, length).into()))
        }
    }

    fn filter(&self, py: Python, filter: &PySeries) -> PyResult<Self> {
        let filter_series = &filter.series;
        if let Ok(ca) = filter_series.bool() {
            let series = py
                .allow_threads(|| self.series.filter(ca))
                .map_err(PyPolarsErr::from)?;
            Ok(PySeries { series })
        } else {
            Err(PyRuntimeError::new_err("Expected a boolean mask"))
        }
    }

    fn sort(
        &mut self,
        py: Python,
        descending: bool,
        nulls_last: bool,
        multithreaded: bool,
    ) -> PyResult<Self> {
        Ok(py
            .allow_threads(|| {
                self.series.sort(
                    SortOptions::default()
                        .with_order_descending(descending)
                        .with_nulls_last(nulls_last)
                        .with_multithreaded(multithreaded),
                )
            })
            .map_err(PyPolarsErr::from)?
            .into())
    }

    fn gather_with_series(&self, py: Python, indices: &PySeries) -> PyResult<Self> {
        py.allow_threads(|| {
            let indices = indices.series.idx().map_err(PyPolarsErr::from)?;
            let s = self.series.take(indices).map_err(PyPolarsErr::from)?;
            Ok(s.into())
        })
    }

    fn null_count(&self) -> PyResult<usize> {
        Ok(self.series.null_count())
    }

    fn has_nulls(&self) -> bool {
        self.series.has_nulls()
    }

    fn equals(
        &self,
        py: Python,
        other: &PySeries,
        check_dtypes: bool,
        check_names: bool,
        null_equal: bool,
    ) -> bool {
        if check_dtypes && (self.series.dtype() != other.series.dtype()) {
            return false;
        }
        if check_names && (self.series.name() != other.series.name()) {
            return false;
        }
        if null_equal {
            py.allow_threads(|| self.series.equals_missing(&other.series))
        } else {
            py.allow_threads(|| self.series.equals(&other.series))
        }
    }

    fn as_str(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.series))
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.series.len()
    }

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    fn as_single_ptr(&mut self, py: Python) -> PyResult<usize> {
        let ptr = py
            .allow_threads(|| self.series.as_single_ptr())
            .map_err(PyPolarsErr::from)?;
        Ok(ptr)
    }

    fn clone(&self) -> Self {
        self.series.clone().into()
    }

    fn zip_with(&self, py: Python, mask: &PySeries, other: &PySeries) -> PyResult<Self> {
        let mask = mask.series.bool().map_err(PyPolarsErr::from)?;
        let s = py
            .allow_threads(|| self.series.zip_with(mask, &other.series))
            .map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    #[pyo3(signature = (separator, drop_first=false))]
    fn to_dummies(
        &self,
        py: Python,
        separator: Option<&str>,
        drop_first: bool,
    ) -> PyResult<PyDataFrame> {
        let df = py
            .allow_threads(|| self.series.to_dummies(separator, drop_first))
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    fn get_list(&self, index: usize) -> Option<Self> {
        let ca = self.series.list().ok()?;
        Some(ca.get_as_series(index)?.into())
    }

    fn n_unique(&self, py: Python) -> PyResult<usize> {
        let n = py
            .allow_threads(|| self.series.n_unique())
            .map_err(PyPolarsErr::from)?;
        Ok(n)
    }

    fn floor(&self, py: Python) -> PyResult<Self> {
        let s = py
            .allow_threads(|| self.series.floor())
            .map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    fn shrink_to_fit(&mut self, py: Python) {
        py.allow_threads(|| self.series.shrink_to_fit());
    }

    fn dot<'py>(&self, other: &PySeries, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let lhs_dtype = self.series.dtype();
        let rhs_dtype = other.series.dtype();

        if !lhs_dtype.is_primitive_numeric() {
            return Err(PyPolarsErr::from(polars_err!(opq = dot, lhs_dtype)).into());
        };
        if !rhs_dtype.is_primitive_numeric() {
            return Err(PyPolarsErr::from(polars_err!(opq = dot, rhs_dtype)).into());
        }

        let result: AnyValue = if lhs_dtype.is_float() || rhs_dtype.is_float() {
            py.allow_threads(|| (&self.series * &other.series)?.sum::<f64>())
                .map_err(PyPolarsErr::from)?
                .into()
        } else {
            py.allow_threads(|| (&self.series * &other.series)?.sum::<i64>())
                .map_err(PyPolarsErr::from)?
                .into()
        };

        Wrap(result).into_pyobject(py)
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // Used in pickle/pickling
        Ok(PyBytes::new(
            py,
            &py.allow_threads(|| self.series.serialize_to_bytes().map_err(PyPolarsErr::from))?,
        ))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Used in pickle/pickling

        use pyo3::pybacked::PyBackedBytes;
        match state.extract::<PyBackedBytes>(py) {
            Ok(s) => py.allow_threads(|| {
                let s = Series::deserialize_from_reader(&mut &*s).map_err(PyPolarsErr::from)?;
                self.series = s;
                Ok(())
            }),
            Err(e) => Err(e),
        }
    }

    fn skew(&self, py: Python, bias: bool) -> PyResult<Option<f64>> {
        let out = py
            .allow_threads(|| self.series.skew(bias))
            .map_err(PyPolarsErr::from)?;
        Ok(out)
    }

    fn kurtosis(&self, py: Python, fisher: bool, bias: bool) -> PyResult<Option<f64>> {
        let out = py
            .allow_threads(|| self.series.kurtosis(fisher, bias))
            .map_err(PyPolarsErr::from)?;
        Ok(out)
    }

    fn cast(
        &self,
        py: Python,
        dtype: Wrap<DataType>,
        strict: bool,
        wrap_numerical: bool,
    ) -> PyResult<Self> {
        let options = if wrap_numerical {
            CastOptions::Overflowing
        } else if strict {
            CastOptions::Strict
        } else {
            CastOptions::NonStrict
        };

        let dtype = dtype.0;
        let out = py.allow_threads(|| self.series.cast_with_options(&dtype, options));
        let out = out.map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn get_chunks(&self) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let wrap_s = py_modules::polars(py).getattr(py, "wrap_s").unwrap();
            flatten_series(&self.series)
                .into_iter()
                .map(|s| wrap_s.call1(py, (Self::new(s),)))
                .collect()
        })
    }

    fn is_sorted(&self, py: Python, descending: bool, nulls_last: bool) -> PyResult<bool> {
        let options = SortOptions {
            descending,
            nulls_last,
            multithreaded: true,
            maintain_order: false,
            limit: None,
        };
        Ok(py
            .allow_threads(|| self.series.is_sorted(options))
            .map_err(PyPolarsErr::from)?)
    }

    fn clear(&self) -> Self {
        self.series.clear().into()
    }

    fn head(&self, py: Python, n: usize) -> Self {
        py.allow_threads(|| self.series.head(Some(n))).into()
    }

    fn tail(&self, py: Python, n: usize) -> Self {
        py.allow_threads(|| self.series.tail(Some(n))).into()
    }

    fn value_counts(
        &self,
        py: Python,
        sort: bool,
        parallel: bool,
        name: String,
        normalize: bool,
    ) -> PyResult<PyDataFrame> {
        let out = py
            .allow_threads(|| {
                self.series
                    .value_counts(sort, parallel, name.into(), normalize)
            })
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    #[pyo3(signature = (offset, length=None))]
    fn slice(&self, offset: i64, length: Option<usize>) -> Self {
        let length = length.unwrap_or_else(|| self.series.len());
        self.series.slice(offset, length).into()
    }

    pub fn not_(&self, py: Python) -> PyResult<Self> {
        let out = py
            .allow_threads(|| polars_ops::series::negate_bitwise(&self.series))
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    /// Internal utility function to allow direct access to the row encoding from python.
    #[pyo3(signature = (dtypes, opts))]
    fn _row_decode<'py>(
        &'py self,
        py: Python<'py>,
        dtypes: Vec<(String, Wrap<DataType>)>,
        opts: Vec<(bool, bool, bool)>,
    ) -> PyResult<PyDataFrame> {
        py.allow_threads(|| {
            assert_eq!(dtypes.len(), opts.len());

            let opts = opts
                .into_iter()
                .map(|(descending, nulls_last, no_order)| {
                    let mut opt = RowEncodingOptions::default();

                    opt.set(RowEncodingOptions::DESCENDING, descending);
                    opt.set(RowEncodingOptions::NULLS_LAST, nulls_last);
                    opt.set(RowEncodingOptions::NO_ORDER, no_order);

                    opt
                })
                .collect::<Vec<_>>();

            // The polars-row crate expects the physical arrow types.
            let arrow_dtypes = dtypes
                .iter()
                .map(|(_, dtype)| dtype.0.to_physical().to_arrow(CompatLevel::newest()))
                .collect::<Vec<_>>();

            let dicts = dtypes
                .iter()
                .map(|(_, dtype)| get_row_encoding_context(&dtype.0))
                .collect::<Vec<_>>();

            // Get the BinaryOffset array.
            let arr = self.series.rechunk();
            let arr = arr.binary_offset().map_err(PyPolarsErr::from)?;
            assert_eq!(arr.chunks().len(), 1);
            let mut values = arr
                .downcast_iter()
                .next()
                .unwrap()
                .values_iter()
                .collect::<Vec<&[u8]>>();

            let columns = PyResult::Ok(unsafe {
                polars_row::decode::decode_rows(&mut values, &opts, &dicts, &arrow_dtypes)
            })?;

            // Construct a DataFrame from the result.
            let columns = columns
                .into_iter()
                .zip(dtypes)
                .map(|(arr, (name, dtype))| unsafe {
                    Series::from_chunks_and_dtype_unchecked(
                        PlSmallStr::from(name),
                        vec![arr],
                        &dtype.0.to_physical(),
                    )
                    .into_column()
                    .from_physical_unchecked(&dtype.0)
                })
                .collect::<PolarsResult<Vec<_>>>()
                .map_err(PyPolarsErr::from)?;
            Ok(DataFrame::new(columns).map_err(PyPolarsErr::from)?.into())
        })
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
            #[pyo3(signature = (filter, value))]
            fn $name(
                &self,
                py: Python,
                filter: &PySeries,
                value: Option<$native>,
            ) -> PyResult<Self> {
                let series = py
                    .allow_threads(|| $name(&self.series, filter, value))
                    .map_err(PyPolarsErr::from)?;
                Ok(Self::new(series))
            }
        }
    };
}

impl_set_with_mask!(set_with_mask_str, &str, str, String);
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
impl_get!(get_str, str, &str);
impl_get!(get_date, date, i32);
impl_get!(get_datetime, datetime, i64);
impl_get!(get_duration, duration, i64);

#[cfg(test)]
mod test {
    use super::*;
    use crate::series::ToSeries;

    #[test]
    fn transmute_to_series() {
        // NOTE: This is only possible because PySeries is #[repr(transparent)]
        // https://doc.rust-lang.org/reference/type-layout.html
        let ps = PySeries {
            series: [1i32, 2, 3].iter().collect(),
        };

        let s = unsafe { std::mem::transmute::<PySeries, Series>(ps.clone()) };

        assert_eq!(s.sum::<i32>().unwrap(), 6);
        let collection = vec![ps];
        let s = collection.to_series();
        assert_eq!(
            s.iter()
                .map(|s| s.sum::<i32>().unwrap())
                .collect::<Vec<_>>(),
            vec![6]
        );
    }
}
