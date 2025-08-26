use polars_core::chunked_array::cast::CastOptions;
use polars_core::series::IsSorted;
use polars_core::utils::flatten::flatten_series;
use polars_utils::python_function::PythonObject;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{IntoPyObjectExt, Python};

use super::PySeries;
use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::prelude::*;
use crate::py_modules::polars;
use crate::utils::EnterPolarsExt;

#[pymethods]
impl PySeries {
    fn struct_unnest(&self, py: Python) -> PyResult<PyDataFrame> {
        py.enter_polars_df(|| Ok(self.series.read().struct_()?.clone().unnest()))
    }

    fn struct_fields(&self) -> PyResult<Vec<String>> {
        let s = self.series.read();
        let ca = s.struct_().map_err(PyPolarsErr::from)?;
        Ok(ca
            .struct_fields()
            .iter()
            .map(|s| s.name().to_string())
            .collect())
    }

    fn is_sorted_ascending_flag(&self) -> bool {
        matches!(self.series.read().is_sorted_flag(), IsSorted::Ascending)
    }

    fn is_sorted_descending_flag(&self) -> bool {
        matches!(self.series.read().is_sorted_flag(), IsSorted::Descending)
    }

    fn can_fast_explode_flag(&self) -> bool {
        match self.series.read().list() {
            Err(_) => false,
            Ok(list) => list._can_fast_explode(),
        }
    }

    pub fn cat_uses_lexical_ordering(&self) -> PyResult<bool> {
        Ok(true)
    }

    pub fn cat_is_local(&self) -> PyResult<bool> {
        Ok(false)
    }

    pub fn cat_to_local(&self, _py: Python) -> PyResult<Self> {
        Ok(self.clone())
    }

    fn estimated_size(&self) -> usize {
        self.series.read().estimated_size()
    }

    #[cfg(feature = "object")]
    fn get_object<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        let s = self.series.read();
        if matches!(s.dtype(), DataType::Object(_)) {
            let obj: Option<&ObjectValue> = s.get_object(index).map(|any| any.into());
            Ok(obj.into_pyobject(py)?)
        } else {
            Ok(py.None().into_bound(py))
        }
    }

    #[cfg(feature = "dtype-array")]
    fn reshape(&self, py: Python<'_>, dims: Vec<i64>) -> PyResult<Self> {
        let dims = dims
            .into_iter()
            .map(ReshapeDimension::new)
            .collect::<Vec<_>>();

        py.enter_polars_series(|| self.series.read().reshape_array(&dims))
    }

    /// Returns the string format of a single element of the Series.
    fn get_fmt(&self, index: usize, str_len_limit: usize) -> String {
        let s = self.series.read();
        let v = format!("{}", s.get(index).unwrap());
        if let DataType::String | DataType::Categorical(_, _) | DataType::Enum(_, _) = s.dtype() {
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

    pub fn rechunk(&self, py: Python<'_>, in_place: bool) -> PyResult<Option<Self>> {
        let series = py.enter_polars_ok(|| self.series.read().rechunk())?;
        if in_place {
            *self.series.write() = series;
            Ok(None)
        } else {
            Ok(Some(series.into()))
        }
    }

    /// Get a value by index.
    fn get_index(&self, py: Python<'_>, index: usize) -> PyResult<PyObject> {
        let s = self.series.read();
        let av = match s.get(index) {
            Ok(v) => v,
            Err(PolarsError::OutOfBounds(err)) => {
                return Err(PyIndexError::new_err(err.to_string()));
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
    fn get_index_signed(&self, py: Python<'_>, index: isize) -> PyResult<PyObject> {
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

    fn bitand(&self, py: Python<'_>, other: &PySeries) -> PyResult<Self> {
        py.enter_polars_series(|| &*self.series.read() & &*other.series.read())
    }

    fn bitor(&self, py: Python<'_>, other: &PySeries) -> PyResult<Self> {
        py.enter_polars_series(|| &*self.series.read() | &*other.series.read())
    }

    fn bitxor(&self, py: Python<'_>, other: &PySeries) -> PyResult<Self> {
        py.enter_polars_series(|| &*self.series.read() ^ &*other.series.read())
    }

    fn chunk_lengths(&self) -> Vec<usize> {
        self.series.read().chunk_lengths().collect()
    }

    pub fn name(&self) -> String {
        self.series.read().name().to_string()
    }

    fn rename(&self, name: &str) {
        self.series.write().rename(name.into());
    }

    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Wrap(self.series.read().dtype().clone()).into_pyobject(py)
    }

    fn set_sorted_flag(&self, descending: bool) -> Self {
        let mut out = self.series.read().clone();
        if descending {
            out.set_sorted_flag(IsSorted::Descending);
        } else {
            out.set_sorted_flag(IsSorted::Ascending)
        }
        out.into()
    }

    fn n_chunks(&self) -> usize {
        self.series.read().n_chunks()
    }

    fn append(&self, py: Python<'_>, other: &PySeries) -> PyResult<()> {
        py.enter_polars(|| {
            // Prevent self-append deadlocks.
            let other = other.series.read().clone();
            let mut s = self.series.write();
            s.append(&other)?;
            PolarsResult::Ok(())
        })
    }

    fn extend(&self, py: Python<'_>, other: &PySeries) -> PyResult<()> {
        py.enter_polars(|| {
            // Prevent self-extend deadlocks.
            let other = other.series.read().clone();
            let mut s = self.series.write();
            s.extend(&other)?;
            PolarsResult::Ok(())
        })
    }

    fn new_from_index(&self, py: Python<'_>, index: usize, length: usize) -> PyResult<Self> {
        let s = self.series.read();
        if index >= s.len() {
            Err(PyValueError::new_err("index is out of bounds"))
        } else {
            py.enter_polars_series(|| Ok(s.new_from_index(index, length)))
        }
    }

    fn filter(&self, py: Python<'_>, filter: &PySeries) -> PyResult<Self> {
        let filter_series = &filter.series.read();
        if let Ok(ca) = filter_series.bool() {
            py.enter_polars_series(|| self.series.read().filter(ca))
        } else {
            Err(PyRuntimeError::new_err("Expected a boolean mask"))
        }
    }

    fn sort(
        &self,
        py: Python<'_>,
        descending: bool,
        nulls_last: bool,
        multithreaded: bool,
    ) -> PyResult<Self> {
        py.enter_polars_series(|| {
            self.series.read().sort(
                SortOptions::default()
                    .with_order_descending(descending)
                    .with_nulls_last(nulls_last)
                    .with_multithreaded(multithreaded),
            )
        })
    }

    fn gather_with_series(&self, py: Python<'_>, indices: &PySeries) -> PyResult<Self> {
        py.enter_polars_series(|| self.series.read().take(indices.series.read().idx()?))
    }

    fn null_count(&self) -> PyResult<usize> {
        Ok(self.series.read().null_count())
    }

    fn has_nulls(&self) -> bool {
        self.series.read().has_nulls()
    }

    fn equals(
        &self,
        py: Python<'_>,
        other: &PySeries,
        check_dtypes: bool,
        check_names: bool,
        null_equal: bool,
    ) -> PyResult<bool> {
        let s = self.series.read();
        let o = other.series.read();
        if check_dtypes && (s.dtype() != o.dtype()) {
            return Ok(false);
        }
        if check_names && (s.name() != o.name()) {
            return Ok(false);
        }
        if null_equal {
            py.enter_polars_ok(|| s.equals_missing(&o))
        } else {
            py.enter_polars_ok(|| s.equals(&o))
        }
    }

    fn as_str(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.series.read()))
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.series.read().len()
    }

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    fn as_single_ptr(&self, py: Python) -> PyResult<usize> {
        py.enter_polars(|| self.series.write().as_single_ptr())
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    fn zip_with(&self, py: Python<'_>, mask: &PySeries, other: &PySeries) -> PyResult<Self> {
        let ms = mask.series.read();
        let mask = ms.bool().map_err(PyPolarsErr::from)?;
        py.enter_polars_series(|| self.series.read().zip_with(mask, &other.series.read()))
    }

    #[pyo3(signature = (separator, drop_first, drop_nulls))]
    fn to_dummies(
        &self,
        py: Python<'_>,
        separator: Option<&str>,
        drop_first: bool,
        drop_nulls: bool,
    ) -> PyResult<PyDataFrame> {
        py.enter_polars_df(|| {
            self.series
                .read()
                .to_dummies(separator, drop_first, drop_nulls)
        })
    }

    fn get_list(&self, index: usize) -> Option<Self> {
        let s = self.series.read();
        let ca = s.list().ok()?;
        Some(ca.get_as_series(index)?.into())
    }

    fn n_unique(&self, py: Python) -> PyResult<usize> {
        py.enter_polars(|| self.series.read().n_unique())
    }

    fn floor(&self, py: Python) -> PyResult<Self> {
        py.enter_polars_series(|| self.series.read().floor())
    }

    fn shrink_to_fit(&self, py: Python) -> PyResult<()> {
        py.enter_polars_ok(|| self.series.write().shrink_to_fit())
    }

    fn dot<'py>(&self, other: &PySeries, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let s = &*self.series.read();
        let o = &*other.series.read();
        let lhs_dtype = s.dtype();
        let rhs_dtype = o.dtype();

        if !lhs_dtype.is_primitive_numeric() {
            return Err(PyPolarsErr::from(polars_err!(opq = dot, lhs_dtype)).into());
        };
        if !rhs_dtype.is_primitive_numeric() {
            return Err(PyPolarsErr::from(polars_err!(opq = dot, rhs_dtype)).into());
        }

        let result: AnyValue = if lhs_dtype.is_float() || rhs_dtype.is_float() {
            py.enter_polars(|| (s * o)?.sum::<f64>())?.into()
        } else {
            py.enter_polars(|| (s * o)?.sum::<i64>())?.into()
        };

        Wrap(result).into_pyobject(py)
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // Used in pickle/pickling
        Ok(PyBytes::new(
            py,
            &py.enter_polars(|| self.series.read().serialize_to_bytes())?,
        ))
    }

    fn __setstate__(&self, py: Python<'_>, state: PyObject) -> PyResult<()> {
        // Used in pickle/pickling
        use pyo3::pybacked::PyBackedBytes;
        match state.extract::<PyBackedBytes>(py) {
            Ok(bytes) => py.enter_polars(|| {
                *self.series.write() = Series::deserialize_from_reader(&mut &*bytes)?;
                PolarsResult::Ok(())
            }),
            Err(e) => Err(e),
        }
    }

    fn skew(&self, py: Python<'_>, bias: bool) -> PyResult<Option<f64>> {
        py.enter_polars(|| self.series.read().skew(bias))
    }

    fn kurtosis(&self, py: Python<'_>, fisher: bool, bias: bool) -> PyResult<Option<f64>> {
        py.enter_polars(|| self.series.read().kurtosis(fisher, bias))
    }

    fn cast(
        &self,
        py: Python<'_>,
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
        py.enter_polars_series(|| self.series.read().cast_with_options(&dtype.0, options))
    }

    fn get_chunks(&self) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let wrap_s = py_modules::polars(py).getattr(py, "wrap_s").unwrap();
            flatten_series(&self.series.read())
                .into_iter()
                .map(|s| wrap_s.call1(py, (Self::new(s),)))
                .collect()
        })
    }

    fn is_sorted(&self, py: Python<'_>, descending: bool, nulls_last: bool) -> PyResult<bool> {
        let options = SortOptions {
            descending,
            nulls_last,
            multithreaded: true,
            maintain_order: false,
            limit: None,
        };
        py.enter_polars(|| self.series.read().is_sorted(options))
    }

    fn clear(&self) -> Self {
        self.series.read().clear().into()
    }

    fn head(&self, py: Python<'_>, n: usize) -> PyResult<Self> {
        py.enter_polars_series(|| Ok(self.series.read().head(Some(n))))
    }

    fn tail(&self, py: Python<'_>, n: usize) -> PyResult<Self> {
        py.enter_polars_series(|| Ok(self.series.read().tail(Some(n))))
    }

    fn value_counts(
        &self,
        py: Python<'_>,
        sort: bool,
        parallel: bool,
        name: String,
        normalize: bool,
    ) -> PyResult<PyDataFrame> {
        py.enter_polars_df(|| {
            self.series
                .read()
                .value_counts(sort, parallel, name.into(), normalize)
        })
    }

    #[pyo3(signature = (offset, length))]
    fn slice(&self, offset: i64, length: Option<usize>) -> Self {
        let s = self.series.read();
        let length = length.unwrap_or_else(|| s.len());
        s.slice(offset, length).into()
    }

    pub fn not_(&self, py: Python) -> PyResult<Self> {
        py.enter_polars_series(|| polars_ops::series::negate_bitwise(&self.series.read()))
    }

    pub fn shrink_dtype(&self, py: Python<'_>) -> PyResult<Self> {
        py.enter_polars(|| {
            self.series
                .read()
                .shrink_type()
                .map(Into::into)
                .map_err(PyPolarsErr::from)
                .map_err(PyErr::from)
        })
    }

    fn str_to_datetime_infer(
        &self,
        py: Python,
        time_unit: Option<Wrap<TimeUnit>>,
        strict: bool,
        exact: bool,
        ambiguous: PySeries,
    ) -> PyResult<Self> {
        Ok(py
            .enter_polars(|| {
                let s = self.series.read();
                let datetime_strings = s.str()?;
                let ambiguous = ambiguous.series.into_inner();
                let ambiguous = ambiguous.str()?;

                polars_time::prelude::string::infer::to_datetime_with_inferred_tz(
                    datetime_strings,
                    time_unit.map_or(TimeUnit::Microseconds, |v| v.0),
                    strict,
                    exact,
                    ambiguous,
                )
            })?
            .into_series()
            .into())
    }

    pub fn str_to_decimal_infer(&self, py: Python, inference_length: usize) -> PyResult<Self> {
        py.enter_polars_series(|| {
            let s = self.series.read();
            let ca = s.str()?;
            ca.to_decimal_infer(inference_length).map(Series::from)
        })
    }

    pub fn list_to_struct(
        &self,
        py: Python<'_>,
        width_strat: Wrap<ListToStructWidthStrategy>,
        name_gen: Option<PyObject>,
    ) -> PyResult<Self> {
        py.enter_polars(|| {
            let get_index_name =
                name_gen.map(|f| PlanCallback::<usize, String>::new_python(PythonObject(f)));
            let get_index_name = get_index_name.map(|f| {
                NameGenerator(Arc::new(move |i| f.call(i).map(PlSmallStr::from)) as Arc<_>)
            });
            self.series
                .read()
                .list()?
                .to_struct(&ListToStructArgs::InferWidth {
                    infer_field_strategy: width_strat.0,
                    get_index_name,
                    max_fields: None,
                })
                .map(IntoSeries::into_series)
        })
        .map(Into::into)
        .map_err(PyPolarsErr::from)
        .map_err(PyErr::from)
    }

    #[cfg(feature = "extract_jsonpath")]
    fn str_json_decode(
        &self,
        py: Python<'_>,
        infer_schema_length: Option<usize>,
    ) -> PyResult<Self> {
        py.enter_polars(|| {
            let lock = self.series.read();
            lock.str()?
                .json_decode(None, infer_schema_length)
                .map(|s| s.with_name(lock.name().clone()))
        })
        .map(Into::into)
        .map_err(PyPolarsErr::from)
        .map_err(PyErr::from)
    }
}

macro_rules! impl_set_with_mask {
    ($name:ident, $native:ty, $cast:ident, $variant:ident) => {
        fn $name(
            series: &Series,
            filter: &PySeries,
            value: Option<$native>,
        ) -> PolarsResult<Series> {
            let fs = filter.series.read();
            let mask = fs.bool()?;
            let ca = series.$cast()?;
            let new = ca.set(mask, value)?;
            Ok(new.into_series())
        }

        #[pymethods]
        impl PySeries {
            #[pyo3(signature = (filter, value))]
            fn $name(
                &self,
                py: Python<'_>,
                filter: &PySeries,
                value: Option<$native>,
            ) -> PyResult<Self> {
                py.enter_polars_series(|| $name(&self.series.read(), filter, value))
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
                let s = self.series.read();
                if let Ok(ca) = s.$series_variant() {
                    let index = if index < 0 {
                        (ca.len() as i64 + index) as usize
                    } else {
                        index as usize
                    };
                    ca.get(index).map(|r| r.to_owned())
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
impl_get!(get_str, str, String);

macro_rules! impl_get_phys {
    ($name:ident, $series_variant:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            fn $name(&self, index: i64) -> Option<$type> {
                let s = self.series.read();
                if let Ok(ca) = s.$series_variant() {
                    let index = if index < 0 {
                        (ca.len() as i64 + index) as usize
                    } else {
                        index as usize
                    };
                    ca.physical().get(index)
                } else {
                    None
                }
            }
        }
    };
}

impl_get_phys!(get_date, date, i32);
impl_get_phys!(get_datetime, datetime, i64);
impl_get_phys!(get_duration, duration, i64);
