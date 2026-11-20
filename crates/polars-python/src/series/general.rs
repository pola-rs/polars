use polars_core::chunked_array::cast::CastOptions;
use polars_core::frame::column::DictionaryColumn;
use polars_core::series::IsSorted;
use polars_core::utils::flatten::flatten_series;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{IntoPyObjectExt, Python};

use super::PySeries;
use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::prelude::*;
use crate::py_modules::polars;
use crate::utils::EnterPolarsExt;

/// A zero-length result for a Scalar or Dictionary column, without materialising it. A Scalar
/// runs `on_single` on its one-value Series (that keeps the flags of the Series path, a List's
/// fast-explode among them); a Dictionary takes `dict_out`'s Series (zero rows: cheap).
fn empty_result(
    c: &Column,
    on_single: impl FnOnce(&Series) -> Series,
    dict_out: impl FnOnce() -> Column,
) -> PySeries {
    match c {
        Column::Scalar(sc) => PySeries::new(on_single(&sc.as_single_value_series())),
        _ => PySeries::new(dict_out().as_materialized_series().clone()),
    }
}

/// `value_counts` of a dictionary column from the values its rows use, with the Series path's
/// columns and checks. The order of the rows is unspecified, as it is there (a hash order), or
/// by count when `sort`. `None` for a column with a sorted flag (as the Series path).
fn dictionary_value_counts(
    d: &DictionaryColumn,
    sort: bool,
    parallel: bool,
    name: &str,
    normalize: bool,
) -> PolarsResult<Option<DataFrame>> {
    polars_ensure!(
        d.name().as_str() != name,
        Duplicate: "using `value_counts` on a column/series named '{}' would lead to duplicate \
        column names; change `name` to fix", name,
    );
    if d.is_sorted_flag() != IsSorted::Not || d.is_empty() {
        return Ok(None);
    }
    let Some((values, counts)) = d.used_values_with_counts() else {
        return Ok(None);
    };
    let height = counts.len();
    let counts = IdxCa::from_vec(name.into(), counts);
    let counts = if normalize {
        let len = d.len() as f64;
        let counts: Float64Chunked =
            polars_core::prelude::arity::unary_elementwise_values(&counts, |count| {
                count as f64 / len
            });
        counts.into_column()
    } else {
        counts.into_column()
    };
    let df = unsafe { DataFrame::new_unchecked(height, vec![values.into(), counts]) };
    if sort {
        df.sort(
            [name],
            SortMultipleOptions::default()
                .with_order_descending(true)
                .with_multithreaded(parallel),
        )
        .map(Some)
    } else {
        Ok(Some(df))
    }
}

#[pymethods]
impl PySeries {
    fn struct_unnest(&self, py: Python) -> PyResult<PyDataFrame> {
        py.enter_polars_df(|| Ok(self.series().struct_()?.clone().unnest()))
    }

    fn struct_fields(&self) -> PyResult<Vec<String>> {
        let s = self.series();
        let ca = s.struct_().map_err(PyPolarsErr::from)?;
        Ok(ca
            .struct_fields()
            .iter()
            .map(|s| s.name().to_string())
            .collect())
    }

    // From the materialised Series, as before: the flag of a `Column::Dictionary` was not
    // checked to equal it. (A `Column::Scalar` says `Ascending`, as its Series does.)
    fn is_sorted_ascending_flag(&self) -> bool {
        matches!(self.series().is_sorted_flag(), IsSorted::Ascending)
    }

    fn is_sorted_descending_flag(&self) -> bool {
        matches!(self.series().is_sorted_flag(), IsSorted::Descending)
    }

    fn can_fast_explode_flag(&self) -> bool {
        match self.series().list() {
            Err(_) => false,
            Ok(list) => list._can_fast_explode(),
        }
    }

    fn estimated_size(&self) -> usize {
        self.series().estimated_size()
    }

    #[cfg(feature = "object")]
    fn get_object<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        let s = self.series();
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

        py.enter_polars_series(|| self.series().reshape_array(&dims))
    }

    /// Returns the string format of a single element of the Series.
    fn get_fmt(&self, index: usize, str_len_limit: usize) -> String {
        let s = self.series();
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
        // The write lock must be taken with the GIL released: readers such as `to_arrow`
        // hold the read lock across calls into Python, so blocking on the lock while
        // holding the GIL deadlocks both threads.
        py.enter_polars_ok(|| {
            if in_place {
                // One write lock, so the rechunk is atomic.
                let mut column = self.column.write();
                *column = rechunk_column(&column);
                None
            } else {
                Some(Self::from_column(rechunk_column(&self.column.read())))
            }
        })
    }

    /// Get a value by index.
    fn get_index(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        // A scalar or dictionary column reads the value without materialising.
        let column = self.compressed();
        let series;
        let got = match &column {
            Some(c) => c.get(index),
            None => {
                series = self.series();
                series.get(index)
            },
        };
        let av = match got {
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
    fn get_index_signed(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
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
        py.enter_polars_series(|| &self.series() & &other.series())
    }

    fn bitor(&self, py: Python<'_>, other: &PySeries) -> PyResult<Self> {
        py.enter_polars_series(|| &self.series() | &other.series())
    }

    fn bitxor(&self, py: Python<'_>, other: &PySeries) -> PyResult<Self> {
        py.enter_polars_series(|| &self.series() ^ &other.series())
    }

    fn chunk_lengths(&self) -> Vec<usize> {
        self.series().chunk_lengths().collect()
    }

    pub fn name(&self) -> String {
        self.column.read().name().to_string()
    }

    fn rename(&self, name: &str) {
        self.column.write().rename(name.into());
    }

    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Wrap(self.column.read().dtype().clone()).into_pyobject(py)
    }

    fn set_sorted_flag(&self, descending: bool) -> Self {
        let mut out = self.series();
        if descending {
            out.set_sorted_flag(IsSorted::Descending);
        } else {
            out.set_sorted_flag(IsSorted::Ascending)
        }
        out.into()
    }

    /// Which kind of column backs this Series: "series", "scalar" or "dictionary".
    /// For tests only: there is no public Python API for it.
    fn _column_kind(&self) -> &'static str {
        self.column_kind()
    }

    /// A copy of this Series holding a `Column::Series` of the materialised column (its
    /// cache is shared), for running an expression on a Scalar-backed Series as on a plain one.
    fn _materialized(&self) -> Self {
        self.series().into()
    }

    /// Whether the column holds a materialised `Series`: always for a `Column::Series`, else
    /// when the (shared) cache is filled. For tests only.
    fn _is_materialised(&self) -> bool {
        match &*self.column.read() {
            Column::Series(_) => true,
            Column::Scalar(s) => s.lazy_as_materialized_series().is_some(),
            Column::Dictionary(d) => d.lazy_as_materialized_series().is_some(),
        }
    }

    fn n_chunks(&self) -> usize {
        self.series().n_chunks()
    }

    fn append(&self, py: Python<'_>, other: &PySeries) -> PyResult<()> {
        py.enter_polars(|| {
            // Prevent self-append deadlocks.
            // A scalar column keeps the Series path: it merges equal values into one chunk.
            if matches!(self.compressed(), Some(Column::Dictionary(_))) {
                let other = other.column.read().clone();
                return self.column.write().append(&other).map(drop);
            }
            let other = other.series();
            self.with_series_mut(|s| s.append(&other).map(drop))
        })
    }

    fn extend(&self, py: Python<'_>, other: &PySeries) -> PyResult<()> {
        py.enter_polars(|| {
            // Prevent self-extend deadlocks.
            // A scalar column keeps the Series path: `extend` clears the sorted flag of a
            // String Series, which `Column::extend` (ascending, as a scalar column always
            // is) would not.
            if matches!(self.compressed(), Some(Column::Dictionary(_))) {
                let other = other.column.read().clone();
                return self.column.write().extend(&other).map(drop);
            }
            let other = other.series();
            self.with_series_mut(|s| s.extend(&other).map(drop))
        })
    }

    fn new_from_index(&self, py: Python<'_>, index: usize, length: usize) -> PyResult<Self> {
        if index >= self.len() {
            return Err(PyValueError::new_err("index is out of bounds"));
        }
        if let Some(c) = self.compressed() {
            if length > 0 {
                return py.enter_polars_column(|| Ok(c.new_from_index(index, length)));
            }
            return Ok(empty_result(
                &c,
                |one| one.new_from_index(0, 0),
                || c.new_from_index(index, 0),
            ));
        }
        let s = self.series();
        py.enter_polars_series(|| Ok(s.new_from_index(index, length)))
    }

    fn filter(&self, py: Python<'_>, filter: &PySeries) -> PyResult<Self> {
        let filter_series = &filter.series();
        if let Ok(ca) = filter_series.bool() {
            if let Some(c) = self.compressed() {
                // A scalar column does not check the length of the mask. Any error is raised
                // by the Series path below, so that its message is the same.
                let len_ok = ca.len() == c.len() || ca.len() == 1;
                if let Some(out) =
                    py.enter_polars_ok(|| len_ok.then(|| c.filter(ca).ok()).flatten())?
                {
                    if !out.is_empty() {
                        return Ok(Self::from_column(out));
                    }
                    return Ok(empty_result(&c, |one| one.head(Some(0)), || out.clone()));
                }
            }
            py.enter_polars_series(|| self.series().filter(ca))
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
            self.series().sort(
                SortOptions::default()
                    .with_order_descending(descending)
                    .with_nulls_last(nulls_last)
                    .with_multithreaded(multithreaded),
            )
        })
    }

    fn gather_with_series(&self, py: Python<'_>, indices: &PySeries) -> PyResult<Self> {
        // A scalar column keeps the Series path: a gather clears the sorted flag of the
        // Series, but a scalar column of any length reports itself as sorted.
        if let Some(c @ Column::Dictionary(_)) = self.compressed() {
            // Errors (non-index dtype, out of bounds) are raised by the Series path below.
            let indices = indices.series();
            let out = py.enter_polars_ok(|| indices.idx().ok().and_then(|i| c.take(i).ok()))?;
            if let Some(out) = out {
                return Ok(Self::from_column(out));
            }
            return py.enter_polars_series(|| self.series().take(indices.idx()?));
        }
        py.enter_polars_series(|| self.series().take(indices.series().idx()?))
    }

    fn null_count(&self) -> PyResult<usize> {
        Ok(self.column.read().null_count())
    }

    fn has_nulls(&self) -> bool {
        self.column.read().has_nulls()
    }

    fn equals(
        &self,
        py: Python<'_>,
        other: &PySeries,
        check_dtypes: bool,
        check_names: bool,
        null_equal: bool,
    ) -> PyResult<bool> {
        let s = self.series();
        let o = other.series();
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
        Ok(format!("{:?}", self.series()))
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.column.read().len()
    }

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    fn as_single_ptr(&self, py: Python) -> PyResult<usize> {
        py.enter_polars(|| self.with_series_mut(|s| s.as_single_ptr()))
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    fn zip_with(&self, py: Python<'_>, mask: &PySeries, other: &PySeries) -> PyResult<Self> {
        let ms = mask.series();
        let mask = ms.bool().map_err(PyPolarsErr::from)?;
        py.enter_polars_series(|| self.series().zip_with(mask, &other.series()))
    }

    #[pyo3(signature = (separator, drop_first, drop_nulls))]
    fn to_dummies(
        &self,
        py: Python<'_>,
        separator: Option<&str>,
        drop_first: bool,
        drop_nulls: bool,
    ) -> PyResult<PyDataFrame> {
        py.enter_polars_df(|| self.series().to_dummies(separator, drop_first, drop_nulls))
    }

    fn get_list(&self, index: usize) -> Option<Self> {
        let s = self.series();
        let ca = s.list().ok()?;
        Some(ca.get_as_series(index)?.into())
    }

    fn n_unique(&self, py: Python) -> PyResult<usize> {
        py.enter_polars(|| match self.compressed() {
            // One row of a scalar column says it; the values a dictionary column's rows use.
            Some(c @ Column::Dictionary(_)) => c.n_unique(),
            _ => self.repeat_invariant_series(|n| n.min(1)).n_unique(),
        })
    }

    fn floor(&self, py: Python) -> PyResult<Self> {
        py.enter_polars_series(|| self.series().floor())
    }

    fn shrink_to_fit(&self, py: Python) -> PyResult<()> {
        py.enter_polars_ok(|| {
            let mut column = self.column.write();
            // `Column::shrink_to_fit` does nothing for a dictionary column, while the Series
            // rechunks (`n_chunks` shows it).
            if matches!(&*column, Column::Dictionary(_)) {
                column.into_materialized_series().shrink_to_fit();
            } else {
                column.shrink_to_fit();
            }
        })
    }

    fn dot<'py>(&self, other: &PySeries, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let s = &self.series();
        let o = &other.series();
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
            &py.enter_polars(|| self.series().serialize_to_bytes())?,
        ))
    }

    fn __setstate__(&self, py: Python<'_>, state: Py<PyAny>) -> PyResult<()> {
        // Used in pickle/pickling
        use pyo3::pybacked::PyBackedBytes;
        match state.extract::<PyBackedBytes>(py).map_err(PyErr::from) {
            Ok(bytes) => py.enter_polars(|| {
                let mut reader = std::io::Cursor::new(&*bytes);
                self.set_series(Series::deserialize_from_reader(&mut reader)?);
                PolarsResult::Ok(())
            }),
            Err(e) => Err(e),
        }
    }

    fn skew(&self, py: Python<'_>, bias: bool) -> PyResult<Option<f64>> {
        py.enter_polars(|| self.series().skew(bias))
    }

    fn kurtosis(&self, py: Python<'_>, fisher: bool, bias: bool) -> PyResult<Option<f64>> {
        py.enter_polars(|| self.series().kurtosis(fisher, bias))
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
        // Only a dictionary column cast to String (the column itself) runs on the column. Other
        // targets differ from the Series path: a Scalar column reports itself as sorted after
        // a cast, and a dictionary column's elementwise cast is one chunk and numbers a
        // Categorical by the dictionary's order.
        if let Some(c @ Column::Dictionary(_)) = self.compressed() {
            if dtype.0 == DataType::String {
                return Ok(Self::from_column(
                    c.cast_with_options(&dtype.0, options)
                        .map_err(PyPolarsErr::from)?,
                ));
            }
        }
        py.enter_polars_series(|| self.series().cast_with_options(&dtype.0, options))
    }

    fn get_chunks(&self) -> PyResult<Vec<Py<PyAny>>> {
        Python::attach(|py| {
            let wrap_s = py_modules::polars(py).getattr(py, "wrap_s").unwrap();
            flatten_series(&self.series())
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
        py.enter_polars(|| {
            // Two rows of a scalar column are sorted as all of them are.
            self.repeat_invariant_series(|n| n.min(2))
                .is_sorted(options)
        })
    }

    fn clear(&self) -> Self {
        match self.compressed() {
            Some(c) => empty_result(&c, |one| one.clear(), || c.clear()),
            None => self.series().clear().into(),
        }
    }

    fn head(&self, py: Python<'_>, n: usize) -> PyResult<Self> {
        if let Some(c) = self.compressed() {
            if n == 0 {
                return Ok(empty_result(
                    &c,
                    |one| one.head(Some(0)),
                    || c.head(Some(0)),
                ));
            }
            let out = py.enter_polars_column(|| Ok(c.head(Some(n))))?;
            if out.len() > 0 {
                return Ok(out);
            }
        }
        py.enter_polars_series(|| Ok(self.series().head(Some(n))))
    }

    fn tail(&self, py: Python<'_>, n: usize) -> PyResult<Self> {
        if let Some(c) = self.compressed() {
            if n == 0 {
                return Ok(empty_result(
                    &c,
                    |one| one.head(Some(0)),
                    || c.tail(Some(0)),
                ));
            }
            let out = py.enter_polars_column(|| Ok(c.tail(Some(n))))?;
            if out.len() > 0 {
                return Ok(out);
            }
        }
        py.enter_polars_series(|| Ok(self.series().tail(Some(n))))
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
            if let Some(Column::Dictionary(d)) = self.compressed() {
                if let Some(df) = dictionary_value_counts(&d, sort, parallel, &name, normalize)? {
                    return Ok(df);
                }
            }
            self.series()
                .value_counts(sort, parallel, name.into(), normalize)
        })
    }

    #[pyo3(signature = (offset, length))]
    fn slice(&self, offset: i64, length: Option<usize>) -> Self {
        if let Some(c) = self.compressed() {
            let c_orig = c.clone();
            let out = match Slice::new(offset, length) {
                Some(slice) => c.slice(slice),
                None => c,
            };
            // A zero-length result goes the Series path (a List scalar column loses the
            // fast-explode flag).
            if !out.is_empty() {
                return Self::from_column(out);
            }
            return empty_result(&c_orig, |one| one.head(Some(0)), || out.clone());
        }
        let s = self.series();
        match Slice::new(offset, length) {
            Some(slice) => s.slice(slice),
            None => s.clone(),
        }
        .into()
    }

    pub fn not_(&self, py: Python) -> PyResult<Self> {
        py.enter_polars_series(|| polars_ops::series::negate_bitwise(&self.series()))
    }

    pub fn shrink_dtype(&self, py: Python<'_>) -> PyResult<Self> {
        py.enter_polars(|| {
            self.series()
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
                let s = self.series();
                let datetime_strings = s.str()?;
                let ambiguous = ambiguous.into_series();
                let ambiguous = ambiguous.str()?;

                polars_core::chunked_array::temporal::string::infer::to_datetime_with_inferred_tz(
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
            let s = self.series();
            let ca = s.str()?;
            ca.to_decimal_infer(inference_length).map(Series::from)
        })
    }

    #[cfg(feature = "extract_jsonpath")]
    fn str_json_decode(
        &self,
        py: Python<'_>,
        infer_schema_length: Option<usize>,
    ) -> PyResult<Self> {
        py.enter_polars(|| {
            let lock = self.series();
            lock.str()?
                .json_decode(None, infer_schema_length)
                .map(|s| s.with_name(lock.name().clone()))
        })
        .map(Into::into)
        .map_err(PyPolarsErr::from)
        .map_err(PyErr::from)
    }

    fn ext_to(&self, dtype: Wrap<DataType>) -> PyResult<Self> {
        let DataType::Extension(typ, storage) = &dtype.0 else {
            return Err(PyTypeError::new_err(
                "ext.to(dtype) can only be used with Extension dtypes",
            ));
        };

        let s = self.series();

        if storage.as_ref() != s.dtype() {
            return Err(PyErr::from(PyPolarsErr::from(polars_err!(SchemaMismatch:
                "storage type mismatch in ext.to(): expected {}, got {}",
                storage,
                s.dtype()
            ))));
        }

        Ok(s.clone().into_extension(typ.clone()).into())
    }

    fn ext_storage(&self) -> Self {
        self.series().to_storage().clone().into()
    }

    fn set(&self, py: Python<'_>, mask: PySeries, value: PySeries) -> PyResult<Self> {
        assert_eq!(value.len(), 1);
        py.enter_polars(|| {
            let slf = self.series();
            let mask = mask.series();
            let value = value.series();

            let mask = mask.bool()?;

            PolarsResult::Ok(
                value
                    .zip_with_same_type(mask, &slf)?
                    .with_name(slf.name().clone()),
            )
        })
        .map(Into::into)
        .map_err(PyPolarsErr::from)
        .map_err(PyErr::from)
    }
}

macro_rules! impl_get {
    ($name:ident, $series_variant:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            fn $name(&self, index: i64) -> Option<$type> {
                let s = self.series();
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
                let s = self.series();
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

/// Rechunk a column. A `Column::Series` stays one (`Column::rechunk` would turn a length-1
/// result into a `Column::Scalar`); other variants use their own rechunk.
fn rechunk_column(column: &Column) -> Column {
    match column {
        Column::Series(s) => Column::Series(s.rechunk().into()),
        c => c.rechunk(),
    }
}
