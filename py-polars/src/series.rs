use numpy::PyArray1;
use polars_core::prelude::QuantileInterpolOptions;
use polars_core::series::IsSorted;
use polars_core::utils::{flatten_series, CustomIterTools};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use pyo3::Python;

use crate::apply::series::{call_lambda_and_extract, ApplyLambda};
use crate::arrow_interop::to_rust::array_to_rust;
use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::list_construction::py_seq_to_list;
use crate::prelude::*;
use crate::set::set_at_idx;
use crate::{apply_method_all_arrow_series2, arrow_interop};

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PySeries {
    pub series: Series,
}

impl PySeries {
    pub(crate) fn new(series: Series) -> Self {
        PySeries { series }
    }
}

// Init with numpy arrays
macro_rules! init_method {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            #[staticmethod]
            pub fn $name(
                py: Python,
                name: &str,
                array: &PyArray1<$type>,
                _strict: bool,
            ) -> PySeries {
                let array = array.readonly();
                let vals = array.as_slice().unwrap();
                py.allow_threads(|| PySeries {
                    series: Series::new(name, vals),
                })
            }
        }
    };
}

init_method!(new_i8, i8);
init_method!(new_i16, i16);
init_method!(new_i32, i32);
init_method!(new_i64, i64);
init_method!(new_bool, bool);
init_method!(new_u8, u8);
init_method!(new_u16, u16);
init_method!(new_u32, u32);
init_method!(new_u64, u64);

#[pymethods]
impl PySeries {
    #[staticmethod]
    pub fn new_f32(py: Python, name: &str, array: &PyArray1<f32>, nan_is_null: bool) -> PySeries {
        let array = array.readonly();
        let vals = array.as_slice().unwrap();
        py.allow_threads(|| {
            if nan_is_null {
                let mut ca: Float32Chunked = vals
                    .iter()
                    .map(|&val| if f32::is_nan(val) { None } else { Some(val) })
                    .collect_trusted();
                ca.rename(name);
                ca.into_series().into()
            } else {
                Series::new(name, vals).into()
            }
        })
    }

    #[staticmethod]
    pub fn new_f64(py: Python, name: &str, array: &PyArray1<f64>, nan_is_null: bool) -> PySeries {
        let array = array.readonly();
        let vals = array.as_slice().unwrap();
        py.allow_threads(|| {
            if nan_is_null {
                let mut ca: Float64Chunked = vals
                    .iter()
                    .map(|&val| if f64::is_nan(val) { None } else { Some(val) })
                    .collect_trusted();
                ca.rename(name);
                ca.into_series().into()
            } else {
                Series::new(name, vals).into()
            }
        })
    }

    pub fn struct_unnest(&self) -> PyResult<PyDataFrame> {
        let ca = self.series.struct_().map_err(PyPolarsErr::from)?;
        let df: DataFrame = ca.clone().into();
        Ok(df.into())
    }

    pub fn struct_fields(&self) -> PyResult<Vec<&str>> {
        let ca = self.series.struct_().map_err(PyPolarsErr::from)?;
        Ok(ca.fields().iter().map(|s| s.name()).collect())
    }
}

#[pymethods]
impl PySeries {
    pub fn is_sorted_flag(&self) -> bool {
        matches!(self.series.is_sorted_flag(), IsSorted::Ascending)
    }
    pub fn is_sorted_reverse_flag(&self) -> bool {
        matches!(self.series.is_sorted_flag(), IsSorted::Descending)
    }
    pub fn can_fast_explode_flag(&self) -> bool {
        match self.series.list() {
            Err(_) => false,
            Ok(list) => list._can_fast_explode(),
        }
    }

    #[staticmethod]
    pub fn new_opt_bool(name: &str, obj: &PyAny, strict: bool) -> PyResult<PySeries> {
        let len = obj.len()?;
        let mut builder = BooleanChunkedBuilder::new(name, len);

        for res in obj.iter()? {
            let item = res?;
            if item.is_none() {
                builder.append_null()
            } else {
                match item.extract::<bool>() {
                    Ok(val) => builder.append_value(val),
                    Err(e) => {
                        if strict {
                            return Err(e);
                        }
                        builder.append_null()
                    }
                }
            }
        }
        let ca = builder.finish();

        let s = ca.into_series();
        Ok(PySeries { series: s })
    }
}

fn new_primitive<'a, T>(name: &str, obj: &'a PyAny, strict: bool) -> PyResult<PySeries>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
    T::Native: FromPyObject<'a>,
{
    let len = obj.len()?;
    let mut builder = PrimitiveChunkedBuilder::<T>::new(name, len);

    for res in obj.iter()? {
        let item = res?;

        if item.is_none() {
            builder.append_null()
        } else {
            match item.extract::<T::Native>() {
                Ok(val) => builder.append_value(val),
                Err(e) => {
                    if strict {
                        return Err(e);
                    }
                    builder.append_null()
                }
            }
        }
    }
    let ca = builder.finish();

    let s = ca.into_series();
    Ok(PySeries { series: s })
}

// Init with lists that can contain Nones
macro_rules! init_method_opt {
    ($name:ident, $type:ty, $native: ty) => {
        #[pymethods]
        impl PySeries {
            #[staticmethod]
            pub fn $name(name: &str, obj: &PyAny, strict: bool) -> PyResult<PySeries> {
                new_primitive::<$type>(name, obj, strict)
            }
        }
    };
}

init_method_opt!(new_opt_u8, UInt8Type, u8);
init_method_opt!(new_opt_u16, UInt16Type, u16);
init_method_opt!(new_opt_u32, UInt32Type, u32);
init_method_opt!(new_opt_u64, UInt64Type, u64);
init_method_opt!(new_opt_i8, Int8Type, i8);
init_method_opt!(new_opt_i16, Int16Type, i16);
init_method_opt!(new_opt_i32, Int32Type, i32);
init_method_opt!(new_opt_i64, Int64Type, i64);
init_method_opt!(new_opt_f32, Float32Type, f32);
init_method_opt!(new_opt_f64, Float64Type, f64);

impl From<Series> for PySeries {
    fn from(s: Series) -> Self {
        PySeries::new(s)
    }
}

#[pymethods]
#[allow(
    clippy::wrong_self_convention,
    clippy::should_implement_trait,
    clippy::len_without_is_empty
)]
impl PySeries {
    #[staticmethod]
    pub fn new_from_anyvalues(name: &str, val: Vec<Wrap<AnyValue<'_>>>) -> PyResult<PySeries> {
        let avs = slice_extract_wrapped(&val);
        // from anyvalues is fallible
        let s = Series::from_any_values(name, avs).map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    #[staticmethod]
    pub fn new_str(name: &str, val: Wrap<Utf8Chunked>, _strict: bool) -> Self {
        let mut s = val.0.into_series();
        s.rename(name);
        PySeries::new(s)
    }

    #[staticmethod]
    pub fn new_binary(name: &str, val: Wrap<BinaryChunked>, _strict: bool) -> Self {
        let mut s = val.0.into_series();
        s.rename(name);
        PySeries::new(s)
    }

    #[staticmethod]
    pub fn new_null(name: &str, val: &PyAny, _strict: bool) -> PyResult<Self> {
        let s = Series::new_null(name, val.len()?);
        Ok(PySeries::new(s))
    }

    #[staticmethod]
    pub fn new_object(name: &str, val: Vec<ObjectValue>, _strict: bool) -> Self {
        #[cfg(feature = "object")]
        {
            let s = ObjectChunked::<ObjectValue>::new_from_vec(name, val).into_series();
            s.into()
        }
        #[cfg(not(feature = "object"))]
        {
            todo!()
        }
    }

    #[staticmethod]
    pub fn new_series_list(name: &str, val: Vec<Self>, _strict: bool) -> Self {
        let series_vec = to_series_collection(val);
        Series::new(name, &series_vec).into()
    }

    #[staticmethod]
    pub fn repeat(name: &str, val: &PyAny, n: usize, dtype: Wrap<DataType>) -> Self {
        match dtype.0 {
            DataType::Utf8 => {
                let val = val.extract::<&str>().unwrap();
                let mut ca: Utf8Chunked = (0..n).map(|_| val).collect_trusted();
                ca.rename(name);
                ca.into_series().into()
            }
            DataType::Int64 => {
                let val = val.extract::<i64>().unwrap();
                let mut ca: NoNull<Int64Chunked> = (0..n).map(|_| val).collect_trusted();
                ca.rename(name);
                ca.into_inner().into_series().into()
            }
            DataType::Float64 => {
                let val = val.extract::<f64>().unwrap();
                let mut ca: NoNull<Float64Chunked> = (0..n).map(|_| val).collect_trusted();
                ca.rename(name);
                ca.into_inner().into_series().into()
            }
            DataType::Boolean => {
                let val = val.extract::<bool>().unwrap();
                let mut ca: BooleanChunked = (0..n).map(|_| val).collect_trusted();
                ca.rename(name);
                ca.into_series().into()
            }
            dt => {
                panic!("cannot create repeat with dtype: {dt:?}");
            }
        }
    }

    #[staticmethod]
    pub fn from_arrow(name: &str, array: &PyAny) -> PyResult<Self> {
        let arr = array_to_rust(array)?;

        match arr.data_type() {
            ArrowDataType::LargeList(_) => {
                let array = arr.as_any().downcast_ref::<LargeListArray>().unwrap();

                let mut previous = 0;
                let mut fast_explode = true;
                for &o in array.offsets().as_slice()[1..].iter() {
                    if o == previous {
                        fast_explode = false;
                        break;
                    }
                    previous = o;
                }
                let mut out = unsafe { ListChunked::from_chunks(name, vec![arr]) };
                if fast_explode {
                    out.set_fast_explode()
                }
                Ok(out.into_series().into())
            }
            _ => {
                let series: Series =
                    std::convert::TryFrom::try_from((name, arr)).map_err(PyPolarsErr::from)?;
                Ok(series.into())
            }
        }
    }

    #[staticmethod]
    pub fn new_list(name: &str, seq: &PyAny, dtype: Wrap<DataType>) -> PyResult<Self> {
        py_seq_to_list(name, seq, &dtype.0).map(|s| s.into())
    }

    pub fn estimated_size(&self) -> usize {
        self.series.estimated_size()
    }

    #[cfg(feature = "object")]
    pub fn get_object(&self, index: usize) -> PyObject {
        Python::with_gil(|py| {
            if matches!(self.series.dtype(), DataType::Object(_)) {
                let obj: Option<&ObjectValue> = self.series.get_object(index).map(|any| any.into());
                obj.to_object(py)
            } else {
                py.None()
            }
        })
    }

    pub fn get_fmt(&self, index: usize, str_lengths: usize) -> String {
        let val = format!("{}", self.series.get(index).unwrap());
        if let DataType::Utf8 | DataType::Categorical(_) = self.series.dtype() {
            let v_trunc = &val[..val
                .char_indices()
                .take(str_lengths)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0)];
            if val == v_trunc {
                val
            } else {
                format!("{v_trunc}...")
            }
        } else {
            val
        }
    }

    pub fn rechunk(&mut self, in_place: bool) -> Option<Self> {
        let series = self.series.rechunk();
        if in_place {
            self.series = series;
            None
        } else {
            Some(series.into())
        }
    }

    pub fn get_idx(&self, py: Python, idx: usize) -> PyResult<PyObject> {
        Ok(Wrap(self.series.get(idx).map_err(PyPolarsErr::from)?).into_py(py))
    }

    pub fn bitand(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .bitand(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    pub fn bitor(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .bitor(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }
    pub fn bitxor(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .bitxor(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    pub fn chunk_lengths(&self) -> Vec<usize> {
        self.series.chunk_lengths().collect()
    }

    pub fn name(&self) -> &str {
        self.series.name()
    }

    pub fn rename(&mut self, name: &str) {
        self.series.rename(name);
    }

    pub fn dtype(&self, py: Python) -> PyObject {
        Wrap(self.series.dtype().clone()).to_object(py)
    }

    pub fn inner_dtype(&self, py: Python) -> Option<PyObject> {
        self.series
            .dtype()
            .inner_dtype()
            .map(|dt| Wrap(dt.clone()).to_object(py))
    }

    fn set_sorted_flag(&self, reverse: bool) -> Self {
        let mut out = self.series.clone();
        if reverse {
            out.set_sorted_flag(IsSorted::Descending);
        } else {
            out.set_sorted_flag(IsSorted::Ascending)
        }
        out.into()
    }

    pub fn mean(&self) -> Option<f64> {
        match self.series.dtype() {
            DataType::Boolean => {
                let s = self.series.cast(&DataType::UInt8).unwrap();
                s.mean()
            }
            _ => self.series.mean(),
        }
    }

    pub fn max(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .max_as_series()
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    pub fn min(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .min_as_series()
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    pub fn sum(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .sum_as_series()
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    pub fn n_chunks(&self) -> usize {
        self.series.n_chunks()
    }

    pub fn append(&mut self, other: &PySeries) -> PyResult<()> {
        self.series
            .append(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn extend(&mut self, other: &PySeries) -> PyResult<()> {
        self.series
            .extend(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn new_from_index(&self, index: usize, length: usize) -> PyResult<Self> {
        if index >= self.series.len() {
            Err(PyValueError::new_err("index is out of bounds"))
        } else {
            Ok(self.series.new_from_index(index, length).into())
        }
    }

    pub fn filter(&self, filter: &PySeries) -> PyResult<Self> {
        let filter_series = &filter.series;
        if let Ok(ca) = filter_series.bool() {
            let series = self.series.filter(ca).map_err(PyPolarsErr::from)?;
            Ok(PySeries { series })
        } else {
            Err(PyRuntimeError::new_err("Expected a boolean mask"))
        }
    }

    pub fn add(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .try_add(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    pub fn sub(&self, other: &PySeries) -> Self {
        (&self.series - &other.series).into()
    }

    pub fn mul(&self, other: &PySeries) -> Self {
        (&self.series * &other.series).into()
    }

    pub fn div(&self, other: &PySeries) -> Self {
        (&self.series / &other.series).into()
    }

    pub fn rem(&self, other: &PySeries) -> Self {
        (&self.series % &other.series).into()
    }

    pub fn sort(&mut self, reverse: bool) -> Self {
        PySeries::new(self.series.sort(reverse))
    }

    pub fn value_counts(&self, sorted: bool) -> PyResult<PyDataFrame> {
        let df = self
            .series
            .value_counts(true, sorted)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn arg_min(&self) -> Option<usize> {
        self.series.arg_min()
    }

    pub fn arg_max(&self) -> Option<usize> {
        self.series.arg_max()
    }

    pub fn take_with_series(&self, indices: &PySeries) -> PyResult<Self> {
        let idx = indices.series.idx().map_err(PyPolarsErr::from)?;
        let take = self.series.take(idx).map_err(PyPolarsErr::from)?;
        Ok(PySeries::new(take))
    }

    pub fn null_count(&self) -> PyResult<usize> {
        Ok(self.series.null_count())
    }

    pub fn has_validity(&self) -> bool {
        self.series.has_validity()
    }

    pub fn series_equal(&self, other: &PySeries, null_equal: bool, strict: bool) -> bool {
        if strict {
            self.series.eq(&other.series)
        } else if null_equal {
            self.series.series_equal_missing(&other.series)
        } else {
            self.series.series_equal(&other.series)
        }
    }
    pub fn eq(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.equal(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(Self::new(s.into_series()))
    }

    pub fn neq(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self
            .series
            .not_equal(&rhs.series)
            .map_err(PyPolarsErr::from)?;
        Ok(Self::new(s.into_series()))
    }

    pub fn gt(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.gt(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(Self::new(s.into_series()))
    }

    pub fn gt_eq(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.gt_eq(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(Self::new(s.into_series()))
    }

    pub fn lt(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.lt(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(Self::new(s.into_series()))
    }

    pub fn lt_eq(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.lt_eq(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(Self::new(s.into_series()))
    }

    pub fn _not(&self) -> PyResult<Self> {
        let bool = self.series.bool().map_err(PyPolarsErr::from)?;
        Ok((!bool).into_series().into())
    }

    pub fn as_str(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.series))
    }

    pub fn len(&self) -> usize {
        self.series.len()
    }

    pub fn to_list(&self) -> PyObject {
        Python::with_gil(|py| {
            let series = &self.series;

            fn to_list_recursive(py: Python, series: &Series) -> PyObject {
                let pylist = match series.dtype() {
                    DataType::Boolean => PyList::new(py, series.bool().unwrap()),
                    DataType::UInt8 => PyList::new(py, series.u8().unwrap()),
                    DataType::UInt16 => PyList::new(py, series.u16().unwrap()),
                    DataType::UInt32 => PyList::new(py, series.u32().unwrap()),
                    DataType::UInt64 => PyList::new(py, series.u64().unwrap()),
                    DataType::Int8 => PyList::new(py, series.i8().unwrap()),
                    DataType::Int16 => PyList::new(py, series.i16().unwrap()),
                    DataType::Int32 => PyList::new(py, series.i32().unwrap()),
                    DataType::Int64 => PyList::new(py, series.i64().unwrap()),
                    DataType::Float32 => PyList::new(py, series.f32().unwrap()),
                    DataType::Float64 => PyList::new(py, series.f64().unwrap()),
                    DataType::Decimal128(_) => todo!(),
                    DataType::Categorical(_) => {
                        PyList::new(py, series.categorical().unwrap().iter_str())
                    }
                    #[cfg(feature = "object")]
                    DataType::Object(_) => {
                        let v = PyList::empty(py);
                        for i in 0..series.len() {
                            let obj: Option<&ObjectValue> =
                                series.get_object(i).map(|any| any.into());
                            let val = obj.to_object(py);

                            v.append(val).unwrap();
                        }
                        v
                    }
                    DataType::List(_) => {
                        let v = PyList::empty(py);
                        let ca = series.list().unwrap();
                        for opt_s in ca.amortized_iter() {
                            match opt_s {
                                None => {
                                    v.append(py.None()).unwrap();
                                }
                                Some(s) => {
                                    let pylst = to_list_recursive(py, s.as_ref());
                                    v.append(pylst).unwrap();
                                }
                            }
                        }
                        v
                    }
                    DataType::Date => {
                        let ca = series.date().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Time => {
                        let ca = series.time().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Datetime(_, _) => {
                        let ca = series.datetime().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Utf8 => {
                        let ca = series.utf8().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Struct(_) => {
                        let ca = series.struct_().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Duration(_) => {
                        let ca = series.duration().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Binary => {
                        let ca = series.binary().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Null => {
                        let null: Option<u8> = None;
                        let n = series.len();
                        let iter = std::iter::repeat(null).take(n);
                        use std::iter::{Repeat, Take};
                        struct NullIter {
                            iter: Take<Repeat<Option<u8>>>,
                            n: usize,
                        }
                        impl Iterator for NullIter {
                            type Item = Option<u8>;

                            fn next(&mut self) -> Option<Self::Item> {
                                self.iter.next()
                            }
                            fn size_hint(&self) -> (usize, Option<usize>) {
                                (self.n, Some(self.n))
                            }
                        }
                        impl ExactSizeIterator for NullIter {}

                        PyList::new(py, NullIter { iter, n })
                    }
                    DataType::Unknown => {
                        panic!("to_list not implemented for unknown")
                    }
                };
                pylist.to_object(py)
            }

            let pylist = to_list_recursive(py, series);
            pylist.to_object(py)
        })
    }

    pub fn median(&self) -> Option<f64> {
        match self.series.dtype() {
            DataType::Boolean => {
                let s = self.series.cast(&DataType::UInt8).unwrap();
                s.median()
            }
            _ => self.series.median(),
        }
    }

    pub fn quantile(
        &self,
        quantile: f64,
        interpolation: Wrap<QuantileInterpolOptions>,
    ) -> PyObject {
        Python::with_gil(|py| {
            Wrap(
                self.series
                    .quantile_as_series(quantile, interpolation.0)
                    .expect("invalid quantile")
                    .get(0)
                    .unwrap_or(AnyValue::Null),
            )
            .into_py(py)
        })
    }

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    pub fn as_single_ptr(&mut self) -> PyResult<usize> {
        let ptr = self.series.as_single_ptr().map_err(PyPolarsErr::from)?;
        Ok(ptr)
    }

    pub fn to_arrow(&mut self) -> PyResult<PyObject> {
        self.rechunk(true);
        Python::with_gil(|py| {
            let pyarrow = py.import("pyarrow")?;

            arrow_interop::to_py::to_py_array(self.series.to_arrow(0), py, pyarrow)
        })
    }

    pub fn clone(&self) -> Self {
        PySeries::new(self.series.clone())
    }

    #[pyo3(signature = (lambda, output_type, skip_nulls))]
    pub fn apply_lambda(
        &self,
        lambda: &PyAny,
        output_type: Option<Wrap<DataType>>,
        skip_nulls: bool,
    ) -> PyResult<PySeries> {
        Python::with_gil(|py| {
            let series = &self.series;

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

            if matches!(
                self.series.dtype(),
                DataType::Datetime(_, _)
                    | DataType::Date
                    | DataType::Duration(_)
                    | DataType::Categorical(_)
                    | DataType::Time
            ) || !skip_nulls
            {
                let mut avs = Vec::with_capacity(self.series.len());
                let iter = self.series.iter().map(|av| {
                    let input = Wrap(av);
                    call_lambda_and_extract::<_, Wrap<AnyValue>>(py, lambda, input)
                        .unwrap()
                        .0
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
                }
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
                }
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
                }
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
                }
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
                }
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
                }
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
                }
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
                }
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
                }
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
                }
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
                }
                Some(DataType::Date) => {
                    let ca: Int32Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_date().into_series()
                }
                Some(DataType::Datetime(tu, tz)) => {
                    let ca: Int64Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_datetime(tu, tz).into_series()
                }
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
                }
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
                }
                None => return dispatch_apply!(series, apply_lambda_unknown, py, lambda),

                _ => return dispatch_apply!(series, apply_lambda_unknown, py, lambda),
            };

            Ok(PySeries::new(out))
        })
    }

    pub fn zip_with(&self, mask: &PySeries, other: &PySeries) -> PyResult<Self> {
        let mask = mask.series.bool().map_err(PyPolarsErr::from)?;
        let s = self
            .series
            .zip_with(mask, &other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(PySeries::new(s))
    }

    pub fn to_dummies(&self, sep: Option<&str>) -> PyResult<PyDataFrame> {
        let df = self.series.to_dummies(sep).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn get_list(&self, index: usize) -> Option<Self> {
        if let Ok(ca) = &self.series.list() {
            let s = ca.get(index);
            s.map(|s| s.into())
        } else {
            None
        }
    }

    pub fn peak_max(&self) -> Self {
        self.series.peak_max().into_series().into()
    }

    pub fn peak_min(&self) -> Self {
        self.series.peak_min().into_series().into()
    }

    pub fn n_unique(&self) -> PyResult<usize> {
        let n = self.series.n_unique().map_err(PyPolarsErr::from)?;
        Ok(n)
    }

    pub fn floor(&self) -> PyResult<Self> {
        let s = self.series.floor().map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    pub fn shrink_to_fit(&mut self) {
        self.series.shrink_to_fit();
    }

    pub fn dot(&self, other: &PySeries) -> Option<f64> {
        self.series.dot(&other.series)
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        // Used in pickle/pickling
        Ok(PyBytes::new(py, &bincode::serialize(&self.series).unwrap()).to_object(py))
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Used in pickle/pickling
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.series = bincode::deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn skew(&self, bias: bool) -> PyResult<Option<f64>> {
        let out = self.series.skew(bias).map_err(PyPolarsErr::from)?;
        Ok(out)
    }

    pub fn kurtosis(&self, fisher: bool, bias: bool) -> PyResult<Option<f64>> {
        let out = self
            .series
            .kurtosis(fisher, bias)
            .map_err(PyPolarsErr::from)?;
        Ok(out)
    }

    pub fn cast(&self, dtype: Wrap<DataType>, strict: bool) -> PyResult<Self> {
        let dtype = dtype.0;
        let out = if strict {
            self.series.strict_cast(&dtype)
        } else {
            self.series.cast(&dtype)
        };
        let out = out.map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    pub fn time_unit(&self) -> Option<&str> {
        if let DataType::Datetime(tu, _) | DataType::Duration(tu) = self.series.dtype() {
            Some(match tu {
                TimeUnit::Nanoseconds => "ns",
                TimeUnit::Microseconds => "us",
                TimeUnit::Milliseconds => "ms",
            })
        } else {
            None
        }
    }

    pub fn set_at_idx(&mut self, idx: PySeries, values: PySeries) -> PyResult<()> {
        // we take the value because we want a ref count
        // of 1 so that we can have mutable access
        let s = std::mem::take(&mut self.series);
        match set_at_idx(s, &idx.series, &values.series) {
            Ok(out) => {
                self.series = out;
                Ok(())
            }
            Err(e) => Err(PyErr::from(PyPolarsErr::from(e))),
        }
    }
    pub fn get_chunks(&self) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let wrap_s = py_modules::POLARS.getattr(py, "wrap_s").unwrap();
            flatten_series(&self.series)
                .into_iter()
                .map(|s| wrap_s.call1(py, (Self::new(s),)))
                .collect()
        })
    }

    pub fn is_sorted(&self, reverse: bool) -> bool {
        let options = SortOptions {
            descending: reverse,
            nulls_last: reverse,
            multithreaded: true,
        };
        self.series.is_sorted(options)
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
            pub fn $name(&self, filter: &PySeries, value: Option<$native>) -> PyResult<Self> {
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
            pub fn $name(&self, index: i64) -> Option<$type> {
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

macro_rules! impl_arithmetic {
    ($name:ident, $type:ty, $operand:tt) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self, other: $type) -> PyResult<PySeries> {
                Ok(PySeries::new(&self.series $operand other))
            }
        }
    };
}

impl_arithmetic!(add_u8, u8, +);
impl_arithmetic!(add_u16, u16, +);
impl_arithmetic!(add_u32, u32, +);
impl_arithmetic!(add_u64, u64, +);
impl_arithmetic!(add_i8, i8, +);
impl_arithmetic!(add_i16, i16, +);
impl_arithmetic!(add_i32, i32, +);
impl_arithmetic!(add_i64, i64, +);
impl_arithmetic!(add_datetime, i64, +);
impl_arithmetic!(add_duration, i64, +);
impl_arithmetic!(add_f32, f32, +);
impl_arithmetic!(add_f64, f64, +);
impl_arithmetic!(sub_u8, u8, -);
impl_arithmetic!(sub_u16, u16, -);
impl_arithmetic!(sub_u32, u32, -);
impl_arithmetic!(sub_u64, u64, -);
impl_arithmetic!(sub_i8, i8, -);
impl_arithmetic!(sub_i16, i16, -);
impl_arithmetic!(sub_i32, i32, -);
impl_arithmetic!(sub_i64, i64, -);
impl_arithmetic!(sub_datetime, i64, -);
impl_arithmetic!(sub_duration, i64, -);
impl_arithmetic!(sub_f32, f32, -);
impl_arithmetic!(sub_f64, f64, -);
impl_arithmetic!(div_u8, u8, /);
impl_arithmetic!(div_u16, u16, /);
impl_arithmetic!(div_u32, u32, /);
impl_arithmetic!(div_u64, u64, /);
impl_arithmetic!(div_i8, i8, /);
impl_arithmetic!(div_i16, i16, /);
impl_arithmetic!(div_i32, i32, /);
impl_arithmetic!(div_i64, i64, /);
impl_arithmetic!(div_f32, f32, /);
impl_arithmetic!(div_f64, f64, /);
impl_arithmetic!(mul_u8, u8, *);
impl_arithmetic!(mul_u16, u16, *);
impl_arithmetic!(mul_u32, u32, *);
impl_arithmetic!(mul_u64, u64, *);
impl_arithmetic!(mul_i8, i8, *);
impl_arithmetic!(mul_i16, i16, *);
impl_arithmetic!(mul_i32, i32, *);
impl_arithmetic!(mul_i64, i64, *);
impl_arithmetic!(mul_f32, f32, *);
impl_arithmetic!(mul_f64, f64, *);
impl_arithmetic!(rem_u8, u8, %);
impl_arithmetic!(rem_u16, u16, %);
impl_arithmetic!(rem_u32, u32, %);
impl_arithmetic!(rem_u64, u64, %);
impl_arithmetic!(rem_i8, i8, %);
impl_arithmetic!(rem_i16, i16, %);
impl_arithmetic!(rem_i32, i32, %);
impl_arithmetic!(rem_i64, i64, %);
impl_arithmetic!(rem_f32, f32, %);
impl_arithmetic!(rem_f64, f64, %);

macro_rules! impl_rhs_arithmetic {
    ($name:ident, $type:ty, $operand:ident) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self, other: $type) -> PyResult<PySeries> {
                Ok(PySeries::new(other.$operand(&self.series)))
            }
        }
    };
}

impl_rhs_arithmetic!(add_u8_rhs, u8, add);
impl_rhs_arithmetic!(add_u16_rhs, u16, add);
impl_rhs_arithmetic!(add_u32_rhs, u32, add);
impl_rhs_arithmetic!(add_u64_rhs, u64, add);
impl_rhs_arithmetic!(add_i8_rhs, i8, add);
impl_rhs_arithmetic!(add_i16_rhs, i16, add);
impl_rhs_arithmetic!(add_i32_rhs, i32, add);
impl_rhs_arithmetic!(add_i64_rhs, i64, add);
impl_rhs_arithmetic!(add_f32_rhs, f32, add);
impl_rhs_arithmetic!(add_f64_rhs, f64, add);
impl_rhs_arithmetic!(sub_u8_rhs, u8, sub);
impl_rhs_arithmetic!(sub_u16_rhs, u16, sub);
impl_rhs_arithmetic!(sub_u32_rhs, u32, sub);
impl_rhs_arithmetic!(sub_u64_rhs, u64, sub);
impl_rhs_arithmetic!(sub_i8_rhs, i8, sub);
impl_rhs_arithmetic!(sub_i16_rhs, i16, sub);
impl_rhs_arithmetic!(sub_i32_rhs, i32, sub);
impl_rhs_arithmetic!(sub_i64_rhs, i64, sub);
impl_rhs_arithmetic!(sub_f32_rhs, f32, sub);
impl_rhs_arithmetic!(sub_f64_rhs, f64, sub);
impl_rhs_arithmetic!(div_u8_rhs, u8, div);
impl_rhs_arithmetic!(div_u16_rhs, u16, div);
impl_rhs_arithmetic!(div_u32_rhs, u32, div);
impl_rhs_arithmetic!(div_u64_rhs, u64, div);
impl_rhs_arithmetic!(div_i8_rhs, i8, div);
impl_rhs_arithmetic!(div_i16_rhs, i16, div);
impl_rhs_arithmetic!(div_i32_rhs, i32, div);
impl_rhs_arithmetic!(div_i64_rhs, i64, div);
impl_rhs_arithmetic!(div_f32_rhs, f32, div);
impl_rhs_arithmetic!(div_f64_rhs, f64, div);
impl_rhs_arithmetic!(mul_u8_rhs, u8, mul);
impl_rhs_arithmetic!(mul_u16_rhs, u16, mul);
impl_rhs_arithmetic!(mul_u32_rhs, u32, mul);
impl_rhs_arithmetic!(mul_u64_rhs, u64, mul);
impl_rhs_arithmetic!(mul_i8_rhs, i8, mul);
impl_rhs_arithmetic!(mul_i16_rhs, i16, mul);
impl_rhs_arithmetic!(mul_i32_rhs, i32, mul);
impl_rhs_arithmetic!(mul_i64_rhs, i64, mul);
impl_rhs_arithmetic!(mul_f32_rhs, f32, mul);
impl_rhs_arithmetic!(mul_f64_rhs, f64, mul);
impl_rhs_arithmetic!(rem_u8_rhs, u8, rem);
impl_rhs_arithmetic!(rem_u16_rhs, u16, rem);
impl_rhs_arithmetic!(rem_u32_rhs, u32, rem);
impl_rhs_arithmetic!(rem_u64_rhs, u64, rem);
impl_rhs_arithmetic!(rem_i8_rhs, i8, rem);
impl_rhs_arithmetic!(rem_i16_rhs, i16, rem);
impl_rhs_arithmetic!(rem_i32_rhs, i32, rem);
impl_rhs_arithmetic!(rem_i64_rhs, i64, rem);
impl_rhs_arithmetic!(rem_f32_rhs, f32, rem);
impl_rhs_arithmetic!(rem_f64_rhs, f64, rem);

macro_rules! impl_eq_num {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self, rhs: $type) -> PyResult<PySeries> {
                let s = self.series.equal(rhs).map_err(PyPolarsErr::from)?;
                Ok(PySeries::new(s.into_series()))
            }
        }
    };
}

impl_eq_num!(eq_u8, u8);
impl_eq_num!(eq_u16, u16);
impl_eq_num!(eq_u32, u32);
impl_eq_num!(eq_u64, u64);
impl_eq_num!(eq_i8, i8);
impl_eq_num!(eq_i16, i16);
impl_eq_num!(eq_i32, i32);
impl_eq_num!(eq_i64, i64);
impl_eq_num!(eq_f32, f32);
impl_eq_num!(eq_f64, f64);
impl_eq_num!(eq_str, &str);

macro_rules! impl_neq_num {
    ($name:ident, $type:ty) => {
        #[allow(clippy::nonstandard_macro_braces)]
        #[pymethods]
        impl PySeries {
            pub fn $name(&self, rhs: $type) -> PyResult<PySeries> {
                let s = self.series.not_equal(rhs).map_err(PyPolarsErr::from)?;
                Ok(PySeries::new(s.into_series()))
            }
        }
    };
}

impl_neq_num!(neq_u8, u8);
impl_neq_num!(neq_u16, u16);
impl_neq_num!(neq_u32, u32);
impl_neq_num!(neq_u64, u64);
impl_neq_num!(neq_i8, i8);
impl_neq_num!(neq_i16, i16);
impl_neq_num!(neq_i32, i32);
impl_neq_num!(neq_i64, i64);
impl_neq_num!(neq_f32, f32);
impl_neq_num!(neq_f64, f64);
impl_neq_num!(neq_str, &str);

macro_rules! impl_gt_num {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self, rhs: $type) -> PyResult<PySeries> {
                let s = self.series.gt(rhs).map_err(PyPolarsErr::from)?;
                Ok(PySeries::new(s.into_series()))
            }
        }
    };
}

impl_gt_num!(gt_u8, u8);
impl_gt_num!(gt_u16, u16);
impl_gt_num!(gt_u32, u32);
impl_gt_num!(gt_u64, u64);
impl_gt_num!(gt_i8, i8);
impl_gt_num!(gt_i16, i16);
impl_gt_num!(gt_i32, i32);
impl_gt_num!(gt_i64, i64);
impl_gt_num!(gt_f32, f32);
impl_gt_num!(gt_f64, f64);
impl_gt_num!(gt_str, &str);

macro_rules! impl_gt_eq_num {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self, rhs: $type) -> PyResult<PySeries> {
                let s = self.series.gt_eq(rhs).map_err(PyPolarsErr::from)?;
                Ok(PySeries::new(s.into_series()))
            }
        }
    };
}

impl_gt_eq_num!(gt_eq_u8, u8);
impl_gt_eq_num!(gt_eq_u16, u16);
impl_gt_eq_num!(gt_eq_u32, u32);
impl_gt_eq_num!(gt_eq_u64, u64);
impl_gt_eq_num!(gt_eq_i8, i8);
impl_gt_eq_num!(gt_eq_i16, i16);
impl_gt_eq_num!(gt_eq_i32, i32);
impl_gt_eq_num!(gt_eq_i64, i64);
impl_gt_eq_num!(gt_eq_f32, f32);
impl_gt_eq_num!(gt_eq_f64, f64);
impl_gt_eq_num!(gt_eq_str, &str);

macro_rules! impl_lt_num {
    ($name:ident, $type:ty) => {
        #[allow(clippy::nonstandard_macro_braces)]
        #[pymethods]
        impl PySeries {
            pub fn $name(&self, rhs: $type) -> PyResult<PySeries> {
                let s = self.series.lt(rhs).map_err(PyPolarsErr::from)?;
                Ok(PySeries::new(s.into_series()))
            }
        }
    };
}

impl_lt_num!(lt_u8, u8);
impl_lt_num!(lt_u16, u16);
impl_lt_num!(lt_u32, u32);
impl_lt_num!(lt_u64, u64);
impl_lt_num!(lt_i8, i8);
impl_lt_num!(lt_i16, i16);
impl_lt_num!(lt_i32, i32);
impl_lt_num!(lt_i64, i64);
impl_lt_num!(lt_f32, f32);
impl_lt_num!(lt_f64, f64);
impl_lt_num!(lt_str, &str);

macro_rules! impl_lt_eq_num {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self, rhs: $type) -> PyResult<PySeries> {
                let s = self.series.lt_eq(rhs).map_err(PyPolarsErr::from)?;
                Ok(PySeries::new(s.into_series()))
            }
        }
    };
}

impl_lt_eq_num!(lt_eq_u8, u8);
impl_lt_eq_num!(lt_eq_u16, u16);
impl_lt_eq_num!(lt_eq_u32, u32);
impl_lt_eq_num!(lt_eq_u64, u64);
impl_lt_eq_num!(lt_eq_i8, i8);
impl_lt_eq_num!(lt_eq_i16, i16);
impl_lt_eq_num!(lt_eq_i32, i32);
impl_lt_eq_num!(lt_eq_i64, i64);
impl_lt_eq_num!(lt_eq_f32, f32);
impl_lt_eq_num!(lt_eq_f64, f64);
impl_lt_eq_num!(lt_eq_str, &str);

pub(crate) fn to_series_collection(ps: Vec<PySeries>) -> Vec<Series> {
    // prevent destruction of ps
    let mut ps = std::mem::ManuallyDrop::new(ps);

    // get mutable pointer and reinterpret as Series
    let p = ps.as_mut_ptr() as *mut Series;
    let len = ps.len();
    let cap = ps.capacity();

    // The pointer ownership will be transferred to Vec and this will be responsible for dealloc
    unsafe { Vec::from_raw_parts(p, len, cap) }
}

pub(crate) fn to_pyseries_collection(s: Vec<Series>) -> Vec<PySeries> {
    let mut s = std::mem::ManuallyDrop::new(s);

    let p = s.as_mut_ptr() as *mut PySeries;
    let len = s.len();
    let cap = s.capacity();

    unsafe { Vec::from_raw_parts(p, len, cap) }
}

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
        let s = to_series_collection(collection);
        assert_eq!(
            s.iter().map(|s| s.sum::<i32>()).collect::<Vec<_>>(),
            vec![Some(6)]
        );
    }
}
