use crate::dataframe::PyDataFrame;
use crate::datatypes::DataType;
use crate::error::PyPolarsEr;
use crate::utils::str_to_arrow_type;
use crate::{dispatch::ApplyLambda, npy::aligned_array, prelude::*};
use numpy::PyArray1;
use polars::chunked_array::builder::get_bitmap;
use pyo3::types::{PyList, PyTuple};
use pyo3::{exceptions::PyRuntimeError, prelude::*, Python};
use std::any::Any;

#[derive(Clone, Debug)]
pub struct ObjectValue {
    inner: PyObject,
}

impl<'a> FromPyObject<'a> for ObjectValue {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let python = gil.python();
        Ok(ObjectValue {
            inner: ob.to_object(python),
        })
    }
}

/// # Safety
///
/// The caller is responsible for checking that val is Object otherwise UB
impl From<&dyn Any> for &ObjectValue {
    fn from(val: &dyn Any) -> Self {
        unsafe { &*(val as *const dyn Any as *const ObjectValue) }
    }
}

impl ToPyObject for ObjectValue {
    fn to_object(&self, _py: Python) -> PyObject {
        self.inner.clone()
    }
}

impl Default for ObjectValue {
    fn default() -> Self {
        let gil = Python::acquire_gil();
        let python = gil.python();
        ObjectValue {
            inner: python.None(),
        }
    }
}

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
            pub fn $name(name: &str, val: &PyArray1<$type>) -> PySeries {
                unsafe {
                    PySeries {
                        series: Series::new(name, val.as_slice().unwrap()),
                    }
                }
            }
        }
    };
}

init_method!(new_i8, i8);
init_method!(new_i16, i16);
init_method!(new_i32, i32);
init_method!(new_i64, i64);
init_method!(new_f32, f32);
init_method!(new_f64, f64);
init_method!(new_bool, bool);
init_method!(new_u8, u8);
init_method!(new_u16, u16);
init_method!(new_u32, u32);
init_method!(new_u64, u64);
init_method!(new_date32, i32);
init_method!(new_date64, i64);
init_method!(new_duration_ns, i64);
init_method!(new_time_ns, i64);

// Init with lists that can contain Nones
macro_rules! init_method_opt {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            #[staticmethod]
            pub fn $name(name: &str, val: Wrap<ChunkedArray<$type>>) -> PySeries {
                let mut s = val.0.into_series();
                s.rename(name);
                PySeries { series: s }
            }
        }
    };
}

init_method_opt!(new_opt_u8, UInt8Type);
init_method_opt!(new_opt_u16, UInt16Type);
init_method_opt!(new_opt_u32, UInt32Type);
init_method_opt!(new_opt_u64, UInt64Type);
init_method_opt!(new_opt_i8, Int8Type);
init_method_opt!(new_opt_i16, Int16Type);
init_method_opt!(new_opt_i32, Int32Type);
init_method_opt!(new_opt_i64, Int64Type);
init_method_opt!(new_opt_f32, Float32Type);
init_method_opt!(new_opt_f64, Float64Type);
init_method_opt!(new_opt_bool, BooleanType);
init_method_opt!(new_opt_date32, Int32Type);
init_method_opt!(new_opt_date64, Int64Type);
init_method_opt!(new_opt_duration_ns, Int64Type);
init_method_opt!(new_opt_time_ns, Int64Type);

impl From<Series> for PySeries {
    fn from(s: Series) -> Self {
        PySeries::new(s)
    }
}

macro_rules! parse_temporal_from_str_slice {
    ($name:ident, $ca_type:ident) => {
        #[pymethods]
        impl PySeries {
            #[staticmethod]
            pub fn $name(name: &str, val: Vec<&str>, fmt: &str) -> Self {
                let parsed = $ca_type::parse_from_str_slice(name, &val, fmt);
                PySeries::new(parsed.into_series())
            }
        }
    };
}

// TODO: add other temporals
parse_temporal_from_str_slice!(parse_date32_from_str_slice, Date32Chunked);

#[pymethods]
impl PySeries {
    #[staticmethod]
    pub fn new_str(name: &str, val: Wrap<Utf8Chunked>) -> Self {
        let mut s = val.0.into_series();
        s.rename(name);
        PySeries::new(s)
    }

    #[staticmethod]
    pub fn new_object(name: &str, val: Vec<ObjectValue>) -> Self {
        let s = ObjectChunked::<ObjectValue>::new_from_vec(name, val).into_series();
        s.into()
    }

    pub fn get_object(&self, index: usize) -> PyObject {
        let gil = Python::acquire_gil();
        let python = gil.python();
        if self.series.dtype() == &ArrowDataType::Binary {
            // we don't use the null bitmap in this context as T::default is pyobject None
            let any = self.series.get_as_any(index);
            let obj: &ObjectValue = any.into();
            let gil = Python::acquire_gil();
            let python = gil.python();
            obj.to_object(python)
        } else {
            python.None()
        }
    }

    pub fn rechunk(&mut self, in_place: bool) -> Option<Self> {
        let series = self.series.rechunk(None).expect("should not fail");
        if in_place {
            self.series = series;
            None
        } else {
            Some(PySeries::new(series))
        }
    }

    pub fn chunk_lengths(&self) -> Vec<usize> {
        self.series.chunk_lengths().clone()
    }

    pub fn name(&self) -> &str {
        self.series.name()
    }

    pub fn rename(&mut self, name: &str) {
        self.series.rename(name);
    }

    pub fn dtype(&self) -> u8 {
        let dt: DataType = self.series.dtype().into();
        dt as u8
    }

    pub fn n_chunks(&self) -> usize {
        self.series.n_chunks()
    }

    pub fn limit(&self, num_elements: usize) -> PyResult<Self> {
        let series = self.series.limit(num_elements).map_err(PyPolarsEr::from)?;
        Ok(PySeries { series })
    }

    pub fn slice(&self, offset: usize, length: usize) -> PyResult<Self> {
        let series = self
            .series
            .slice(offset, length)
            .map_err(PyPolarsEr::from)?;
        Ok(PySeries { series })
    }

    pub fn append(&mut self, other: &PySeries) -> PyResult<()> {
        self.series
            .append(&other.series)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn filter(&self, filter: &PySeries) -> PyResult<Self> {
        let filter_series = &filter.series;
        if let Ok(ca) = filter_series.bool() {
            let series = self.series.filter(ca).map_err(PyPolarsEr::from)?;
            Ok(PySeries { series })
        } else {
            Err(PyRuntimeError::new_err("Expected a boolean mask"))
        }
    }

    pub fn add(&self, other: &PySeries) -> PyResult<Self> {
        Ok(PySeries::new(&self.series + &other.series))
    }

    pub fn sub(&self, other: &PySeries) -> PyResult<Self> {
        Ok(PySeries::new(&self.series - &other.series))
    }

    pub fn mul(&self, other: &PySeries) -> PyResult<Self> {
        Ok(PySeries::new(&self.series * &other.series))
    }

    pub fn div(&self, other: &PySeries) -> PyResult<Self> {
        Ok(PySeries::new(&self.series / &other.series))
    }

    pub fn head(&self, length: Option<usize>) -> PyResult<Self> {
        Ok(PySeries::new(self.series.head(length)))
    }

    pub fn tail(&self, length: Option<usize>) -> PyResult<Self> {
        Ok(PySeries::new(self.series.tail(length)))
    }

    pub fn sort_in_place(&mut self, reverse: bool) {
        self.series.sort_in_place(reverse);
    }

    pub fn sort(&mut self, reverse: bool) -> Self {
        PySeries::new(self.series.sort(reverse))
    }

    pub fn argsort(&self, reverse: bool) -> Py<PyArray1<usize>> {
        let gil = pyo3::Python::acquire_gil();
        let pyarray = PyArray1::from_vec(gil.python(), self.series.argsort(reverse));
        pyarray.to_owned()
    }

    pub fn unique(&self) -> PyResult<Self> {
        let unique = self.series.unique().map_err(PyPolarsEr::from)?;
        Ok(unique.into())
    }

    pub fn value_counts(&self) -> PyResult<PyDataFrame> {
        let df = self.series.value_counts().map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn arg_unique(&self) -> PyResult<Py<PyArray1<usize>>> {
        let gil = pyo3::Python::acquire_gil();
        let arg_unique = self.series.arg_unique().map_err(PyPolarsEr::from)?;
        let pyarray = PyArray1::from_vec(gil.python(), arg_unique);
        Ok(pyarray.to_owned())
    }

    pub fn take(&self, indices: Vec<usize>) -> Self {
        let take = self.series.take(&indices);
        PySeries::new(take)
    }

    pub fn take_with_series(&self, indices: &PySeries) -> PyResult<Self> {
        let idx = indices.series.u32().map_err(PyPolarsEr::from)?;
        let take = self.series.take(&idx);
        Ok(PySeries::new(take))
    }

    pub fn null_count(&self) -> PyResult<usize> {
        Ok(self.series.null_count())
    }

    pub fn is_null(&self) -> PySeries {
        Self::new(self.series.is_null().into_series())
    }

    pub fn is_not_null(&self) -> PySeries {
        Self::new(self.series.is_not_null().into_series())
    }

    pub fn is_unique(&self) -> PyResult<Self> {
        let ca = self.series.is_unique().map_err(PyPolarsEr::from)?;
        Ok(ca.into_series().into())
    }

    pub fn arg_true(&self) -> PyResult<Self> {
        let ca = self.series.arg_true().map_err(PyPolarsEr::from)?;
        Ok(ca.into_series().into())
    }

    pub fn sample_n(&self, n: usize, with_replacement: bool) -> PyResult<Self> {
        let s = self
            .series
            .sample_n(n, with_replacement)
            .map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn sample_frac(&self, frac: f64, with_replacement: bool) -> PyResult<Self> {
        let s = self
            .series
            .sample_frac(frac, with_replacement)
            .map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn is_duplicated(&self) -> PyResult<Self> {
        let ca = self.series.is_duplicated().map_err(PyPolarsEr::from)?;
        Ok(ca.into_series().into())
    }

    pub fn explode(&self) -> PyResult<Self> {
        let s = self.series.explode().map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn series_equal(&self, other: &PySeries, null_equal: bool) -> bool {
        if null_equal {
            self.series.series_equal_missing(&other.series)
        } else {
            self.series.series_equal(&other.series)
        }
    }
    pub fn eq(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(self.series.eq(&rhs.series).into_series()))
    }

    pub fn neq(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(self.series.neq(&rhs.series).into_series()))
    }

    pub fn gt(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(self.series.gt(&rhs.series).into_series()))
    }

    pub fn gt_eq(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(self.series.gt_eq(&rhs.series).into_series()))
    }

    pub fn lt(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(self.series.lt(&rhs.series).into_series()))
    }

    pub fn lt_eq(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(self.series.lt_eq(&rhs.series).into_series()))
    }

    pub fn _not(&self) -> PyResult<Self> {
        let bool = self.series.bool().map_err(PyPolarsEr::from)?;
        Ok((!bool).into_series().into())
    }

    pub fn as_str(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.series))
    }

    pub fn len(&self) -> usize {
        self.series.len()
    }

    pub fn to_list(&self) -> PyObject {
        let gil = Python::acquire_gil();
        let python = gil.python();

        let series = &self.series;

        let pylist = match series.dtype() {
            ArrowDataType::Boolean => PyList::new(python, series.bool().unwrap()),
            ArrowDataType::Utf8 => PyList::new(python, series.utf8().unwrap()),
            ArrowDataType::UInt8 => PyList::new(python, series.u8().unwrap()),
            ArrowDataType::UInt16 => PyList::new(python, series.u16().unwrap()),
            ArrowDataType::UInt32 => PyList::new(python, series.u32().unwrap()),
            ArrowDataType::UInt64 => PyList::new(python, series.u64().unwrap()),
            ArrowDataType::Int8 => PyList::new(python, series.i8().unwrap()),
            ArrowDataType::Int16 => PyList::new(python, series.i16().unwrap()),
            ArrowDataType::Int32 => PyList::new(python, series.i32().unwrap()),
            ArrowDataType::Int64 => PyList::new(python, series.i64().unwrap()),
            ArrowDataType::Float32 => PyList::new(python, series.f32().unwrap()),
            ArrowDataType::Float64 => PyList::new(python, series.f64().unwrap()),
            ArrowDataType::Date32(DateUnit::Day) => PyList::new(python, series.date32().unwrap()),
            ArrowDataType::Date64(DateUnit::Millisecond) => {
                PyList::new(python, series.date64().unwrap())
            }
            ArrowDataType::Time64(TimeUnit::Nanosecond) => {
                PyList::new(python, series.time64_nanosecond().unwrap())
            }
            ArrowDataType::Duration(TimeUnit::Nanosecond) => {
                PyList::new(python, series.duration_nanosecond().unwrap())
            }
            ArrowDataType::Duration(TimeUnit::Millisecond) => {
                PyList::new(python, series.duration_millisecond().unwrap())
            }
            ArrowDataType::Binary => {
                let v = PyList::empty(python);
                for i in 0..series.len() {
                    let val = series
                        .get_as_any(i)
                        .downcast_ref::<ObjectValue>()
                        .map(|obj| obj.inner.clone())
                        .unwrap_or(python.None());

                    v.append(val).unwrap();
                }
                v
            }
            dt => panic!(format!("to_list() not implemented for {:?}", dt)),
        };
        pylist.to_object(python)
    }

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    pub fn as_single_ptr(&mut self) -> PyResult<usize> {
        let ptr = self.series.as_single_ptr().map_err(PyPolarsEr::from)?;
        Ok(ptr)
    }

    pub fn drop_nulls(&self) -> Self {
        self.series.drop_nulls().into()
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
        let series = self.series.fill_none(strat).map_err(PyPolarsEr::from)?;
        Ok(PySeries::new(series))
    }

    /// Attempts to copy data to numpy arrays. If integer types have missing values
    /// they will be casted to floating point values, where NaNs are used to represent missing.
    /// Strings will be converted to python lists and booleans will be a numpy array if there are no
    /// missing values, otherwise a python list is made.
    pub fn to_numpy(&self) -> PyObject {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let series = &self.series;

        // if has null values we use floats and np.nan to represent missing values
        macro_rules! impl_to_np_array {
            ($ca:expr, $float_type:ty) => {{
                match $ca.cont_slice() {
                    Ok(slice) => PyArray1::from_slice(py, slice).to_object(py),
                    Err(_) => {
                        if $ca.null_count() == 0 {
                            let v = $ca.into_no_null_iter().collect::<Vec<_>>();
                            PyArray1::from_vec(py, v).to_object(py)
                        } else {
                            let v = $ca
                                .into_iter()
                                .map(|opt_v| match opt_v {
                                    Some(v) => v as $float_type,
                                    None => <$float_type>::NAN,
                                })
                                .collect::<Vec<_>>();
                            PyArray1::from_vec(py, v).to_object(py)
                        }
                    }
                }
            }};
        }

        // we use floating types only if null values
        match series.dtype() {
            ArrowDataType::Utf8 => self.to_list(),
            ArrowDataType::List(_) => self.to_list(),
            // hack for object
            ArrowDataType::Binary => self.to_list(),
            ArrowDataType::UInt8 => impl_to_np_array!(series.u8().unwrap(), f32),
            ArrowDataType::UInt16 => impl_to_np_array!(series.u16().unwrap(), f32),
            ArrowDataType::UInt32 => impl_to_np_array!(series.u32().unwrap(), f32),
            ArrowDataType::UInt64 => impl_to_np_array!(series.u64().unwrap(), f64),
            ArrowDataType::Int8 => impl_to_np_array!(series.i8().unwrap(), f32),
            ArrowDataType::Int16 => impl_to_np_array!(series.i16().unwrap(), f32),
            ArrowDataType::Int32 => impl_to_np_array!(series.i32().unwrap(), f32),
            ArrowDataType::Int64 => impl_to_np_array!(series.i64().unwrap(), f64),
            ArrowDataType::Float32 => impl_to_np_array!(series.f32().unwrap(), f32),
            ArrowDataType::Float64 => impl_to_np_array!(series.f64().unwrap(), f64),
            ArrowDataType::Date32(DateUnit::Day) => {
                impl_to_np_array!(series.date32().unwrap(), f32)
            }
            ArrowDataType::Date64(DateUnit::Millisecond) => {
                impl_to_np_array!(series.date64().unwrap(), f64)
            }
            ArrowDataType::Boolean => {
                let ca = series.bool().unwrap();
                if ca.null_count() == 0 {
                    let v = ca.into_no_null_iter().collect::<Vec<_>>();
                    PyArray1::from_vec(py, v).to_object(py)
                } else {
                    self.to_list()
                }
            }
            _ => unimplemented!(),
        }
    }

    pub fn clone(&self) -> Self {
        PySeries::new(self.series.clone())
    }

    pub fn apply_lambda(&self, lambda: &PyAny, output_type: &PyAny) -> PyResult<PySeries> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let series = &self.series;

        let output_type = match output_type.is_none() {
            true => None,
            false => {
                let str_repr = output_type.str().unwrap().to_str().unwrap();
                Some(str_to_arrow_type(str_repr))
            }
        };

        let out = match output_type {
            Some(ArrowDataType::Int8) => {
                let ca: Int8Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::Int16) => {
                let ca: Int16Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::Int32) => {
                let ca: Int32Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::Int64) => {
                let ca: Int64Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::UInt8) => {
                let ca: UInt8Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::UInt16) => {
                let ca: UInt16Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::UInt32) => {
                let ca: UInt32Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::UInt64) => {
                let ca: UInt64Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::Float32) => {
                let ca: Float32Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::Float64) => {
                let ca: Float64Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::Boolean) => {
                let ca: BooleanChunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::Date32(_)) => {
                let ca: Date32Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::Date64(_)) => {
                let ca: Date64Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            Some(ArrowDataType::Utf8) => {
                let ca: Utf8Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_utf8_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_series()
            }
            None => {
                return apply_method_all_arrow_series!(series, apply_lambda_unknown, py, lambda)
            }

            _ => return apply_method_all_arrow_series!(series, apply_lambda, py, lambda),
        };

        Ok(PySeries::new(out))
    }

    pub fn shift(&self, periods: i32) -> PyResult<Self> {
        let s = self.series.shift(periods).map_err(PyPolarsEr::from)?;
        Ok(PySeries::new(s))
    }

    pub fn zip_with(&self, mask: &PySeries, other: &PySeries) -> PyResult<Self> {
        let mask = mask.series.bool().map_err(PyPolarsEr::from)?;
        let s = self
            .series
            .zip_with(mask, &other.series)
            .map_err(PyPolarsEr::from)?;
        Ok(PySeries::new(s))
    }

    pub fn str_lengths(&self) -> PyResult<Self> {
        let ca = self.series.utf8().map_err(PyPolarsEr::from)?;
        let s = ca.str_lengths().into_series();
        Ok(PySeries::new(s))
    }

    pub fn str_contains(&self, pat: &str) -> PyResult<Self> {
        let ca = self.series.utf8().map_err(PyPolarsEr::from)?;
        let s = ca.contains(pat).map_err(PyPolarsEr::from)?.into_series();
        Ok(s.into())
    }

    pub fn str_replace(&self, pat: &str, val: &str) -> PyResult<Self> {
        let ca = self.series.utf8().map_err(PyPolarsEr::from)?;
        let s = ca
            .replace(pat, val)
            .map_err(PyPolarsEr::from)?
            .into_series();
        Ok(s.into())
    }

    pub fn str_replace_all(&self, pat: &str, val: &str) -> PyResult<Self> {
        let ca = self.series.utf8().map_err(PyPolarsEr::from)?;
        let s = ca
            .replace_all(pat, val)
            .map_err(PyPolarsEr::from)?
            .into_series();
        Ok(s.into())
    }

    pub fn str_to_uppercase(&self) -> PyResult<Self> {
        let ca = self.series.utf8().map_err(PyPolarsEr::from)?;
        let s = ca.to_uppercase().into_series();
        Ok(s.into())
    }

    pub fn str_to_lowercase(&self) -> PyResult<Self> {
        let ca = self.series.utf8().map_err(PyPolarsEr::from)?;
        let s = ca.to_lowercase().into_series();
        Ok(s.into())
    }

    pub fn str_parse_date32(&self, fmt: Option<&str>) -> PyResult<Self> {
        if let Ok(ca) = &self.series.utf8() {
            let ca = ca.as_date32(fmt).map_err(PyPolarsEr::from)?;
            Ok(PySeries::new(ca.into_series()))
        } else {
            Err(PyPolarsEr::Other("cannot parse date32 expected utf8 type".into()).into())
        }
    }

    pub fn str_parse_date64(&self, fmt: Option<&str>) -> PyResult<Self> {
        if let Ok(ca) = &self.series.utf8() {
            let ca = ca.as_date64(fmt).map_err(PyPolarsEr::from)?;
            Ok(ca.into_series().into())
        } else {
            Err(PyPolarsEr::Other("cannot parse date64 expected utf8 type".into()).into())
        }
    }

    pub fn datetime_str_fmt(&self, fmt: &str) -> PyResult<Self> {
        let s = self
            .series
            .datetime_str_fmt(fmt)
            .map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn as_duration(&self) -> PyResult<Self> {
        match self.series.dtype() {
            ArrowDataType::Date64(_) => {
                let ca = self.series.date64().unwrap().as_duration();
                Ok(ca.into_series().into())
            }
            ArrowDataType::Date32(_) => {
                let ca = self.series.date32().unwrap().as_duration();
                Ok(ca.into_series().into())
            }
            _ => Err(PyPolarsEr::Other(
                "Only date32 and date64 can be transformed as duration".into(),
            )
            .into()),
        }
    }

    pub fn to_dummies(&self) -> PyResult<PyDataFrame> {
        let df = self.series.to_dummies().map_err(PyPolarsEr::from)?;
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
    pub fn rolling_sum(
        &self,
        window_size: usize,
        weight: Option<Vec<f64>>,
        ignore_null: bool,
    ) -> PyResult<Self> {
        let s = self
            .series
            .rolling_sum(window_size, weight.as_deref(), ignore_null)
            .map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn rolling_mean(
        &self,
        window_size: usize,
        weight: Option<Vec<f64>>,
        ignore_null: bool,
    ) -> PyResult<Self> {
        let s = self
            .series
            .rolling_mean(window_size, weight.as_deref(), ignore_null)
            .map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn rolling_max(
        &self,
        window_size: usize,
        weight: Option<Vec<f64>>,
        ignore_null: bool,
    ) -> PyResult<Self> {
        let s = self
            .series
            .rolling_max(window_size, weight.as_deref(), ignore_null)
            .map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }
    pub fn rolling_min(
        &self,
        window_size: usize,
        weight: Option<Vec<f64>>,
        ignore_null: bool,
    ) -> PyResult<Self> {
        let s = self
            .series
            .rolling_min(window_size, weight.as_deref(), ignore_null)
            .map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn year(&self) -> PyResult<Self> {
        let s = self.series.year().map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn month(&self) -> PyResult<Self> {
        let s = self.series.month().map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn day(&self) -> PyResult<Self> {
        let s = self.series.day().map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn ordinal_day(&self) -> PyResult<Self> {
        let s = self.series.ordinal_day().map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn hour(&self) -> PyResult<Self> {
        let s = self.series.hour().map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn minute(&self) -> PyResult<Self> {
        let s = self.series.minute().map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn second(&self) -> PyResult<Self> {
        let s = self.series.second().map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn nanosecond(&self) -> PyResult<Self> {
        let s = self.series.nanosecond().map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }
}

macro_rules! impl_ufuncs {
    ($name:ident, $type:ty, $unsafe_from_ptr_method:ident) => {
        #[pymethods]
        impl PySeries {
            // applies a ufunc by accepting a lambda out: ufunc(*args, out=out)
            // the out array is allocated in this method, send to Python and once the ufunc is applied
            // ownership is taken by Rust again to prevent memory leak.
            // if the ufunc fails, we first must take ownership back.
            pub fn $name(&self, lambda: &PyAny) -> PyResult<PySeries> {
                // numpy array object, and a *mut ptr
                let gil = Python::acquire_gil();
                let py = gil.python();
                let size = self.len();
                let (out_array, ptr) = unsafe { aligned_array::<$type>(py, size) };

                debug_assert_eq!(out_array.get_refcnt(), 1);
                // inserting it in a tuple increase the reference count by 1.
                let args = PyTuple::new(py, &[out_array]);
                debug_assert_eq!(out_array.get_refcnt(), 2);

                // whatever the result, we must take the leaked memory ownership back
                let s = match lambda.call1(args) {
                    Ok(_) => {
                        // if this assert fails, the lambda has taken a reference to the object, so we must panic
                        // args and the lambda return have a reference, making a total of 3
                        assert_eq!(out_array.get_refcnt(), 3);
                        self.$unsafe_from_ptr_method(ptr as usize, size)
                    }
                    Err(e) => {
                        // first take ownership from the leaked memory
                        // so the destructor gets called when we go out of scope
                        self.$unsafe_from_ptr_method(ptr as usize, size);
                        // return error information
                        return Err(e);
                    }
                };

                Ok(s)
            }
        }
    };
}
impl_ufuncs!(apply_ufunc_f32, f32, unsafe_from_ptr_f32);
impl_ufuncs!(apply_ufunc_f64, f64, unsafe_from_ptr_f64);
impl_ufuncs!(apply_ufunc_u8, u8, unsafe_from_ptr_u8);
impl_ufuncs!(apply_ufunc_u16, u16, unsafe_from_ptr_u16);
impl_ufuncs!(apply_ufunc_u32, u32, unsafe_from_ptr_u32);
impl_ufuncs!(apply_ufunc_u64, u64, unsafe_from_ptr_u64);
impl_ufuncs!(apply_ufunc_i8, i8, unsafe_from_ptr_i8);
impl_ufuncs!(apply_ufunc_i16, i16, unsafe_from_ptr_i16);
impl_ufuncs!(apply_ufunc_i32, i32, unsafe_from_ptr_i32);
impl_ufuncs!(apply_ufunc_i64, i64, unsafe_from_ptr_i64);

macro_rules! impl_set_with_mask {
    ($name:ident, $native:ty, $cast:ident, $variant:ident) => {
        fn $name(series: &Series, filter: &PySeries, value: Option<$native>) -> Result<Series> {
            let mask = filter.series.bool()?;
            let ca = series.$cast()?;
            let new = ca.set(mask, value)?;
            Ok(new.into_series())
        }

        #[pymethods]
        impl PySeries {
            pub fn $name(&self, filter: &PySeries, value: Option<$native>) -> PyResult<Self> {
                let series = $name(&self.series, filter, value).map_err(PyPolarsEr::from)?;
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

macro_rules! impl_set_at_idx {
    ($name:ident, $native:ty, $cast:ident, $variant:ident) => {
        fn $name(series: &Series, idx: &[usize], value: Option<$native>) -> Result<Series> {
            let ca = series.$cast()?;
            let new = ca.set_at_idx(&idx, value)?;
            Ok(new.into_series())
        }

        #[pymethods]
        impl PySeries {
            pub fn $name(&self, idx: &PyArray1<usize>, value: Option<$native>) -> PyResult<Self> {
                let idx = unsafe { idx.as_slice().unwrap() };
                let series = $name(&self.series, &idx, value).map_err(PyPolarsEr::from)?;
                Ok(Self::new(series))
            }
        }
    };
}

impl_set_at_idx!(set_at_idx_str, &str, utf8, Utf8);
impl_set_at_idx!(set_at_idx_f64, f64, f64, Float64);
impl_set_at_idx!(set_at_idx_f32, f32, f32, Float32);
impl_set_at_idx!(set_at_idx_u8, u8, u8, UInt8);
impl_set_at_idx!(set_at_idx_u16, u16, u16, UInt16);
impl_set_at_idx!(set_at_idx_u32, u32, u32, UInt32);
impl_set_at_idx!(set_at_idx_u64, u64, u64, UInt64);
impl_set_at_idx!(set_at_idx_i8, i8, i8, Int8);
impl_set_at_idx!(set_at_idx_i16, i16, i16, Int16);
impl_set_at_idx!(set_at_idx_i32, i32, i32, Int32);
impl_set_at_idx!(set_at_idx_i64, i64, i64, Int64);

macro_rules! impl_get {
    ($name:ident, $series_variant:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self, index: usize) -> Option<$type> {
                if let Ok(ca) = self.series.$series_variant() {
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
impl_get!(get_date32, date32, i32);
impl_get!(get_date64, date64, i64);

// Not public methods.
macro_rules! impl_unsafe_from_ptr {
    ($name:ident, $ca_type:ident) => {
        impl PySeries {
            fn $name(&self, ptr: usize, len: usize) -> Self {
                let av = unsafe { AlignedVec::from_ptr(ptr, len, len) };
                let (null_count, null_bitmap) = get_bitmap(self.series.chunks()[0].as_ref());
                let ca = ChunkedArray::<$ca_type>::new_from_owned_with_null_bitmap(
                    self.name(),
                    av,
                    null_bitmap,
                    null_count,
                );
                Self::new(ca.into_series())
            }
        }
    };
}

impl_unsafe_from_ptr!(unsafe_from_ptr_f32, Float32Type);
impl_unsafe_from_ptr!(unsafe_from_ptr_f64, Float64Type);
impl_unsafe_from_ptr!(unsafe_from_ptr_u8, UInt8Type);
impl_unsafe_from_ptr!(unsafe_from_ptr_u16, UInt16Type);
impl_unsafe_from_ptr!(unsafe_from_ptr_u32, UInt32Type);
impl_unsafe_from_ptr!(unsafe_from_ptr_u64, UInt64Type);
impl_unsafe_from_ptr!(unsafe_from_ptr_i8, Int8Type);
impl_unsafe_from_ptr!(unsafe_from_ptr_i16, Int16Type);
impl_unsafe_from_ptr!(unsafe_from_ptr_i32, Int32Type);
impl_unsafe_from_ptr!(unsafe_from_ptr_i64, Int64Type);

macro_rules! impl_cast {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self) -> PyResult<PySeries> {
                let s = self.series.cast::<$type>().map_err(PyPolarsEr::from)?;
                Ok(PySeries::new(s))
            }
        }
    };
}

impl_cast!(cast_u8, UInt8Type);
impl_cast!(cast_u16, UInt16Type);
impl_cast!(cast_u32, UInt32Type);
impl_cast!(cast_u64, UInt64Type);
impl_cast!(cast_i8, Int8Type);
impl_cast!(cast_i16, Int16Type);
impl_cast!(cast_i32, Int32Type);
impl_cast!(cast_i64, Int64Type);
impl_cast!(cast_f32, Float32Type);
impl_cast!(cast_f64, Float64Type);
impl_cast!(cast_date32, Date32Type);
impl_cast!(cast_date64, Date64Type);
impl_cast!(cast_time64ns, Time64NanosecondType);
impl_cast!(cast_duration_ns, DurationNanosecondType);
impl_cast!(cast_str, Utf8Type);

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

macro_rules! impl_sum {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self) -> PyResult<Option<$type>> {
                Ok(self.series.sum())
            }
        }
    };
}

impl_sum!(sum_u8, u8);
impl_sum!(sum_u16, u16);
impl_sum!(sum_u32, u32);
impl_sum!(sum_u64, u64);
impl_sum!(sum_i8, i8);
impl_sum!(sum_i16, i16);
impl_sum!(sum_i32, i32);
impl_sum!(sum_i64, i64);
impl_sum!(sum_f32, f32);
impl_sum!(sum_f64, f64);

macro_rules! impl_min {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self) -> PyResult<Option<$type>> {
                Ok(self.series.min())
            }
        }
    };
}

impl_min!(min_u8, u8);
impl_min!(min_u16, u16);
impl_min!(min_u32, u32);
impl_min!(min_u64, u64);
impl_min!(min_i8, i8);
impl_min!(min_i16, i16);
impl_min!(min_i32, i32);
impl_min!(min_i64, i64);
impl_min!(min_f32, f32);
impl_min!(min_f64, f64);

macro_rules! impl_max {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self) -> PyResult<Option<$type>> {
                Ok(self.series.max())
            }
        }
    };
}

impl_max!(max_u8, u8);
impl_max!(max_u16, u16);
impl_max!(max_u32, u32);
impl_max!(max_u64, u64);
impl_max!(max_i8, i8);
impl_max!(max_i16, i16);
impl_max!(max_i32, i32);
impl_max!(max_i64, i64);
impl_max!(max_f32, f32);
impl_max!(max_f64, f64);

macro_rules! impl_mean {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self) -> PyResult<Option<$type>> {
                Ok(self.series.mean())
            }
        }
    };
}

impl_mean!(mean_u8, u8);
impl_mean!(mean_u16, u16);
impl_mean!(mean_u32, u32);
impl_mean!(mean_u64, u64);
impl_mean!(mean_i8, i8);
impl_mean!(mean_i16, i16);
impl_mean!(mean_i32, i32);
impl_mean!(mean_i64, i64);
impl_mean!(mean_f32, f32);
impl_mean!(mean_f64, f64);

macro_rules! impl_eq_num {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            pub fn $name(&self, rhs: $type) -> PyResult<PySeries> {
                Ok(PySeries::new(self.series.eq(rhs).into_series()))
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
        #[pymethods]
        impl PySeries {
            pub fn $name(&self, rhs: $type) -> PyResult<PySeries> {
                Ok(PySeries::new(self.series.neq(rhs).into_series()))
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
                Ok(PySeries::new(self.series.gt(rhs).into_series()))
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
                Ok(PySeries::new(self.series.gt_eq(rhs).into_series()))
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
        #[pymethods]
        impl PySeries {
            pub fn $name(&self, rhs: $type) -> PyResult<PySeries> {
                Ok(PySeries::new(self.series.lt(rhs).into_series()))
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
                Ok(PySeries::new(self.series.lt_eq(rhs).into_series()))
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

    // The pointer ownership will be transferred to Vec and this will be responsible for dealoc
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
