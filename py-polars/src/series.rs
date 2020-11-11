use crate::datatypes::DataType;
use crate::error::PyPolarsEr;
use crate::{dispatch::ApplyLambda, npy::aligned_array, prelude::*};
use numpy::PyArray1;
use polars::chunked_array::builder::get_bitmap;
use pyo3::types::{PyList, PyTuple};
use pyo3::{exceptions::PyRuntimeError, prelude::*, Python};

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
        if let Series::Bool(ca) = filter_series {
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

    pub fn arg_unique(&self) -> PyResult<Py<PyArray1<usize>>> {
        let gil = pyo3::Python::acquire_gil();
        let arg_unique = self.series.arg_unique().map_err(PyPolarsEr::from)?;
        let pyarray = PyArray1::from_vec(gil.python(), arg_unique);
        Ok(pyarray.to_owned())
    }

    pub fn take(&self, indices: Vec<usize>) -> PyResult<Self> {
        let take = self.series.take(&indices).map_err(PyPolarsEr::from)?;
        Ok(PySeries::new(take))
    }

    pub fn take_with_series(&self, indices: &PySeries) -> PyResult<Self> {
        let idx = indices.series.u32().map_err(PyPolarsEr::from)?;
        let take = self.series.take(&idx).map_err(PyPolarsEr::from)?;
        Ok(PySeries::new(take))
    }

    pub fn null_count(&self) -> PyResult<usize> {
        Ok(self.series.null_count())
    }

    pub fn is_null(&self) -> PySeries {
        Self::new(Series::Bool(self.series.is_null()))
    }

    pub fn is_not_null(&self) -> PySeries {
        Self::new(Series::Bool(self.series.is_not_null()))
    }

    pub fn is_unique(&self) -> PyResult<Self> {
        let ca = self.series.is_unique().map_err(PyPolarsEr::from)?;
        Ok(Series::Bool(ca).into())
    }

    pub fn is_duplicated(&self) -> PyResult<Self> {
        let ca = self.series.is_duplicated().map_err(PyPolarsEr::from)?;
        Ok(Series::Bool(ca).into())
    }

    pub fn series_equal(&self, other: &PySeries, null_equal: bool) -> bool {
        if null_equal {
            self.series.series_equal_missing(&other.series)
        } else {
            self.series.series_equal(&other.series)
        }
    }
    pub fn eq(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(Series::Bool(self.series.eq(&rhs.series))))
    }

    pub fn neq(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(Series::Bool(self.series.neq(&rhs.series))))
    }

    pub fn gt(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(Series::Bool(self.series.gt(&rhs.series))))
    }

    pub fn gt_eq(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(Series::Bool(self.series.gt_eq(&rhs.series))))
    }

    pub fn lt(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(Series::Bool(self.series.lt(&rhs.series))))
    }

    pub fn lt_eq(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(Series::Bool(self.series.lt_eq(&rhs.series))))
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

        let pylist = match &self.series {
            Series::UInt8(ca) => PyList::new(python, ca),
            Series::UInt16(ca) => PyList::new(python, ca),
            Series::UInt32(ca) => PyList::new(python, ca),
            Series::UInt64(ca) => PyList::new(python, ca),
            Series::Int8(ca) => PyList::new(python, ca),
            Series::Int16(ca) => PyList::new(python, ca),
            Series::Int32(ca) => PyList::new(python, ca),
            Series::Int64(ca) => PyList::new(python, ca),
            Series::Float32(ca) => PyList::new(python, ca),
            Series::Float64(ca) => PyList::new(python, ca),
            Series::Date32(ca) => PyList::new(python, ca),
            Series::Date64(ca) => PyList::new(python, ca),
            Series::Time64Nanosecond(ca) => PyList::new(python, ca),
            Series::DurationNanosecond(ca) => PyList::new(python, ca),
            Series::Bool(ca) => PyList::new(python, ca),
            Series::Utf8(ca) => PyList::new(python, ca),
            _ => todo!(),
        };
        pylist.to_object(python)
    }

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    pub fn as_single_ptr(&mut self) -> usize {
        self.series.as_single_ptr()
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
            ($ca:ident, $float_type:ty) => {{
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

        match series {
            Series::UInt8(ca) => impl_to_np_array!(ca, f32),
            Series::UInt16(ca) => impl_to_np_array!(ca, f32),
            Series::UInt32(ca) => impl_to_np_array!(ca, f32),
            Series::UInt64(ca) => impl_to_np_array!(ca, f64),
            Series::Int8(ca) => impl_to_np_array!(ca, f32),
            Series::Int16(ca) => impl_to_np_array!(ca, f32),
            Series::Int32(ca) => impl_to_np_array!(ca, f32),
            Series::Int64(ca) => impl_to_np_array!(ca, f64),
            Series::Float32(ca) => impl_to_np_array!(ca, f32),
            Series::Float64(ca) => impl_to_np_array!(ca, f64),
            Series::Date32(ca) => impl_to_np_array!(ca, f32),
            Series::Date64(ca) => impl_to_np_array!(ca, f64),
            Series::Bool(ca) => {
                if ca.null_count() == 0 {
                    let v = ca.into_no_null_iter().collect::<Vec<_>>();
                    PyArray1::from_vec(py, v).to_object(py)
                } else {
                    self.to_list()
                }
            }
            Series::Utf8(_) => self.to_list(),
            Series::List(_) => self.to_list(),
            _ => todo!(),
        }
    }

    pub fn clone(&self) -> Self {
        PySeries::new(self.series.clone())
    }

    pub fn apply_lambda(&self, lambda: &PyAny, out_dtype: Option<u8>) -> PyResult<PySeries> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let series = &self.series;

        let out = match out_dtype {
            Some(0) => {
                let ca: Int8Chunked = apply_method_all_series!(
                    series,
                    apply_lambda_with_primitive_dtype,
                    py,
                    lambda
                )?;
                ca.into_series()
            }
            Some(1) => {
                let ca: Int16Chunked = apply_method_all_series!(
                    series,
                    apply_lambda_with_primitive_dtype,
                    py,
                    lambda
                )?;
                ca.into_series()
            }
            Some(2) => {
                let ca: Int32Chunked = apply_method_all_series!(
                    series,
                    apply_lambda_with_primitive_dtype,
                    py,
                    lambda
                )?;
                ca.into_series()
            }
            Some(3) => {
                let ca: Int64Chunked = apply_method_all_series!(
                    series,
                    apply_lambda_with_primitive_dtype,
                    py,
                    lambda
                )?;
                ca.into_series()
            }
            Some(4) => {
                let ca: UInt8Chunked = apply_method_all_series!(
                    series,
                    apply_lambda_with_primitive_dtype,
                    py,
                    lambda
                )?;
                ca.into_series()
            }
            Some(5) => {
                let ca: UInt16Chunked = apply_method_all_series!(
                    series,
                    apply_lambda_with_primitive_dtype,
                    py,
                    lambda
                )?;
                ca.into_series()
            }
            Some(6) => {
                let ca: UInt32Chunked = apply_method_all_series!(
                    series,
                    apply_lambda_with_primitive_dtype,
                    py,
                    lambda
                )?;
                ca.into_series()
            }
            Some(7) => {
                let ca: UInt64Chunked = apply_method_all_series!(
                    series,
                    apply_lambda_with_primitive_dtype,
                    py,
                    lambda
                )?;
                ca.into_series()
            }
            Some(8) => {
                let ca: Float32Chunked = apply_method_all_series!(
                    series,
                    apply_lambda_with_primitive_dtype,
                    py,
                    lambda
                )?;
                ca.into_series()
            }
            Some(9) => {
                let ca: Float64Chunked = apply_method_all_series!(
                    series,
                    apply_lambda_with_primitive_dtype,
                    py,
                    lambda
                )?;
                ca.into_series()
            }
            Some(10) => {
                let ca: BooleanChunked = apply_method_all_series!(
                    series,
                    apply_lambda_with_primitive_dtype,
                    py,
                    lambda
                )?;
                ca.into_series()
            }
            None => return apply_method_all_series!(series, apply_lambda, py, lambda),

            _ => return apply_method_all_series!(series, apply_lambda, py, lambda),
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
        if let Series::Utf8(ca) = &self.series {
            let ca = ca.as_date32(fmt).map_err(PyPolarsEr::from)?;
            Ok(PySeries::new(ca.into_series()))
        } else {
            Err(PyPolarsEr::Other("cannot parse date32 expected utf8 type".into()).into())
        }
    }

    pub fn str_parse_date64(&self, fmt: Option<&str>) -> PyResult<Self> {
        if let Series::Utf8(ca) = &self.series {
            let ca = ca.as_date64(fmt).map_err(PyPolarsEr::from)?;
            Ok(ca.into_series().into())
        } else {
            Err(PyPolarsEr::Other("cannot parse date64 expected utf8 type".into()).into())
        }
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

    pub fn get_list(&self, index: usize) -> Option<Self> {
        if let Series::List(ca) = &self.series {
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
                let (out_array, ptr) = aligned_array::<$type>(py, size);

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
            Ok(Series::$variant(new))
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
            Ok(Series::$variant(new))
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
                if let Series::$series_variant(ca) = &self.series {
                    ca.get(index)
                } else {
                    None
                }
            }
        }
    };
}

impl_get!(get_f32, Float32, f32);
impl_get!(get_f64, Float64, f64);
impl_get!(get_u8, UInt8, u8);
impl_get!(get_u16, UInt16, u16);
impl_get!(get_u32, UInt32, u32);
impl_get!(get_u64, UInt64, u64);
impl_get!(get_i8, Int8, i8);
impl_get!(get_i16, Int16, i16);
impl_get!(get_i32, Int32, i32);
impl_get!(get_i64, Int64, i64);
impl_get!(get_str, Utf8, &str);
impl_get!(get_date32, Date32, i32);
impl_get!(get_date64, Date64, i64);

// Not public methods.
macro_rules! impl_unsafe_from_ptr {
    ($name:ident, $series_variant:ident) => {
        impl PySeries {
            fn $name(&self, ptr: usize, len: usize) -> Self {
                let av = unsafe { AlignedVec::from_ptr(ptr, len, len) };
                let (null_count, null_bitmap) = get_bitmap(self.series.chunks()[0].as_ref());
                let ca = ChunkedArray::new_from_owned_with_null_bitmap(
                    self.name(),
                    av,
                    null_bitmap,
                    null_count,
                );
                Self::new(Series::$series_variant(ca))
            }
        }
    };
}

impl_unsafe_from_ptr!(unsafe_from_ptr_f32, Float32);
impl_unsafe_from_ptr!(unsafe_from_ptr_f64, Float64);
impl_unsafe_from_ptr!(unsafe_from_ptr_u8, UInt8);
impl_unsafe_from_ptr!(unsafe_from_ptr_u16, UInt16);
impl_unsafe_from_ptr!(unsafe_from_ptr_u32, UInt32);
impl_unsafe_from_ptr!(unsafe_from_ptr_u64, UInt64);
impl_unsafe_from_ptr!(unsafe_from_ptr_i8, Int8);
impl_unsafe_from_ptr!(unsafe_from_ptr_i16, Int16);
impl_unsafe_from_ptr!(unsafe_from_ptr_i32, Int32);
impl_unsafe_from_ptr!(unsafe_from_ptr_i64, Int64);

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
                Ok(PySeries::new(Series::Bool(self.series.eq(rhs))))
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
                Ok(PySeries::new(Series::Bool(self.series.neq(rhs))))
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
                Ok(PySeries::new(Series::Bool(self.series.gt(rhs))))
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
                Ok(PySeries::new(Series::Bool(self.series.gt_eq(rhs))))
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
                Ok(PySeries::new(Series::Bool(self.series.lt(rhs))))
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
                Ok(PySeries::new(Series::Bool(self.series.lt_eq(rhs))))
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
