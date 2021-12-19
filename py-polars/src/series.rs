use crate::apply::series::ApplyLambda;
use crate::arrow_interop::to_rust::array_to_rust;
use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsEr;
use crate::list_construction::py_seq_to_list;
use crate::utils::{reinterpret, str_to_polarstype};
use crate::{
    arrow_interop,
    npy::{aligned_array, get_refcnt},
    prelude::*,
};
use numpy::PyArray1;
use polars_core::utils::CustomIterTools;
use pyo3::types::{PyBytes, PyList, PyTuple};
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
            pub fn $name(name: &str, val: &PyArray1<$type>, _strict: bool) -> PySeries {
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
init_method!(new_bool, bool);
init_method!(new_u8, u8);
init_method!(new_u16, u16);
init_method!(new_u32, u32);
init_method!(new_u64, u64);

#[pymethods]
impl PySeries {
    #[staticmethod]
    pub fn new_f32(name: &str, val: &PyArray1<f32>, nan_is_null: bool) -> PySeries {
        // numpy array as slice is unsafe
        unsafe {
            if nan_is_null {
                let mut ca: Float32Chunked = val
                    .as_slice()
                    .expect("contiguous array")
                    .iter()
                    .map(|&val| if f32::is_nan(val) { None } else { Some(val) })
                    .collect_trusted();
                ca.rename(name);
                ca.into_series().into()
            } else {
                Series::new(name, val.as_slice().unwrap()).into()
            }
        }
    }

    #[staticmethod]
    pub fn new_f64(name: &str, val: &PyArray1<f64>, nan_is_null: bool) -> PySeries {
        // numpy array as slice is unsafe
        unsafe {
            if nan_is_null {
                let mut ca: Float64Chunked = val
                    .as_slice()
                    .expect("contiguous array")
                    .iter()
                    .map(|&val| if f64::is_nan(val) { None } else { Some(val) })
                    .collect_trusted();
                ca.rename(name);
                ca.into_series().into()
            } else {
                Series::new(name, val.as_slice().unwrap()).into()
            }
        }
    }
}

#[pymethods]
impl PySeries {
    #[staticmethod]
    pub fn new_opt_bool(name: &str, obj: &PyAny, strict: bool) -> PyResult<PySeries> {
        let (seq, len) = get_pyseq(obj)?;
        let mut builder = BooleanChunkedBuilder::new(name, len);

        for res in seq.iter()? {
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

// Init with lists that can contain Nones
macro_rules! init_method_opt {
    ($name:ident, $type:ty, $native: ty) => {
        #[pymethods]
        impl PySeries {
            #[staticmethod]
            pub fn $name(name: &str, obj: &PyAny, strict: bool) -> PyResult<PySeries> {
                let (seq, len) = get_pyseq(obj)?;
                let mut builder = PrimitiveChunkedBuilder::<$type>::new(name, len);

                for res in seq.iter()? {
                    let item = res?;

                    if item.is_none() {
                        builder.append_null()
                    } else {
                        match item.extract::<$native>() {
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
    pub fn new_str(name: &str, val: Wrap<Utf8Chunked>, _strict: bool) -> Self {
        let mut s = val.0.into_series();
        s.rename(name);
        PySeries::new(s)
    }

    #[staticmethod]
    pub fn new_object(name: &str, val: Vec<ObjectValue>, _strict: bool) -> Self {
        let s = ObjectChunked::<ObjectValue>::new_from_vec(name, val).into_series();
        s.into()
    }

    #[staticmethod]
    pub fn new_series_list(name: &str, val: Vec<Self>, _strict: bool) -> Self {
        let series_vec = to_series_collection(val);
        Series::new(name, &series_vec).into()
    }

    #[staticmethod]
    pub fn repeat(name: &str, val: &PyAny, n: usize, dtype: &PyAny) -> Self {
        let str_repr = dtype.str().unwrap().to_str().unwrap();
        let dtype = str_to_polarstype(str_repr);

        match dtype {
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
                panic!("cannot create repeat with dtype: {:?}", dt);
            }
        }
    }

    #[staticmethod]
    pub fn from_arrow(name: &str, array: &PyAny) -> PyResult<Self> {
        let arr = array_to_rust(array)?;
        let series: Series =
            std::convert::TryFrom::try_from((name, arr)).map_err(PyPolarsEr::from)?;
        Ok(series.into())
    }

    #[staticmethod]
    pub fn new_list(name: &str, seq: &PyAny, dtype: &PyAny) -> PyResult<Self> {
        py_seq_to_list(name, seq, dtype).map(|s| s.into())
    }

    /// Should only be called for Series with null types.
    /// This will cast to floats so that `None = np.nan`
    pub fn to_numpy(&self, py: Python) -> PyObject {
        let s = &self.series;
        if s.bit_repr_is_large() {
            let s = s.cast(&DataType::Float64).unwrap();
            let ca = s.f64().unwrap();
            let np_arr = PyArray1::from_iter(
                py,
                ca.into_iter().map(|opt_v| match opt_v {
                    Some(v) => v,
                    None => f64::NAN,
                }),
            );
            np_arr.into_py(py)
        } else {
            let s = s.cast(&DataType::Float32).unwrap();
            let ca = s.f32().unwrap();
            let np_arr = PyArray1::from_iter(
                py,
                ca.into_iter().map(|opt_v| match opt_v {
                    Some(v) => v,
                    None => f32::NAN,
                }),
            );
            np_arr.into_py(py)
        }
    }

    pub fn get_object(&self, index: usize) -> PyObject {
        let gil = Python::acquire_gil();
        let python = gil.python();
        if matches!(self.series.dtype(), DataType::Object(_)) {
            let obj: Option<&ObjectValue> = self.series.get_object(index).map(|any| any.into());
            obj.to_object(python)
        } else {
            python.None()
        }
    }

    pub fn get_fmt(&self, index: usize) -> String {
        format!("{}", self.series.get(index))
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

    pub fn get_idx(&self, py: Python, idx: usize) -> PyObject {
        Wrap(self.series.get(idx)).into_py(py)
    }

    pub fn bitand(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .bitand(&other.series)
            .map_err(PyPolarsEr::from)?;
        Ok(out.into())
    }

    pub fn bitor(&self, other: &PySeries) -> PyResult<Self> {
        let out = self.series.bitor(&other.series).map_err(PyPolarsEr::from)?;
        Ok(out.into())
    }
    pub fn bitxor(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .bitxor(&other.series)
            .map_err(PyPolarsEr::from)?;
        Ok(out.into())
    }

    pub fn cumsum(&self, reverse: bool) -> Self {
        self.series.cumsum(reverse).into()
    }

    pub fn cummax(&self, reverse: bool) -> Self {
        self.series.cummax(reverse).into()
    }

    pub fn cummin(&self, reverse: bool) -> Self {
        self.series.cummin(reverse).into()
    }

    pub fn cumprod(&self, reverse: bool) -> Self {
        self.series.cumprod(reverse).into()
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

    pub fn mean(&self) -> Option<f64> {
        match self.series.dtype() {
            DataType::Boolean => {
                let s = self.series.cast(&DataType::UInt8).unwrap();
                s.mean()
            }
            _ => self.series.mean(),
        }
    }

    pub fn max(&self) -> PyObject {
        let gil = Python::acquire_gil();
        let py = gil.python();
        match self.series.dtype() {
            DataType::Float32 | DataType::Float64 => self.series.max::<f64>().to_object(py),
            DataType::Boolean => self.series.max::<u32>().map(|v| v == 1).to_object(py),
            _ => self.series.max::<i64>().to_object(py),
        }
    }

    pub fn min(&self) -> PyObject {
        let gil = Python::acquire_gil();
        let py = gil.python();
        match self.series.dtype() {
            DataType::Float32 | DataType::Float64 => self.series.min::<f64>().to_object(py),
            DataType::Boolean => self.series.min::<u32>().map(|v| v == 1).to_object(py),
            _ => self.series.min::<i64>().to_object(py),
        }
    }

    pub fn sum(&self) -> PyObject {
        let gil = Python::acquire_gil();
        let py = gil.python();
        match self.series.dtype() {
            DataType::Float32 | DataType::Float64 => self.series.sum::<f64>().to_object(py),
            DataType::Boolean => self.series.sum::<u64>().to_object(py),
            _ => self.series.sum::<i64>().to_object(py),
        }
    }

    pub fn n_chunks(&self) -> usize {
        self.series.n_chunks()
    }

    pub fn limit(&self, num_elements: usize) -> Self {
        let series = self.series.limit(num_elements);
        series.into()
    }

    pub fn slice(&self, offset: i64, length: usize) -> Self {
        let series = self.series.slice(offset, length);
        series.into()
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

    pub fn add(&self, other: &PySeries) -> Self {
        (&self.series + &other.series).into()
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

    pub fn head(&self, length: Option<usize>) -> Self {
        (self.series.head(length)).into()
    }

    pub fn tail(&self, length: Option<usize>) -> Self {
        (self.series.tail(length)).into()
    }

    pub fn sort(&mut self, reverse: bool) -> Self {
        PySeries::new(self.series.sort(reverse))
    }

    pub fn argsort(&self, reverse: bool) -> Self {
        self.series.argsort(reverse).into_series().into()
    }

    pub fn unique(&self) -> PyResult<Self> {
        let unique = self.series.unique().map_err(PyPolarsEr::from)?;
        Ok(unique.into())
    }

    pub fn value_counts(&self) -> PyResult<PyDataFrame> {
        let df = self.series.value_counts().map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn arg_unique(&self) -> PyResult<Self> {
        let arg_unique = self.series.arg_unique().map_err(PyPolarsEr::from)?;
        Ok(arg_unique.into_series().into())
    }

    pub fn arg_min(&self) -> Option<usize> {
        self.series.arg_min()
    }

    pub fn arg_max(&self) -> Option<usize> {
        self.series.arg_max()
    }

    pub fn take(&self, indices: Wrap<AlignedVec<u32>>) -> PyResult<Self> {
        let indices = indices.0;
        let indices = UInt32Chunked::new_from_aligned_vec("", indices);

        let take = self.series.take(&indices).map_err(PyPolarsEr::from)?;
        Ok(PySeries::new(take))
    }

    pub fn take_with_series(&self, indices: &PySeries) -> PyResult<Self> {
        let idx = indices.series.u32().map_err(PyPolarsEr::from)?;
        let take = self.series.take(idx).map_err(PyPolarsEr::from)?;
        Ok(PySeries::new(take))
    }

    pub fn null_count(&self) -> PyResult<usize> {
        Ok(self.series.null_count())
    }

    pub fn has_validity(&self) -> bool {
        self.series.has_validity()
    }

    pub fn is_null(&self) -> PySeries {
        Self::new(self.series.is_null().into_series())
    }

    pub fn is_not_null(&self) -> PySeries {
        Self::new(self.series.is_not_null().into_series())
    }

    pub fn is_not_nan(&self) -> PyResult<Self> {
        let ca = self.series.is_not_nan().map_err(PyPolarsEr::from)?;
        Ok(ca.into_series().into())
    }

    pub fn is_nan(&self) -> PyResult<Self> {
        let ca = self.series.is_nan().map_err(PyPolarsEr::from)?;
        Ok(ca.into_series().into())
    }

    pub fn is_finite(&self) -> PyResult<Self> {
        let ca = self.series.is_finite().map_err(PyPolarsEr::from)?;
        Ok(ca.into_series().into())
    }

    pub fn is_infinite(&self) -> PyResult<Self> {
        let ca = self.series.is_infinite().map_err(PyPolarsEr::from)?;
        Ok(ca.into_series().into())
    }

    pub fn is_unique(&self) -> PyResult<Self> {
        let ca = self.series.is_unique().map_err(PyPolarsEr::from)?;
        Ok(ca.into_series().into())
    }

    pub fn arg_true(&self) -> PyResult<Self> {
        let ca = self.series.arg_true().map_err(PyPolarsEr::from)?;
        Ok(ca.into_series().into())
    }

    pub fn sample_n(&self, n: usize, with_replacement: bool, seed: u64) -> PyResult<Self> {
        let s = self
            .series
            .sample_n(n, with_replacement, seed)
            .map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn sample_frac(&self, frac: f64, with_replacement: bool, seed: u64) -> PyResult<Self> {
        let s = self
            .series
            .sample_frac(frac, with_replacement, seed)
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

    pub fn take_every(&self, n: usize) -> Self {
        let s = self.series.take_every(n);
        s.into()
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
        Ok(Self::new(self.series.equal(&rhs.series).into_series()))
    }

    pub fn neq(&self, rhs: &PySeries) -> PyResult<Self> {
        Ok(Self::new(self.series.not_equal(&rhs.series).into_series()))
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

    pub fn to_physical(&self) -> Self {
        let s = self.series.to_physical_repr().into_owned();
        s.into()
    }

    pub fn to_list(&self) -> PyObject {
        let gil = Python::acquire_gil();
        let python = gil.python();

        let series = &self.series;

        let primitive_to_list = |dt: &DataType, series: &Series| match dt {
            DataType::Boolean => PyList::new(python, series.bool().unwrap()),
            DataType::Utf8 => PyList::new(python, series.utf8().unwrap()),
            DataType::UInt8 => PyList::new(python, series.u8().unwrap()),
            DataType::UInt16 => PyList::new(python, series.u16().unwrap()),
            DataType::UInt32 => PyList::new(python, series.u32().unwrap()),
            DataType::UInt64 => PyList::new(python, series.u64().unwrap()),
            DataType::Int8 => PyList::new(python, series.i8().unwrap()),
            DataType::Int16 => PyList::new(python, series.i16().unwrap()),
            DataType::Int32 => PyList::new(python, series.i32().unwrap()),
            DataType::Int64 => PyList::new(python, series.i64().unwrap()),
            DataType::Float32 => PyList::new(python, series.f32().unwrap()),
            DataType::Float64 => PyList::new(python, series.f64().unwrap()),
            dt => panic!("to_list() not implemented for {:?}", dt),
        };

        let pylist = match series.dtype() {
            DataType::Categorical => PyList::new(python, series.categorical().unwrap().iter_str()),
            DataType::Object(_) => {
                let v = PyList::empty(python);
                for i in 0..series.len() {
                    let obj: Option<&ObjectValue> = self.series.get_object(i).map(|any| any.into());
                    let val = obj.to_object(python);

                    v.append(val).unwrap();
                }
                v
            }
            DataType::List(inner_dtype) => {
                let v = PyList::empty(python);
                let ca = series.list().unwrap();
                for opt_s in ca.amortized_iter() {
                    match opt_s {
                        None => {
                            v.append(python.None()).unwrap();
                        }
                        Some(s) => {
                            let pylst = primitive_to_list(&inner_dtype, s.as_ref());
                            v.append(pylst).unwrap();
                        }
                    }
                }
                v
            }
            DataType::Date => {
                let ca = series.date().unwrap();
                return Wrap(ca).to_object(python);
            }
            DataType::Datetime => {
                let ca = series.datetime().unwrap();
                return Wrap(ca).to_object(python);
            }
            dt => primitive_to_list(dt, series),
        };
        pylist.to_object(python)
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

    pub fn quantile(&self, quantile: f64, interpolation: &str) -> PyObject {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let interpol = match interpolation {
            "nearest" => QuantileInterpolOptions::Nearest,
            "lower" => QuantileInterpolOptions::Lower,
            "higher" => QuantileInterpolOptions::Higher,
            "midpoint" => QuantileInterpolOptions::Midpoint,
            "linear" => QuantileInterpolOptions::Linear,
            _ => panic!("not supported"),
        };

        Wrap(
            self.series
                .quantile_as_series(quantile, interpol)
                .expect("invalid quantile")
                .get(0),
        )
        .into_py(py)
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

    pub fn fill_null(&self, strategy: &str) -> PyResult<Self> {
        let strat = parse_strategy(strategy);
        let series = self.series.fill_null(strat).map_err(PyPolarsEr::from)?;
        Ok(PySeries::new(series))
    }

    pub fn to_arrow(&mut self) -> PyResult<PyObject> {
        self.rechunk(true);
        let gil = Python::acquire_gil();
        let py = gil.python();
        let pyarrow = py.import("pyarrow")?;
        arrow_interop::to_py::to_py_array(self.series.chunks()[0].clone(), py, pyarrow)
    }

    #[cfg(feature = "is_in")]
    pub fn is_in(&self, other: &PySeries) -> PyResult<Self> {
        let out = self.series.is_in(&other.series).map_err(PyPolarsEr::from)?;
        Ok(out.into_series().into())
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
                Some(str_to_polarstype(str_repr))
            }
        };

        let out = match output_type {
            Some(DataType::Int8) => {
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
            Some(DataType::Int16) => {
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
            Some(DataType::Int32) => {
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
            Some(DataType::Int64) => {
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
            Some(DataType::UInt8) => {
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
            Some(DataType::UInt16) => {
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
            Some(DataType::UInt32) => {
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
            Some(DataType::UInt64) => {
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
            Some(DataType::Float32) => {
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
            Some(DataType::Float64) => {
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
            Some(DataType::Boolean) => {
                let ca: BooleanChunked = apply_method_all_arrow_series!(
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
                let ca: Int32Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_date().into_series()
            }
            Some(DataType::Datetime) => {
                let ca: Int64Chunked = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_primitive_out_type,
                    py,
                    lambda,
                    0,
                    None
                )?;
                ca.into_date().into_series()
            }
            Some(DataType::Utf8) => {
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
            Some(DataType::Object(_)) => {
                let ca: ObjectChunked<ObjectValue> = apply_method_all_arrow_series!(
                    series,
                    apply_lambda_with_object_out_type,
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

    pub fn shift(&self, periods: i64) -> Self {
        let s = self.series.shift(periods);
        PySeries::new(s)
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

    pub fn str_json_path_match(&self, pat: &str) -> PyResult<Self> {
        let ca = self.series.utf8().map_err(PyPolarsEr::from)?;
        let s = ca
            .json_path_match(pat)
            .map_err(PyPolarsEr::from)?
            .into_series();
        Ok(s.into())
    }

    pub fn str_extract(&self, pat: &str, group_index: usize) -> PyResult<Self> {
        let ca = self.series.utf8().map_err(PyPolarsEr::from)?;
        let s = ca
            .extract(pat, group_index)
            .map_err(PyPolarsEr::from)?
            .into_series();
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

    pub fn str_parse_date(&self, fmt: Option<&str>) -> PyResult<Self> {
        if let Ok(ca) = &self.series.utf8() {
            let ca = ca.as_date(fmt).map_err(PyPolarsEr::from)?;
            Ok(PySeries::new(ca.into_series()))
        } else {
            Err(PyPolarsEr::Other("cannot parse Date expected utf8 type".into()).into())
        }
    }

    pub fn str_parse_datetime(&self, fmt: Option<&str>) -> PyResult<Self> {
        if let Ok(ca) = &self.series.utf8() {
            let ca = ca.as_datetime(fmt).map_err(PyPolarsEr::from)?;
            Ok(ca.into_series().into())
        } else {
            Err(PyPolarsEr::Other("cannot parse datetime expected utf8 type".into()).into())
        }
    }

    pub fn str_slice(&self, start: i64, length: Option<u64>) -> PyResult<Self> {
        let ca = self.series.utf8().map_err(PyPolarsEr::from)?;
        let s = ca
            .str_slice(start, length)
            .map_err(PyPolarsEr::from)?
            .into_series();
        Ok(s.into())
    }

    pub fn strftime(&self, fmt: &str) -> PyResult<Self> {
        let s = self.series.strftime(fmt).map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn arr_lengths(&self) -> PyResult<Self> {
        let ca = self.series.list().map_err(PyPolarsEr::from)?;
        let s = ca.lst_lengths().into_series();
        Ok(PySeries::new(s))
    }

    pub fn timestamp(&self) -> PyResult<Self> {
        let ca = self.series.timestamp().map_err(PyPolarsEr::from)?;
        Ok(ca.into_series().into())
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
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> PyResult<Self> {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };

        let s = self.series.rolling_sum(options).map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn rolling_mean(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> PyResult<Self> {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };

        let s = self
            .series
            .rolling_mean(options)
            .map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn rolling_max(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> PyResult<Self> {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };

        let s = self.series.rolling_max(options).map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }
    pub fn rolling_min(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> PyResult<Self> {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };

        let s = self.series.rolling_min(options).map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn rolling_var(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> PyResult<Self> {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };

        let s = self.series.rolling_var(options).map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn rolling_std(
        &self,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> PyResult<Self> {
        let options = RollingOptions {
            window_size,
            weights,
            min_periods,
            center,
        };

        let s = self.series.rolling_std(options).map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn year(&self) -> PyResult<Self> {
        let s = self.series.year().map_err(PyPolarsEr::from)?;
        Ok(s.into_series().into())
    }

    pub fn month(&self) -> PyResult<Self> {
        let s = self.series.month().map_err(PyPolarsEr::from)?;
        Ok(s.into_series().into())
    }

    pub fn weekday(&self) -> PyResult<Self> {
        let s = self.series.weekday().map_err(PyPolarsEr::from)?;
        Ok(s.into_series().into())
    }

    pub fn week(&self) -> PyResult<Self> {
        let s = self.series.week().map_err(PyPolarsEr::from)?;
        Ok(s.into_series().into())
    }

    pub fn day(&self) -> PyResult<Self> {
        let s = self.series.day().map_err(PyPolarsEr::from)?;
        Ok(s.into_series().into())
    }

    pub fn ordinal_day(&self) -> PyResult<Self> {
        let s = self.series.ordinal_day().map_err(PyPolarsEr::from)?;
        Ok(s.into_series().into())
    }

    pub fn hour(&self) -> PyResult<Self> {
        let s = self.series.hour().map_err(PyPolarsEr::from)?;
        Ok(s.into_series().into())
    }

    pub fn minute(&self) -> PyResult<Self> {
        let s = self.series.minute().map_err(PyPolarsEr::from)?;
        Ok(s.into_series().into())
    }

    pub fn second(&self) -> PyResult<Self> {
        let s = self.series.second().map_err(PyPolarsEr::from)?;
        Ok(s.into_series().into())
    }

    pub fn nanosecond(&self) -> PyResult<Self> {
        let s = self.series.nanosecond().map_err(PyPolarsEr::from)?;
        Ok(s.into_series().into())
    }

    pub fn dt_epoch_seconds(&self) -> PyResult<Self> {
        let ms = self.series.timestamp().map_err(PyPolarsEr::from)?;
        Ok((ms / 1000).into_series().into())
    }

    pub fn peak_max(&self) -> Self {
        self.series.peak_max().into_series().into()
    }

    pub fn peak_min(&self) -> Self {
        self.series.peak_min().into_series().into()
    }

    pub fn n_unique(&self) -> PyResult<usize> {
        let n = self.series.n_unique().map_err(PyPolarsEr::from)?;
        Ok(n)
    }

    pub fn is_first(&self) -> PyResult<Self> {
        let out = self
            .series
            .is_first()
            .map_err(PyPolarsEr::from)?
            .into_series();
        Ok(out.into())
    }

    pub fn round(&self, decimals: u32) -> PyResult<Self> {
        let s = self.series.round(decimals).map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn floor(&self) -> PyResult<Self> {
        let s = self.series.floor().map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn shrink_to_fit(&mut self) {
        self.series.shrink_to_fit();
    }

    pub fn dot(&self, other: &PySeries) -> Option<f64> {
        self.series.dot(&other.series)
    }

    pub fn hash(&self, k0: u64, k1: u64, k2: u64, k3: u64) -> Self {
        let hb = ahash::RandomState::with_seeds(k0, k1, k2, k3);
        self.series.hash(hb).into_series().into()
    }

    pub fn reinterpret(&self, signed: bool) -> PyResult<Self> {
        let s = reinterpret(&self.series, signed).map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn mode(&self) -> PyResult<Self> {
        let s = self.series.mode().map_err(PyPolarsEr::from)?;
        Ok(s.into())
    }

    pub fn interpolate(&self) -> Self {
        let s = self.series.interpolate();
        s.into()
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &bincode::serialize(&self.series).unwrap()).to_object(py))
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.series = bincode::deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn rank(&self, method: &str, reverse: bool) -> PyResult<Self> {
        let method = str_to_rankmethod(method).unwrap();
        let options = RankOptions {
            method,
            descending: reverse,
        };
        Ok(self.series.rank(options).into())
    }

    pub fn diff(&self, n: usize, null_behavior: &str) -> PyResult<Self> {
        let null_behavior = str_to_null_behavior(null_behavior)?;
        Ok(self.series.diff(n, null_behavior).into())
    }

    pub fn skew(&self, bias: bool) -> PyResult<Option<f64>> {
        let out = self.series.skew(bias).map_err(PyPolarsEr::from)?;
        Ok(out)
    }

    pub fn kurtosis(&self, fisher: bool, bias: bool) -> PyResult<Option<f64>> {
        let out = self
            .series
            .kurtosis(fisher, bias)
            .map_err(PyPolarsEr::from)?;
        Ok(out)
    }

    pub fn cast(&self, dtype: &str, strict: bool) -> PyResult<Self> {
        let dtype = str_to_polarstype(dtype);
        let out = if strict {
            self.series.strict_cast(&dtype)
        } else {
            self.series.cast(&dtype)
        };
        let out = out.map_err(PyPolarsEr::from)?;
        Ok(out.into())
    }

    pub fn abs(&self) -> PyResult<Self> {
        let out = self.series.abs().map_err(PyPolarsEr::from)?;
        Ok(out.into())
    }

    pub fn reshape(&self, dims: Vec<i64>) -> PyResult<Self> {
        let out = self.series.reshape(&dims).map_err(PyPolarsEr::from)?;
        Ok(out.into())
    }
    pub fn shuffle(&self, seed: u64) -> Self {
        self.series.shuffle(seed).into()
    }
}

macro_rules! impl_ufuncs {
    ($name:ident, $type:ident, $unsafe_from_ptr_method:ident) => {
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
                let (out_array, av) =
                    unsafe { aligned_array::<<$type as PolarsNumericType>::Native>(py, size) };

                debug_assert_eq!(get_refcnt(out_array), 1);
                // inserting it in a tuple increase the reference count by 1.
                let args = PyTuple::new(py, &[out_array]);
                debug_assert_eq!(get_refcnt(out_array), 2);

                // whatever the result, we must take the leaked memory ownership back
                let s = match lambda.call1(args) {
                    Ok(_) => {
                        // if this assert fails, the lambda has taken a reference to the object, so we must panic
                        // args and the lambda return have a reference, making a total of 3
                        assert_eq!(get_refcnt(out_array), 3);

                        let validity = self.series.chunks()[0].validity().cloned();
                        let ca = ChunkedArray::<$type>::new_from_owned_with_null_bitmap(
                            self.name(),
                            av,
                            validity,
                        );
                        PySeries::new(ca.into_series())
                    }
                    Err(e) => {
                        // return error information
                        return Err(e);
                    }
                };

                Ok(s)
            }
        }
    };
}
impl_ufuncs!(apply_ufunc_f32, Float32Type, unsafe_from_ptr_f32);
impl_ufuncs!(apply_ufunc_f64, Float64Type, unsafe_from_ptr_f64);
impl_ufuncs!(apply_ufunc_u8, UInt8Type, unsafe_from_ptr_u8);
impl_ufuncs!(apply_ufunc_u16, UInt16Type, unsafe_from_ptr_u16);
impl_ufuncs!(apply_ufunc_u32, UInt32Type, unsafe_from_ptr_u32);
impl_ufuncs!(apply_ufunc_u64, UInt64Type, unsafe_from_ptr_u64);
impl_ufuncs!(apply_ufunc_i8, Int8Type, unsafe_from_ptr_i8);
impl_ufuncs!(apply_ufunc_i16, Int16Type, unsafe_from_ptr_i16);
impl_ufuncs!(apply_ufunc_i32, Int32Type, unsafe_from_ptr_i32);
impl_ufuncs!(apply_ufunc_i64, Int64Type, unsafe_from_ptr_i64);

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
impl_set_with_mask!(set_with_mask_bool, bool, bool, Boolean);

macro_rules! impl_set_at_idx {
    ($name:ident, $native:ty, $cast:ident, $variant:ident) => {
        fn $name(series: &Series, idx: &[u32], value: Option<$native>) -> Result<Series> {
            let ca = series.$cast()?;
            let new = ca.set_at_idx(idx.iter().map(|idx| *idx as usize), value)?;
            Ok(new.into_series())
        }

        #[pymethods]
        impl PySeries {
            pub fn $name(&self, idx: &PyArray1<u32>, value: Option<$native>) -> PyResult<Self> {
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
                Ok(PySeries::new(self.series.equal(rhs).into_series()))
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
                Ok(PySeries::new(self.series.not_equal(rhs).into_series()))
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
        #[allow(clippy::nonstandard_macro_braces)]
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
