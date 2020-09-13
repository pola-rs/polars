use crate::error::PyPolarsEr;
use numpy::PyArray1;
use polars::prelude::*;
use pyo3::types::PyList;
use pyo3::{exceptions::RuntimeError, prelude::*};

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

init_method!(new_i32, i32);
init_method!(new_i64, i64);
init_method!(new_f32, f32);
init_method!(new_f64, f64);
init_method!(new_bool, bool);
init_method!(new_u32, u32);
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
            pub fn $name(name: &str, val: Vec<Option<$type>>) -> PySeries {
                PySeries {
                    series: Series::new(name, &val),
                }
            }
        }
    };
}

init_method_opt!(new_opt_i32, i32);
init_method_opt!(new_opt_i64, i64);
init_method_opt!(new_opt_f32, f32);
init_method_opt!(new_opt_f64, f64);
init_method_opt!(new_opt_bool, bool);
init_method_opt!(new_opt_u32, u32);
init_method_opt!(new_opt_date32, i32);
init_method_opt!(new_opt_date64, i64);
init_method_opt!(new_opt_duration_ns, i64);
init_method_opt!(new_opt_time_ns, i64);

#[pymethods]
impl PySeries {
    #[staticmethod]
    pub fn new_str(name: &str, val: Vec<&str>) -> Self {
        PySeries::new(Series::new(name, &val))
    }

    #[staticmethod]
    pub fn new_opt_str(name: &str, val: Vec<Option<&str>>) -> Self {
        PySeries::new(Series::new(name, &val))
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

    pub fn name(&self) -> &str {
        self.series.name()
    }

    pub fn rename(&mut self, name: &str) {
        self.series.rename(name);
    }

    pub fn dtype(&self) -> String {
        self.series.dtype().to_str()
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
            Err(RuntimeError::py_err("Expected a boolean mask"))
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

    pub fn arg_unique(&self) -> Py<PyArray1<usize>> {
        let gil = pyo3::Python::acquire_gil();
        let pyarray = PyArray1::from_vec(gil.python(), self.series.arg_unique());
        pyarray.to_owned()
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

    pub fn series_equal(&self, other: &PySeries) -> PyResult<bool> {
        Ok(self.series.series_equal(&other.series))
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

    pub fn as_str(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.series))
    }

    pub fn len(&self) -> usize {
        self.series.len()
    }

    pub fn to_list(&self) -> PyObject {
        let gil = pyo3::Python::acquire_gil();
        let python = gil.python();

        let pylist = match &self.series {
            Series::UInt32(ca) => PyList::new(python, ca),
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
}

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

impl_cast!(cast_u32, UInt32Type);
impl_cast!(cast_i32, Int32Type);
impl_cast!(cast_i64, Int64Type);
impl_cast!(cast_f32, Float32Type);
impl_cast!(cast_f64, Float64Type);
impl_cast!(cast_date32, Date32Type);
impl_cast!(cast_date64, Date64Type);
impl_cast!(cast_time64ns, Time64NanosecondType);
impl_cast!(cast_duration_ns, DurationNanosecondType);

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

impl_arithmetic!(add_u32, u32, +);
impl_arithmetic!(add_i32, i32, +);
impl_arithmetic!(add_i64, i64, +);
impl_arithmetic!(add_f32, f32, +);
impl_arithmetic!(add_f64, f64, +);
impl_arithmetic!(sub_u32, u32, -);
impl_arithmetic!(sub_i32, i32, -);
impl_arithmetic!(sub_i64, i64, -);
impl_arithmetic!(sub_f32, f32, -);
impl_arithmetic!(sub_f64, f64, -);
impl_arithmetic!(div_u32, u32, /);
impl_arithmetic!(div_i32, i32, /);
impl_arithmetic!(div_i64, i64, /);
impl_arithmetic!(div_f32, f32, /);
impl_arithmetic!(div_f64, f64, /);
impl_arithmetic!(mul_u32, u32, *);
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

impl_rhs_arithmetic!(add_u32_rhs, u32, add);
impl_rhs_arithmetic!(add_i32_rhs, i32, add);
impl_rhs_arithmetic!(add_i64_rhs, i64, add);
impl_rhs_arithmetic!(add_f32_rhs, f32, add);
impl_rhs_arithmetic!(add_f64_rhs, f64, add);
impl_rhs_arithmetic!(sub_u32_rhs, u32, sub);
impl_rhs_arithmetic!(sub_i32_rhs, i32, sub);
impl_rhs_arithmetic!(sub_i64_rhs, i64, sub);
impl_rhs_arithmetic!(sub_f32_rhs, f32, sub);
impl_rhs_arithmetic!(sub_f64_rhs, f64, sub);
impl_rhs_arithmetic!(div_u32_rhs, u32, div);
impl_rhs_arithmetic!(div_i32_rhs, i32, div);
impl_rhs_arithmetic!(div_i64_rhs, i64, div);
impl_rhs_arithmetic!(div_f32_rhs, f32, div);
impl_rhs_arithmetic!(div_f64_rhs, f64, div);
impl_rhs_arithmetic!(mul_u32_rhs, u32, mul);
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

impl_sum!(sum_u32, u32);
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

impl_min!(min_u32, u32);
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

impl_max!(max_u32, u32);
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

impl_mean!(mean_u32, u32);
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

impl_eq_num!(eq_u32, u32);
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

impl_neq_num!(neq_u32, u32);
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

impl_gt_num!(gt_u32, u32);
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

impl_gt_eq_num!(gt_eq_u32, u32);
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

impl_lt_num!(lt_u32, u32);
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

impl_lt_eq_num!(lt_eq_u32, u32);
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
