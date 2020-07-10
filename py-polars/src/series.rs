use polars::prelude::*;
use pyo3::exceptions::RuntimeError;
use pyo3::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PyPolarsEr {
    #[error(transparent)]
    Any(#[from] PolarsError),
    #[error("{0}")]
    Other(String),
}

impl std::convert::From<PyPolarsEr> for PyErr {
    fn from(err: PyPolarsEr) -> PyErr {
        RuntimeError::py_err(format!("{:?}", err))
    }
}

#[pyclass]
pub struct PSeries {
    pub series: Series,
}

impl PSeries {
    fn new(series: Series) -> Self {
        PSeries { series }
    }
}

macro_rules! init_method {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PSeries {
            #[new]
            pub fn $name(name: &str, val: Vec<$type>) -> PSeries {
                PSeries {
                    series: Series::new(name, &val),
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
init_method!(new_str, &str);

#[pymethods]
impl PSeries {
    pub fn name(&self) -> &str {
        self.series.name()
    }

    pub fn rename(&mut self, name: &str) {
        self.series.rename(name)
    }

    pub fn dtype(&self) -> &str {
        self.series.dtype().to_str()
    }

    pub fn n_chunks(&self) -> usize {
        self.series.n_chunks()
    }

    pub fn limit(&self, num_elements: usize) -> PyResult<Self> {
        let series = self.series.limit(num_elements).map_err(PyPolarsEr::from)?;
        Ok(PSeries { series })
    }

    pub fn slice(&self, offset: usize, length: usize) -> PyResult<Self> {
        let series = self
            .series
            .slice(offset, length)
            .map_err(PyPolarsEr::from)?;
        Ok(PSeries { series })
    }

    pub fn append(&mut self, other: &PSeries) -> PyResult<()> {
        self.series
            .append(&other.series)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn filter(&self, filter: &PSeries) -> PyResult<Self> {
        let filter_series = &filter.series;
        if let Series::Bool(ca) = filter_series {
            let series = self.series.filter(ca).map_err(PyPolarsEr::from)?;
            Ok(PSeries { series })
        } else {
            Err(RuntimeError::py_err("Expected a boolean mask"))
        }
    }

    pub fn add(&self, other: &PSeries) -> PyResult<PSeries> {
        Ok(PSeries::new(&self.series + &other.series))
    }

    pub fn sub(&self, other: &PSeries) -> PyResult<PSeries> {
        Ok(PSeries::new(&self.series - &other.series))
    }

    pub fn mul(&self, other: &PSeries) -> PyResult<PSeries> {
        Ok(PSeries::new(&self.series * &other.series))
    }

    pub fn div(&self, other: &PSeries) -> PyResult<PSeries> {
        Ok(PSeries::new(&self.series / &other.series))
    }

    pub fn head(&self, length: Option<usize>) -> PyResult<PSeries> {
        Ok(PSeries::new(self.series.head(length)))
    }

    pub fn tail(&self, length: Option<usize>) -> PyResult<PSeries> {
        Ok(PSeries::new(self.series.tail(length)))
    }

    pub fn sort(&mut self) {
        self.series.sort();
    }
}

macro_rules! impl_arithmetic {
    ($name:ident, $type:ty, $operand:tt) => {
        #[pymethods]
        impl PSeries {
            pub fn $name(&self, other: $type) -> PyResult<PSeries> {
                Ok(PSeries::new(&self.series $operand other))
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
        impl PSeries {
            pub fn $name(&self, other: $type) -> PyResult<PSeries> {
                Ok(PSeries::new(other.$operand(&self.series)))
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
        impl PSeries {
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
        impl PSeries {
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

macro_rules! impl_mean {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PSeries {
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
