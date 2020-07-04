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

macro_rules! init_method {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PSeries {
            #[new]
            fn $name(name: &str, val: Vec<$type>) -> PSeries {
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
    
}
