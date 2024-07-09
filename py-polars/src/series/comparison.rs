use pyo3::prelude::*;

use crate::error::PyPolarsErr;
use crate::prelude::*;
use crate::PySeries;

#[pymethods]
impl PySeries {
    fn eq(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.equal(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(s.into_series().into())
    }

    fn neq(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self
            .series
            .not_equal(&rhs.series)
            .map_err(PyPolarsErr::from)?;
        Ok(s.into_series().into())
    }

    fn gt(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.gt(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(s.into_series().into())
    }

    fn gt_eq(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.gt_eq(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(s.into_series().into())
    }

    fn lt(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.lt(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(s.into_series().into())
    }

    fn lt_eq(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.lt_eq(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(s.into_series().into())
    }
}

macro_rules! impl_eq_num {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            fn $name(&self, rhs: $type) -> PyResult<Self> {
                let s = self.series.equal(rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
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
            fn $name(&self, rhs: $type) -> PyResult<Self> {
                let s = self.series.not_equal(rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
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
            fn $name(&self, rhs: $type) -> PyResult<Self> {
                let s = self.series.gt(rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
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
            fn $name(&self, rhs: $type) -> PyResult<Self> {
                let s = self.series.gt_eq(rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
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
            fn $name(&self, rhs: $type) -> PyResult<Self> {
                let s = self.series.lt(rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
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
            fn $name(&self, rhs: $type) -> PyResult<Self> {
                let s = self.series.lt_eq(rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
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

struct PyDecimal(i128, usize);

impl<'source> FromPyObject<'source> for PyDecimal {
    fn extract_bound(obj: &Bound<'source, PyAny>) -> PyResult<Self> {
        if let Ok(val) = obj.extract() {
            return Ok(PyDecimal(val, 0));
        }

        let (sign, digits, exponent) = obj
            .call_method0("as_tuple")?
            .extract::<(i8, Vec<u8>, i8)>()?;
        let mut val = 0_i128;
        for d in digits {
            if let Some(v) = val.checked_mul(10).and_then(|val| val.checked_add(d as _)) {
                val = v;
            } else {
                return Err(PyPolarsErr::from(polars_err!(ComputeError: "overflow")).into());
            }
        }
        let exponent = if exponent > 0 {
            if let Some(v) = val.checked_mul(10_i128.pow((-exponent) as u32)) {
                val = v;
            } else {
                return Err(PyPolarsErr::from(polars_err!(ComputeError: "overflow")).into());
            };
            0_usize
        } else {
            -exponent as _
        };
        if sign == 1 {
            val = -val
        };
        Ok(PyDecimal(val, exponent))
    }
}

macro_rules! impl_decimal {
    ($name:ident, $method:ident) => {
        #[pymethods]
        impl PySeries {
            fn $name(&self, rhs: PyDecimal) -> PyResult<Self> {
                let rhs = Series::new("decimal", &[AnyValue::Decimal(rhs.0, rhs.1)]);
                let s = self.series.$method(&rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
            }
        }
    };
}

impl_decimal!(eq_decimal, equal);
impl_decimal!(neq_decimal, not_equal);
impl_decimal!(gt_decimal, gt);
impl_decimal!(gt_eq_decimal, gt_eq);
impl_decimal!(lt_decimal, lt);
impl_decimal!(lt_eq_decimal, lt_eq);
