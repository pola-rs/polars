use pyo3::prelude::*;

use crate::error::PyPolarsErr;
use crate::prelude::*;
use crate::utils::EnterPolarsExt;
use crate::PySeries;

#[pymethods]
impl PySeries {
    fn eq(&self, py: Python, rhs: &PySeries) -> PyResult<Self> {
        py.enter_polars_series(|| self.series.equal(&rhs.series))
    }

    fn neq(&self, py: Python, rhs: &PySeries) -> PyResult<Self> {
        py.enter_polars_series(|| self.series.not_equal(&rhs.series))
    }

    fn gt(&self, py: Python, rhs: &PySeries) -> PyResult<Self> {
        py.enter_polars_series(|| self.series.gt(&rhs.series))
    }

    fn gt_eq(&self, py: Python, rhs: &PySeries) -> PyResult<Self> {
        py.enter_polars_series(|| self.series.gt_eq(&rhs.series))
    }

    fn lt(&self, py: Python, rhs: &PySeries) -> PyResult<Self> {
        py.enter_polars_series(|| self.series.lt(&rhs.series))
    }

    fn lt_eq(&self, py: Python, rhs: &PySeries) -> PyResult<Self> {
        py.enter_polars_series(|| self.series.lt_eq(&rhs.series))
    }
}

macro_rules! impl_op {
    ($op:ident, $name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            fn $name(&self, py: Python, rhs: $type) -> PyResult<Self> {
                py.enter_polars_series(|| self.series.$op(rhs))
            }
        }
    };
}

impl_op!(equal, eq_u8, u8);
impl_op!(equal, eq_u16, u16);
impl_op!(equal, eq_u32, u32);
impl_op!(equal, eq_u64, u64);
impl_op!(equal, eq_i8, i8);
impl_op!(equal, eq_i16, i16);
impl_op!(equal, eq_i32, i32);
impl_op!(equal, eq_i64, i64);
impl_op!(equal, eq_i128, i128);
impl_op!(equal, eq_f32, f32);
impl_op!(equal, eq_f64, f64);
impl_op!(equal, eq_str, &str);

impl_op!(not_equal, neq_u8, u8);
impl_op!(not_equal, neq_u16, u16);
impl_op!(not_equal, neq_u32, u32);
impl_op!(not_equal, neq_u64, u64);
impl_op!(not_equal, neq_i8, i8);
impl_op!(not_equal, neq_i16, i16);
impl_op!(not_equal, neq_i32, i32);
impl_op!(not_equal, neq_i64, i64);
impl_op!(not_equal, neq_i128, i128);
impl_op!(not_equal, neq_f32, f32);
impl_op!(not_equal, neq_f64, f64);
impl_op!(not_equal, neq_str, &str);

impl_op!(gt, gt_u8, u8);
impl_op!(gt, gt_u16, u16);
impl_op!(gt, gt_u32, u32);
impl_op!(gt, gt_u64, u64);
impl_op!(gt, gt_i8, i8);
impl_op!(gt, gt_i16, i16);
impl_op!(gt, gt_i32, i32);
impl_op!(gt, gt_i64, i64);
impl_op!(gt, gt_i128, i128);
impl_op!(gt, gt_f32, f32);
impl_op!(gt, gt_f64, f64);
impl_op!(gt, gt_str, &str);

impl_op!(gt_eq, gt_eq_u8, u8);
impl_op!(gt_eq, gt_eq_u16, u16);
impl_op!(gt_eq, gt_eq_u32, u32);
impl_op!(gt_eq, gt_eq_u64, u64);
impl_op!(gt_eq, gt_eq_i8, i8);
impl_op!(gt_eq, gt_eq_i16, i16);
impl_op!(gt_eq, gt_eq_i32, i32);
impl_op!(gt_eq, gt_eq_i64, i64);
impl_op!(gt_eq, gt_eq_i128, i128);
impl_op!(gt_eq, gt_eq_f32, f32);
impl_op!(gt_eq, gt_eq_f64, f64);
impl_op!(gt_eq, gt_eq_str, &str);

impl_op!(lt, lt_u8, u8);
impl_op!(lt, lt_u16, u16);
impl_op!(lt, lt_u32, u32);
impl_op!(lt, lt_u64, u64);
impl_op!(lt, lt_i8, i8);
impl_op!(lt, lt_i16, i16);
impl_op!(lt, lt_i32, i32);
impl_op!(lt, lt_i64, i64);
impl_op!(lt, lt_i128, i128);
impl_op!(lt, lt_f32, f32);
impl_op!(lt, lt_f64, f64);
impl_op!(lt, lt_str, &str);

impl_op!(lt_eq, lt_eq_u8, u8);
impl_op!(lt_eq, lt_eq_u16, u16);
impl_op!(lt_eq, lt_eq_u32, u32);
impl_op!(lt_eq, lt_eq_u64, u64);
impl_op!(lt_eq, lt_eq_i8, i8);
impl_op!(lt_eq, lt_eq_i16, i16);
impl_op!(lt_eq, lt_eq_i32, i32);
impl_op!(lt_eq, lt_eq_i64, i64);
impl_op!(lt_eq, lt_eq_i128, i128);
impl_op!(lt_eq, lt_eq_f32, f32);
impl_op!(lt_eq, lt_eq_f64, f64);
impl_op!(lt_eq, lt_eq_str, &str);

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
            fn $name(&self, py: Python, rhs: PyDecimal) -> PyResult<Self> {
                let rhs = Series::new(
                    PlSmallStr::from_static("decimal"),
                    &[AnyValue::Decimal(rhs.0, rhs.1)],
                );
                py.enter_polars_series(|| self.series.$method(&rhs))
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
