use polars::prelude::*;
use polars_core::frame::column::ScalarColumn;
use pyo3::prelude::*;

use super::PySeries;
use crate::utils::EnterPolarsExt;

impl PySeries {
    /// An arithmetic operation of a scalar column that does not materialise it: `op` runs on
    /// two rows of it (and on two rows of a full-length scalar right-hand side), and the result
    /// is the first row repeated. Two rows rather than one, and with the flag the materialised
    /// column has (sorted), so that the flags the operation sets on its result are the Series
    /// path's.
    ///
    /// `None` (use the Series path) for a Series or Dictionary column, no rows, or any error.
    fn scalar_arithmetic(
        &self,
        py: Python<'_>,
        full: Option<ScalarColumn>,
        op: impl Fn(&Series, Option<&Series>) -> PolarsResult<Series> + Send + Sync,
    ) -> PyResult<Option<Self>> {
        let Some(Column::Scalar(sc)) = self.compressed().filter(|c| !c.is_empty()) else {
            return Ok(None);
        };
        let out = py.enter_polars_ok(|| {
            let k = sc.len().min(2);
            let rhs = full.as_ref().map(|f| f.as_n_values_series(k));
            let lhs = sc.as_n_values_series(k);
            let few = op(&lhs, rhs.as_ref()).ok()?;
            if few.len() != k {
                return None;
            }
            if k == sc.len() {
                return Some(few);
            }
            let mut out = few.new_from_index(0, sc.len());
            out.set_flags(few.get_flags());
            Some(out)
        })?;
        Ok(out.map(PySeries::new))
    }

    fn arithmetic_with_series(
        &self,
        py: Python<'_>,
        other: &PySeries,
        op: impl Fn(&Series, &Series) -> PolarsResult<Series> + Send + Sync,
    ) -> PyResult<Self> {
        if let Some((one, full)) = self.broadcast_operand(other) {
            if let Some(out) = self.scalar_arithmetic(py, full, |l, r| op(l, r.unwrap_or(&one)))? {
                return Ok(out);
            }
        }
        py.enter_polars_series(|| op(&self.series(), &other.series()))
    }

    fn arithmetic_with_const(
        &self,
        py: Python<'_>,
        op: impl Fn(&Series) -> Series + Send + Sync,
    ) -> PyResult<Self> {
        if let Some(out) = self.scalar_arithmetic(py, None, |l, _| Ok(op(l)))? {
            return Ok(out);
        }
        py.enter_polars_series(|| Ok(op(&self.series())))
    }
}

#[pymethods]
impl PySeries {
    fn add(&self, py: Python<'_>, other: &PySeries) -> PyResult<Self> {
        self.arithmetic_with_series(py, other, |l, r| l + r)
    }
    fn sub(&self, py: Python<'_>, other: &PySeries) -> PyResult<Self> {
        self.arithmetic_with_series(py, other, |l, r| l - r)
    }
    fn mul(&self, py: Python<'_>, other: &PySeries) -> PyResult<Self> {
        self.arithmetic_with_series(py, other, |l, r| l * r)
    }
    fn div(&self, py: Python<'_>, other: &PySeries) -> PyResult<Self> {
        self.arithmetic_with_series(py, other, |l, r| l / r)
    }
    fn rem(&self, py: Python<'_>, other: &PySeries) -> PyResult<Self> {
        self.arithmetic_with_series(py, other, |l, r| l % r)
    }
}

macro_rules! impl_arithmetic {
    ($name:ident, $type:ty, $operand:tt) => {
        #[pymethods]
        impl PySeries {
            fn $name(&self, py: Python<'_>, other: $type) -> PyResult<Self> {
                self.arithmetic_with_const(py, |l| l $operand other)
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
impl_arithmetic!(add_f16, pf16, +);
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
impl_arithmetic!(sub_f16, pf16, -);
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
impl_arithmetic!(div_f16, pf16, /);
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
impl_arithmetic!(mul_f16, pf16, *);
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
impl_arithmetic!(rem_f16, pf16, %);
impl_arithmetic!(rem_f32, f32, %);
impl_arithmetic!(rem_f64, f64, %);

macro_rules! impl_rhs_arithmetic {
    ($name:ident, $type:ty, $operand:ident) => {
        #[pymethods]
        impl PySeries {
            fn $name(&self, py: Python<'_>, other: $type) -> PyResult<Self> {
                self.arithmetic_with_const(py, |l| other.$operand(l))
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
impl_rhs_arithmetic!(add_f16_rhs, pf16, add);
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
impl_rhs_arithmetic!(sub_f16_rhs, pf16, sub);
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
impl_rhs_arithmetic!(div_f16_rhs, pf16, div);
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
impl_rhs_arithmetic!(mul_f16_rhs, pf16, mul);
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
impl_rhs_arithmetic!(rem_f16_rhs, pf16, rem);
impl_rhs_arithmetic!(rem_f32_rhs, f32, rem);
impl_rhs_arithmetic!(rem_f64_rhs, f64, rem);
