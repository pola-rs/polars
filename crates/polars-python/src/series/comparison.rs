use polars_compute::decimal::{DEC128_MAX_PREC, dec128_fits};
use polars_core::frame::column::ScalarColumn;
use polars_core::series::IsSorted;
use pyo3::prelude::*;

use crate::PySeries;
use crate::error::PyPolarsErr;
use crate::prelude::*;
use crate::utils::EnterPolarsExt;

/// `ca` cut into chunks of `lengths` (the chunking the Series path keeps for a Dictionary
/// column's rows).
fn chunked_like(ca: BooleanChunked, lengths: &[usize]) -> BooleanChunked {
    if lengths.len() <= 1 || ca.chunks().len() != 1 {
        return ca;
    }
    let mut offset = 0i64;
    let mut chunks = Vec::with_capacity(lengths.len());
    for len in lengths {
        let part = ca.slice(Slice::new(offset, Some(*len)).unwrap());
        chunks.extend(part.chunks().iter().cloned());
        offset += *len as i64;
    }
    // SAFETY: the chunks are boolean arrays of `ca`.
    unsafe { BooleanChunked::from_chunks_and_dtype(ca.name().clone(), chunks, DataType::Boolean) }
}

impl PySeries {
    /// A comparison of a Scalar or Dictionary column that does not materialise it: `op` runs
    /// on the one value of a scalar column (and is broadcast), or on the dictionary values the
    /// rows use. The result is a plain Series, as the Series path's: a scalar column's flags
    /// (it reports itself sorted) would show on a scalar result.
    ///
    /// `None` (use the Series path) for a plain column, no rows, or any error, so that errors
    /// and their messages are the Series path's.
    fn compare_compressed(
        &self,
        py: Python<'_>,
        full: Option<ScalarColumn>,
        one_chunk: bool,
        op: impl Fn(&Series, Option<&Series>) -> PolarsResult<BooleanChunked> + Send + Sync,
    ) -> PyResult<Option<Self>> {
        let Some(c) = self.compressed().filter(|c| !c.is_empty()) else {
            return Ok(None);
        };
        let out = py.enter_polars_ok(|| match &c {
            Column::Scalar(sc) => {
                // The comparison runs on two rows: the flags it sets on its result depend on
                // the operands being sorted (they are, the rows are equal), and on the
                // right-hand side being a full-length column, not on the values.
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
            },
            Column::Dictionary(d) => {
                // A full-length column of a non-String scalar compares on the Series path.
                if full.as_ref().is_some_and(|f| f.dtype() != &DataType::String) {
                    return None;
                }
                if d.is_sorted_flag() != IsSorted::Not {
                    return None;
                }
                // A null right-hand side gives one chunk (all null) on the Series path
                // too.
                let lengths: Vec<usize> = if !one_chunk {
                    d.codes().chunk_lengths().collect()
                } else {
                    vec![d.len()]
                };
                let ca = d.compare_with(|v| op(v, None)).ok()?;
                (ca.len() == d.len()).then(|| chunked_like(ca, &lengths))
            },
            Column::Series(_) => None,
        })?;
        Ok(out.map(|ca| PySeries::new(ca.into_series())))
    }

    /// The right-hand side of a comparison as one value that broadcasts over this column's
    /// rows, when it is one: a length-1 Series, or a scalar column as long as this one (the
    /// second item then).
    pub(super) fn broadcast_operand(
        &self,
        rhs: &PySeries,
    ) -> Option<(Series, Option<ScalarColumn>)> {
        let n = self.column.read().len();
        let rhs = rhs.column.read().clone();
        match &rhs {
            Column::Series(_) if rhs.len() == 1 => {
                Some((rhs.as_materialized_series().clone(), None))
            },
            Column::Scalar(sc) if sc.len() == 1 => Some((sc.as_single_value_series(), None)),
            Column::Scalar(sc) if sc.len() == n => {
                Some((sc.as_single_value_series(), Some(sc.clone())))
            },
            _ => None,
        }
    }

    fn compare_with_series(
        &self,
        py: Python<'_>,
        rhs: &PySeries,
        op: impl Fn(&Series, &Series) -> PolarsResult<BooleanChunked> + Send + Sync,
    ) -> PyResult<Self> {
        if let Some((one, full)) = self.broadcast_operand(rhs) {
            let one_chunk = full.is_none() && one.null_count() > 0;
            if let Some(out) =
                self.compare_compressed(py, full, one_chunk, |l, r| op(l, r.unwrap_or(&one)))?
            {
                return Ok(out);
            }
        }
        py.enter_polars_series(|| op(&self.series(), &rhs.series()))
    }

    fn compare_with_const(
        &self,
        py: Python<'_>,
        op: impl Fn(&Series) -> PolarsResult<BooleanChunked> + Send + Sync,
    ) -> PyResult<Self> {
        if let Some(out) = self.compare_compressed(py, None, false, |l, _| op(l))? {
            return Ok(out);
        }
        py.enter_polars_series(|| op(&self.series()))
    }
}

#[pymethods]
impl PySeries {
    fn eq(&self, py: Python<'_>, rhs: &PySeries) -> PyResult<Self> {
        self.compare_with_series(py, rhs, |l, r| l.equal(r))
    }

    fn neq(&self, py: Python<'_>, rhs: &PySeries) -> PyResult<Self> {
        self.compare_with_series(py, rhs, |l, r| l.not_equal(r))
    }

    fn gt(&self, py: Python<'_>, rhs: &PySeries) -> PyResult<Self> {
        self.compare_with_series(py, rhs, |l, r| l.gt(r))
    }

    fn gt_eq(&self, py: Python<'_>, rhs: &PySeries) -> PyResult<Self> {
        self.compare_with_series(py, rhs, |l, r| l.gt_eq(r))
    }

    fn lt(&self, py: Python<'_>, rhs: &PySeries) -> PyResult<Self> {
        self.compare_with_series(py, rhs, |l, r| l.lt(r))
    }

    fn lt_eq(&self, py: Python<'_>, rhs: &PySeries) -> PyResult<Self> {
        self.compare_with_series(py, rhs, |l, r| l.lt_eq(r))
    }
}

macro_rules! impl_op {
    ($op:ident, $name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            fn $name(&self, py: Python<'_>, rhs: $type) -> PyResult<Self> {
                self.compare_with_const(py, |s| s.$op(rhs))
            }
        }
    };
}

impl_op!(equal, eq_u8, u8);
impl_op!(equal, eq_u16, u16);
impl_op!(equal, eq_u32, u32);
impl_op!(equal, eq_u64, u64);
impl_op!(equal, eq_u128, u128);
impl_op!(equal, eq_i8, i8);
impl_op!(equal, eq_i16, i16);
impl_op!(equal, eq_i32, i32);
impl_op!(equal, eq_i64, i64);
impl_op!(equal, eq_i128, i128);
impl_op!(equal, eq_f16, pf16);
impl_op!(equal, eq_f32, f32);
impl_op!(equal, eq_f64, f64);
impl_op!(equal, eq_str, &str);

impl_op!(not_equal, neq_u8, u8);
impl_op!(not_equal, neq_u16, u16);
impl_op!(not_equal, neq_u32, u32);
impl_op!(not_equal, neq_u64, u64);
impl_op!(not_equal, neq_u128, u128);
impl_op!(not_equal, neq_i8, i8);
impl_op!(not_equal, neq_i16, i16);
impl_op!(not_equal, neq_i32, i32);
impl_op!(not_equal, neq_i64, i64);
impl_op!(not_equal, neq_i128, i128);
impl_op!(not_equal, neq_f16, pf16);
impl_op!(not_equal, neq_f32, f32);
impl_op!(not_equal, neq_f64, f64);
impl_op!(not_equal, neq_str, &str);

impl_op!(gt, gt_u8, u8);
impl_op!(gt, gt_u16, u16);
impl_op!(gt, gt_u32, u32);
impl_op!(gt, gt_u64, u64);
impl_op!(gt, gt_u128, u128);
impl_op!(gt, gt_i8, i8);
impl_op!(gt, gt_i16, i16);
impl_op!(gt, gt_i32, i32);
impl_op!(gt, gt_i64, i64);
impl_op!(gt, gt_i128, i128);
impl_op!(gt, gt_f16, pf16);
impl_op!(gt, gt_f32, f32);
impl_op!(gt, gt_f64, f64);
impl_op!(gt, gt_str, &str);

impl_op!(gt_eq, gt_eq_u8, u8);
impl_op!(gt_eq, gt_eq_u16, u16);
impl_op!(gt_eq, gt_eq_u32, u32);
impl_op!(gt_eq, gt_eq_u64, u64);
impl_op!(gt_eq, gt_eq_u128, u128);
impl_op!(gt_eq, gt_eq_i8, i8);
impl_op!(gt_eq, gt_eq_i16, i16);
impl_op!(gt_eq, gt_eq_i32, i32);
impl_op!(gt_eq, gt_eq_i64, i64);
impl_op!(gt_eq, gt_eq_i128, i128);
impl_op!(gt_eq, gt_eq_f16, pf16);
impl_op!(gt_eq, gt_eq_f32, f32);
impl_op!(gt_eq, gt_eq_f64, f64);
impl_op!(gt_eq, gt_eq_str, &str);

impl_op!(lt, lt_u8, u8);
impl_op!(lt, lt_u16, u16);
impl_op!(lt, lt_u32, u32);
impl_op!(lt, lt_u64, u64);
impl_op!(lt, lt_u128, u128);
impl_op!(lt, lt_i8, i8);
impl_op!(lt, lt_i16, i16);
impl_op!(lt, lt_i32, i32);
impl_op!(lt, lt_i64, i64);
impl_op!(lt, lt_i128, i128);
impl_op!(lt, lt_f16, pf16);
impl_op!(lt, lt_f32, f32);
impl_op!(lt, lt_f64, f64);
impl_op!(lt, lt_str, &str);

impl_op!(lt_eq, lt_eq_u8, u8);
impl_op!(lt_eq, lt_eq_u16, u16);
impl_op!(lt_eq, lt_eq_u32, u32);
impl_op!(lt_eq, lt_eq_u64, u64);
impl_op!(lt_eq, lt_eq_u128, u128);
impl_op!(lt_eq, lt_eq_i8, i8);
impl_op!(lt_eq, lt_eq_i16, i16);
impl_op!(lt_eq, lt_eq_i32, i32);
impl_op!(lt_eq, lt_eq_i64, i64);
impl_op!(lt_eq, lt_eq_i128, i128);
impl_op!(lt_eq, lt_eq_f16, pf16);
impl_op!(lt_eq, lt_eq_f32, f32);
impl_op!(lt_eq, lt_eq_f64, f64);
impl_op!(lt_eq, lt_eq_str, &str);

struct PyDecimal(i128, usize, usize);

impl<'a, 'py> FromPyObject<'a, 'py> for PyDecimal {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(val) = obj.extract() {
            return Ok(PyDecimal(val, DEC128_MAX_PREC, 0));
        }

        let err = || {
            Err(PyPolarsErr::from(polars_err!(ComputeError: "overflow in Python Decimal to Polars Decimal conversion")).into())
        };

        let (sign, digits, exponent) = obj
            .call_method0("as_tuple")?
            .extract::<(i8, Vec<u8>, i8)>()?;
        let mut val = 0_i128;
        for d in digits {
            if let Some(v) = val.checked_mul(10).and_then(|val| val.checked_add(d as _)) {
                val = v;
            } else {
                return err();
            }
        }
        let scale = if exponent > 0 {
            if let Some(v) = val.checked_mul(10_i128.pow((-exponent) as u32)) {
                val = v;
            } else {
                return err();
            };
            0_usize
        } else {
            (-exponent) as usize
        };
        if sign == 1 {
            val = -val
        };
        if dec128_fits(val, DEC128_MAX_PREC) {
            Ok(PyDecimal(val, DEC128_MAX_PREC, scale))
        } else {
            err()
        }
    }
}

macro_rules! impl_decimal {
    ($name:ident, $method:ident) => {
        #[pymethods]
        impl PySeries {
            fn $name(&self, py: Python<'_>, rhs: PyDecimal) -> PyResult<Self> {
                let rhs = Series::new(
                    PlSmallStr::from_static("decimal"),
                    &[AnyValue::Decimal(rhs.0, rhs.1, rhs.2)],
                );
                self.compare_with_const(py, |s| s.$method(&rhs))
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
