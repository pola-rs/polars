use std::cell::Cell;

use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_compute::decimal::{
    DEC128_MAX_PREC, dec128_add_scaled, dec128_div_scaled, dec128_int_div_scaled,
    dec128_mul_scaled, dec128_rem_scaled, dec128_rescale, dec128_sub_scaled, i64_binary_values,
    i64_unary_values,
};

use super::*;
use crate::prelude::arity::{
    apply_binary_kernel_broadcast, broadcast_binary_elementwise_values,
    broadcast_try_binary_elementwise,
};

impl DecimalChunked {
    /// Applies `kernel(l, left_scale, r, right_scale, scale)` elementwise, producing a
    /// `Decimal(38, scale)`. The kernel returns `None` if the result doesn't fit, or on
    /// division by zero.
    fn apply_scaled_kernel(
        &self,
        rhs: &Self,
        scale: usize,
        op: &str,
        kernel: impl Fn(i128, usize, i128, usize, usize) -> Option<i128>,
    ) -> PolarsResult<Self> {
        let left_s = self.scale();
        let right_s = rhs.scale();
        let phys = broadcast_try_binary_elementwise(
            self.physical(),
            rhs.physical(),
            |opt_l, opt_r| {
                let (Some(l), Some(r)) = (opt_l, opt_r) else {
                    return PolarsResult::Ok(None);
                };
                let ret = kernel(l, left_s, r, right_s, scale).ok_or_else(|| {
                    // Only division and remainder can fail with a zero operand.
                    if r == 0 {
                        polars_err!(ComputeError: "division by zero Decimal")
                    } else {
                        polars_err!(
                            ComputeError: "overflow in decimal {op}: result doesn't fit Decimal({DEC128_MAX_PREC}, {scale})"
                        )
                    }
                })?;
                Ok(Some(ret))
            },
        )?;
        Ok(phys.into_decimal_unchecked(DEC128_MAX_PREC, scale))
    }

    /// [`Self::apply_scaled_kernel`] for kernels that only fail on overflow. It first
    /// computes all slots without per-row validity, and only goes row by row if a slot
    /// failed, as null slots can hold any value.
    fn apply_scaled_kernel_values(
        &self,
        rhs: &Self,
        scale: usize,
        op: &str,
        kernel: impl Fn(i128, usize, i128, usize, usize) -> Option<i128>,
    ) -> PolarsResult<Self> {
        let left_s = self.scale();
        let right_s = rhs.scale();
        let mut failed = false;
        let phys = broadcast_binary_elementwise_values(self.physical(), rhs.physical(), |l, r| {
            let ret = kernel(l, left_s, r, right_s, scale);
            failed |= ret.is_none();
            ret.unwrap_or(0)
        });
        if failed {
            return self.apply_scaled_kernel(rhs, scale, op, kernel);
        }
        Ok(phys.into_decimal_unchecked(DEC128_MAX_PREC, scale))
    }

    /// Applies `op` to all slots as i64s. Returns `None` if a slot, null ones included,
    /// doesn't fit an i64.
    fn apply_i64_values(
        &self,
        rhs: &Self,
        scale: usize,
        op: impl Fn(i64, i64) -> i128,
    ) -> Option<Self> {
        let failed = Cell::new(false);
        let to_arr = |values: Option<Vec<i128>>, len: usize, validity: Option<Bitmap>| {
            let Some(values) = values else {
                failed.set(true);
                return Int128Array::new_null(ArrowDataType::Int128, len);
            };
            Int128Array::from_vec(values).with_validity(validity)
        };
        let phys = apply_binary_kernel_broadcast(
            self.physical(),
            rhs.physical(),
            |l, r| {
                let values = i64_binary_values(l.values(), r.values(), &op);
                to_arr(
                    values,
                    l.len(),
                    combine_validities_and(l.validity(), r.validity()),
                )
            },
            |l, r| {
                let values = i64::try_from(l)
                    .ok()
                    .and_then(|l| i64_unary_values(r.values(), |r| op(l, r)));
                to_arr(values, r.len(), r.validity().cloned())
            },
            |l, r| {
                let values = i64::try_from(r)
                    .ok()
                    .and_then(|r| i64_unary_values(l.values(), |l| op(l, r)));
                to_arr(values, l.len(), l.validity().cloned())
            },
        );
        (!failed.get()).then(|| phys.into_decimal_unchecked(DEC128_MAX_PREC, scale))
    }

    /// A single non-null value at `scale`, if it has another scale and fits.
    fn scalar_with_scale(&self, scale: usize) -> Option<Self> {
        if self.len() != 1 || self.scale() == scale {
            return None;
        }
        let value = dec128_rescale(
            self.physical().get(0)?,
            self.scale(),
            DEC128_MAX_PREC,
            scale,
        )?;
        Some(
            Int128Chunked::from_slice(self.name().clone(), &[value])
                .into_decimal_unchecked(DEC128_MAX_PREC, scale),
        )
    }

    /// Applies an addition or subtraction kernel at the larger scale. A single
    /// value is brought to that scale once instead of in every row.
    fn add_sub(
        &self,
        rhs: &Self,
        op: &str,
        i64_op: impl Fn(i64, i64) -> i128,
        kernel: impl Fn(i128, usize, i128, usize, usize) -> Option<i128>,
    ) -> PolarsResult<Self> {
        let scale = self.scale().max(rhs.scale());
        let lhs_scalar = self.scalar_with_scale(scale);
        let rhs_scalar = rhs.scalar_with_scale(scale);
        let lhs = lhs_scalar.as_ref().unwrap_or(self);
        let rhs = rhs_scalar.as_ref().unwrap_or(rhs);
        if lhs.scale() == rhs.scale()
            && let Some(out) = lhs.apply_i64_values(rhs, scale, i64_op)
        {
            return Ok(out);
        }
        lhs.apply_scaled_kernel_values(rhs, scale, op, kernel)
    }

    /// Multiplies with the result rounded to `scale`.
    pub fn mul_with_scale(&self, rhs: &Self, scale: usize) -> PolarsResult<Self> {
        if self.scale() + rhs.scale() == scale
            && let Some(out) = self.apply_i64_values(rhs, scale, |l, r| l as i128 * r as i128)
        {
            return Ok(out);
        }
        self.apply_scaled_kernel_values(rhs, scale, "multiplication", dec128_mul_scaled)
    }

    /// Divides with the result rounded to `scale`.
    pub fn div_with_scale(&self, rhs: &Self, scale: usize) -> PolarsResult<Self> {
        self.apply_scaled_kernel(rhs, scale, "division", dec128_div_scaled)
    }

    /// The exact remainder, with the sign of `rhs` if `floor` (as `%`), else of `self`.
    pub fn rem_with(&self, rhs: &Self, floor: bool) -> PolarsResult<Self> {
        let scale = self.scale().max(rhs.scale());
        self.apply_scaled_kernel(rhs, scale, "remainder", |l, sl, r, sr, s| {
            dec128_rem_scaled(l, sl, r, sr, s, floor)
        })
    }

    /// The integer quotient, rounded down if `floor` (as `//`), else toward zero.
    pub fn int_div(&self, rhs: &Self, floor: bool) -> PolarsResult<Self> {
        self.int_div_with_scale(rhs, self.scale().max(rhs.scale()), floor)
    }

    /// [`Self::int_div`] with the quotient at `scale`.
    pub fn int_div_with_scale(&self, rhs: &Self, scale: usize, floor: bool) -> PolarsResult<Self> {
        self.apply_scaled_kernel(rhs, scale, "integer division", |l, sl, r, sr, s| {
            dec128_int_div_scaled(l, sl, r, sr, s, floor)
        })
    }
}

impl Add for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_sub(
            rhs,
            "addition",
            |l, r| l as i128 + r as i128,
            dec128_add_scaled,
        )
    }
}

impl Sub for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.add_sub(
            rhs,
            "subtraction",
            |l, r| l as i128 - r as i128,
            dec128_sub_scaled,
        )
    }
}

impl Mul for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_with_scale(rhs, self.scale().max(rhs.scale()))
    }
}

impl Div for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn div(self, rhs: Self) -> Self::Output {
        self.div_with_scale(rhs, self.scale().max(rhs.scale()))
    }
}
