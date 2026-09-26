use polars_compute::decimal::{
    DEC128_MAX_PREC, dec128_add_scaled, dec128_div_scaled, dec128_int_div_scaled,
    dec128_mul_scaled, dec128_rem_scaled, dec128_sub_scaled,
};

use super::*;
use crate::prelude::arity::broadcast_try_binary_elementwise;

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

    /// Multiplies with the result rounded to `scale`.
    pub fn mul_with_scale(&self, rhs: &Self, scale: usize) -> PolarsResult<Self> {
        self.apply_scaled_kernel(rhs, scale, "multiplication", dec128_mul_scaled)
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
        let scale = self.scale().max(rhs.scale());
        self.apply_scaled_kernel(rhs, scale, "addition", dec128_add_scaled)
    }
}

impl Sub for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn sub(self, rhs: Self) -> Self::Output {
        let scale = self.scale().max(rhs.scale());
        self.apply_scaled_kernel(rhs, scale, "subtraction", dec128_sub_scaled)
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
