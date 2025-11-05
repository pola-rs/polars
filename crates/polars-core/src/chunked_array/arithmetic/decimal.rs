use polars_compute::decimal::{
    DEC128_MAX_PREC, dec128_add, dec128_div, dec128_mul, dec128_rescale, dec128_sub,
};

use super::*;
use crate::prelude::arity::broadcast_try_binary_elementwise;

impl Add for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn add(self, rhs: Self) -> Self::Output {
        let left_s = self.scale();
        let right_s = rhs.scale();
        let scale = left_s.max(right_s);
        let prec = DEC128_MAX_PREC;
        let phys = broadcast_try_binary_elementwise(
            self.physical(),
            rhs.physical(),
            |opt_l, opt_r| {
                let (Some(l), Some(r)) = (opt_l, opt_r) else {
                    return PolarsResult::Ok(None);
                };
                let ls = dec128_rescale(l, left_s, prec, scale).ok_or_else(|| {
                    polars_err!(ComputeError: "overflow in Decimal cast for {l} from scale {left_s} to {scale}")
                })?;
                let rs = dec128_rescale(r, right_s, prec, scale).ok_or_else(|| {
                    polars_err!(ComputeError: "overflow in Decimal cast for {r} from scale {right_s} to {scale}")
                })?;
                let ret = dec128_add(ls, rs, prec).ok_or_else(
                    || polars_err!(ComputeError: "overflow in decimal addition for {ls} + {rs}"),
                )?;
                Ok(Some(ret))
            },
        );
        Ok(phys?.into_decimal_unchecked(prec, scale))
    }
}

impl Sub for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn sub(self, rhs: Self) -> Self::Output {
        let left_s = self.scale();
        let right_s = rhs.scale();
        let scale = left_s.max(right_s);
        let prec = DEC128_MAX_PREC;
        let phys = broadcast_try_binary_elementwise(
            self.physical(),
            rhs.physical(),
            |opt_l, opt_r| {
                let (Some(l), Some(r)) = (opt_l, opt_r) else {
                    return PolarsResult::Ok(None);
                };
                let ls = dec128_rescale(l, left_s, prec, scale).ok_or_else(|| {
                    polars_err!(ComputeError: "overflow in Decimal cast for {l} from scale {left_s} to {scale}")
                })?;
                let rs = dec128_rescale(r, right_s, prec, scale).ok_or_else(|| {
                    polars_err!(ComputeError: "overflow in Decimal cast for {r} from scale {right_s} to {scale}")
                })?;
                let ret = dec128_sub(ls, rs, prec).ok_or_else(
                    || polars_err!(ComputeError: "overflow in decimal subtraction for {ls} - {rs}"),
                )?;
                Ok(Some(ret))
            },
        );
        Ok(phys?.into_decimal_unchecked(prec, scale))
    }
}

impl Mul for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn mul(self, rhs: Self) -> Self::Output {
        let left_s = self.scale();
        let right_s = rhs.scale();
        let scale = left_s.max(right_s);
        let prec = DEC128_MAX_PREC;
        let phys = broadcast_try_binary_elementwise(
            self.physical(),
            rhs.physical(),
            |opt_l, opt_r| {
                let (Some(l), Some(r)) = (opt_l, opt_r) else {
                    return PolarsResult::Ok(None);
                };
                let ls = dec128_rescale(l, left_s, prec, scale).ok_or_else(|| {
                    polars_err!(ComputeError: "overflow in Decimal cast for {l} from scale {left_s} to {scale}")
                })?;
                let rs = dec128_rescale(r, right_s, prec, scale).ok_or_else(|| {
                    polars_err!(ComputeError: "overflow in Decimal cast for {r} from scale {right_s} to {scale}")
                })?;
                let ret = dec128_mul(ls, rs, prec, scale).ok_or_else(|| {
                    polars_err!(ComputeError: "overflow in decimal multiplication for {ls} * {rs}")
                })?;
                Ok(Some(ret))
            },
        );
        Ok(phys?.into_decimal_unchecked(prec, scale))
    }
}

impl Div for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn div(self, rhs: Self) -> Self::Output {
        let left_s = self.scale();
        let right_s = rhs.scale();
        let scale = left_s.max(right_s);
        let prec = DEC128_MAX_PREC;
        let phys = broadcast_try_binary_elementwise(
            self.physical(),
            rhs.physical(),
            |opt_l, opt_r| {
                let (Some(l), Some(r)) = (opt_l, opt_r) else {
                    return PolarsResult::Ok(None);
                };
                if r == 0 {
                    polars_bail!(ComputeError: "division by zero Decimal");
                }
                let ls = dec128_rescale(l, left_s, prec, scale).ok_or_else(|| {
                    polars_err!(ComputeError: "overflow in Decimal cast for {l} from scale {left_s} to {scale}")
                })?;
                let rs = dec128_rescale(r, right_s, prec, scale).ok_or_else(|| {
                    polars_err!(ComputeError: "overflow in Decimal cast for {r} from scale {right_s} to {scale}")
                })?;
                let ret = dec128_div(ls, rs, prec, scale).ok_or_else(
                    || polars_err!(ComputeError: "overflow in decimal division for {ls} / {rs}"),
                )?;
                Ok(Some(ret))
            },
        );
        Ok(phys?.into_decimal_unchecked(prec, scale))
    }
}
