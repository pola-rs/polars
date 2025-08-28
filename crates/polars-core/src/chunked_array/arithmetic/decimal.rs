use super::*;

impl Add for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn add(self, rhs: Self) -> Self::Output {
        let scale = _get_decimal_scale_add_sub(self.scale(), rhs.scale());
        let lhs = self.to_scale(scale)?;
        let rhs = rhs.to_scale(scale)?;
        Ok((&lhs.phys + &rhs.phys).into_decimal_unchecked(None, scale))
    }
}

impl Sub for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn sub(self, rhs: Self) -> Self::Output {
        let scale = _get_decimal_scale_add_sub(self.scale(), rhs.scale());
        let lhs = self.to_scale(scale)?;
        let rhs = rhs.to_scale(scale)?;
        Ok((&lhs.phys - &rhs.phys).into_decimal_unchecked(None, scale))
    }
}

impl Mul for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn mul(self, rhs: Self) -> Self::Output {
        let scale = _get_decimal_scale_mul(self.scale(), rhs.scale());
        Ok((&self.phys * &rhs.phys).into_decimal_unchecked(None, scale))
    }
}

impl Div for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn div(self, rhs: Self) -> Self::Output {
        let scale = _get_decimal_scale_div(self.scale());
        let lhs = self.to_scale(scale + rhs.scale())?;
        Ok((&lhs.phys / &rhs.phys).into_decimal_unchecked(None, scale))
    }
}

// Used by polars-plan to determine schema.
pub fn _get_decimal_scale_add_sub(scale_left: usize, scale_right: usize) -> usize {
    scale_left.max(scale_right)
}

pub fn _get_decimal_scale_mul(scale_left: usize, scale_right: usize) -> usize {
    scale_left + scale_right
}

pub fn _get_decimal_scale_div(scale_left: usize) -> usize {
    // Follow postgres and MySQL adding a fixed scale increment of 4
    scale_left + 4
}
