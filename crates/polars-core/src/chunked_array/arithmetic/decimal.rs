use super::*;

impl Add for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn add(self, rhs: Self) -> Self::Output {
        let scale = self.scale().max(rhs.scale());
        let lhs = self.to_scale(scale)?;
        let rhs = rhs.to_scale(scale)?;
        Ok((&lhs.0 + &rhs.0).into_decimal_unchecked(None, scale))
    }
}

impl Sub for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn sub(self, rhs: Self) -> Self::Output {
        let scale = self.scale().max(rhs.scale());
        let lhs = self.to_scale(scale)?;
        let rhs = rhs.to_scale(scale)?;
        Ok((&lhs.0 - &rhs.0).into_decimal_unchecked(None, scale))
    }
}

impl Mul for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn mul(self, rhs: Self) -> Self::Output {
        let scale = self.scale() + rhs.scale();
        Ok((&self.0 * &rhs.0).into_decimal_unchecked(None, scale))
    }
}

impl Div for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn div(self, rhs: Self) -> Self::Output {
        // Follow postgres and MySQL adding a fixed scale increment of 4
        let scale = self.scale() + 4;
        let lhs = self.to_scale(scale + rhs.scale())?;
        Ok((&lhs.0 / &rhs.0).into_decimal_unchecked(None, scale))
    }
}
