use crate::prelude::*;

impl Series {
    /// Round underlying floating point array to given decimal.
    pub fn round(&self, decimals: u32) -> PolarsResult<Self> {
        use num::traits::Pow;
        if let Ok(ca) = self.f32() {
            // Note we do the computation on f64 floats to not loose precision
            // when the computation is done, we cast to f32
            let multiplier = 10.0.pow(decimals as f64);
            let s = ca
                .apply(|val| ((val as f64 * multiplier).round() / multiplier) as f32)
                .into_series();
            return Ok(s);
        }
        if let Ok(ca) = self.f64() {
            let multiplier = 10.0.pow(decimals as f64);
            let s = ca
                .apply(|val| (val * multiplier).round() / multiplier)
                .into_series();
            return Ok(s);
        }
        Err(PolarsError::SchemaMisMatch(
            format!("{:?} is not a floating point datatype", self.dtype()).into(),
        ))
    }

    /// Floor underlying floating point array to the lowest integers smaller or equal to the float value.
    pub fn floor(&self) -> PolarsResult<Self> {
        if let Ok(ca) = self.f32() {
            let s = ca.apply(|val| val.floor()).into_series();
            return Ok(s);
        }
        if let Ok(ca) = self.f64() {
            let s = ca.apply(|val| val.floor()).into_series();
            return Ok(s);
        }
        Err(PolarsError::SchemaMisMatch(
            format!("{:?} is not a floating point datatype", self.dtype()).into(),
        ))
    }

    /// Ceil underlying floating point array to the highest integers smaller or equal to the float value.
    pub fn ceil(&self) -> PolarsResult<Self> {
        if let Ok(ca) = self.f32() {
            let s = ca.apply(|val| val.ceil()).into_series();
            return Ok(s);
        }
        if let Ok(ca) = self.f64() {
            let s = ca.apply(|val| val.ceil()).into_series();
            return Ok(s);
        }
        Err(PolarsError::SchemaMisMatch(
            format!("{:?} is not a floating point datatype", self.dtype()).into(),
        ))
    }
}
