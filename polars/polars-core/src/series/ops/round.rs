use crate::prelude::*;

impl Series {
    /// Round underlying floating point array to given decimal.
    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    pub fn round(&self, decimals: u32) -> Result<Self> {
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

    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    /// Floor underlying floating point array to the lowest integers smaller or equal to the float value.
    pub fn floor(&self) -> Result<Self> {
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

    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    /// Ceil underlying floating point array to the heighest integers smaller or equal to the float value.
    pub fn ceil(&self) -> Result<Self> {
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

    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    /// Ceil underlying floating point array to the heighest integers smaller or equal to the float value.
    pub fn clip(&self, min: f64, max: f64) -> Result<Self> {
        if let Ok(ca) = self.f32() {
            let min = min as f32;
            let max = max as f32;
            let s = ca.apply(|val| val.clamp(min, max)).into_series();
            return Ok(s);
        }
        if let Ok(ca) = self.f64() {
            let s = ca.apply(|val| val.clamp(min, max)).into_series();
            return Ok(s);
        }
        if let Ok(ca) = self.i64() {
            let min = min as i64;
            let max = max as i64;
            let s = ca.apply(|val| val.clamp(min, max)).into_series();
            return Ok(s);
        }
        if let Ok(ca) = self.i32() {
            let min = min as i32;
            let max = max as i32;
            let s = ca.apply(|val| val.clamp(min, max)).into_series();
            return Ok(s);
        }
        if let Ok(ca) = self.u32() {
            let min = min as u32;
            let max = max as u32;
            let s = ca.apply(|val| val.clamp(min, max)).into_series();
            return Ok(s);
        }
        Err(PolarsError::SchemaMisMatch(
            format!("{:?} is not one of {{Float32, Float64, Int32, Int64, UInt32}} consider using a when -> then -> otherwise", self.dtype()).into(),
        ))
    }
}
