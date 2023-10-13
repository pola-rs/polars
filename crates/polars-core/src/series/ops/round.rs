use num_traits::pow::Pow;

use crate::prelude::*;

impl Series {
    /// Round underlying floating point array to given decimal.
    pub fn round(&self, decimals: u32) -> PolarsResult<Self> {
        if let Ok(ca) = self.f32() {
            if decimals == 0 {
                let s = ca.apply_values(|val| val.round()).into_series();
                return Ok(s);
            } else {
                // Note we do the computation on f64 floats to not lose precision
                // when the computation is done, we cast to f32
                let multiplier = 10.0.pow(decimals as f64);
                let s = ca
                    .apply_values(|val| ((val as f64 * multiplier).round() / multiplier) as f32)
                    .into_series();
                return Ok(s);
            }
        }
        if let Ok(ca) = self.f64() {
            if decimals == 0 {
                let s = ca.apply_values(|val| val.round()).into_series();
                return Ok(s);
            } else {
                let multiplier = 10.0.pow(decimals as f64);
                let s = ca
                    .apply_values(|val| (val * multiplier).round() / multiplier)
                    .into_series();
                return Ok(s);
            }
        }
        polars_bail!(opq = round, self.dtype());
    }

    /// Floor underlying floating point array to the lowest integers smaller or equal to the float value.
    pub fn floor(&self) -> PolarsResult<Self> {
        if let Ok(ca) = self.f32() {
            let s = ca.apply_values(|val| val.floor()).into_series();
            return Ok(s);
        }
        if let Ok(ca) = self.f64() {
            let s = ca.apply_values(|val| val.floor()).into_series();
            return Ok(s);
        }
        polars_bail!(opq = floor, self.dtype());
    }

    /// Ceil underlying floating point array to the highest integers smaller or equal to the float value.
    pub fn ceil(&self) -> PolarsResult<Self> {
        if let Ok(ca) = self.f32() {
            let s = ca.apply_values(|val| val.ceil()).into_series();
            return Ok(s);
        }
        if let Ok(ca) = self.f64() {
            let s = ca.apply_values(|val| val.ceil()).into_series();
            return Ok(s);
        }
        polars_bail!(opq = ceil, self.dtype());
    }
}
