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
    /// Ceil underlying floating point array to the highest integers smaller or equal to the float value.
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
    /// Ceil underlying floating point array to the highest integers smaller or equal to the float value.
    pub fn clip(mut self, min: f64, max: f64) -> Result<Self> {
        if self.dtype().is_numeric() {
            macro_rules! apply_clip {
                ($pl_type:ty, $ca:expr, $min:expr, $max: expr) => {{
                    let min = min as <$pl_type as PolarsNumericType>::Native;
                    let max = max as <$pl_type as PolarsNumericType>::Native;

                    $ca.apply_mut(|val| val.clamp(min, max));
                }};
            }
            let mutable = self._get_inner_mut();
            downcast_as_macro_arg_physical_mut!(mutable, apply_clip, min, max);
            Ok(self)
        } else {
            Err(PolarsError::SchemaMisMatch(
                format!("Cannot use 'clip' on dtype {:?}, consider using a when -> then -> otherwise expression", self.dtype()).into(),
            ))
        }
    }
}
