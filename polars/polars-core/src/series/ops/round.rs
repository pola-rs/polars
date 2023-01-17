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

    /// Clamp underlying values to the `min` and `max` values.
    pub fn clip(mut self, min: AnyValue<'_>, max: AnyValue<'_>) -> PolarsResult<Self> {
        if self.dtype().is_numeric() {
            macro_rules! apply_clip {
                ($pl_type:ty, $ca:expr) => {{
                    let min = min
                        .extract::<<$pl_type as PolarsNumericType>::Native>()
                        .unwrap();
                    let max = max
                        .extract::<<$pl_type as PolarsNumericType>::Native>()
                        .unwrap();

                    $ca.apply_mut(|val| val.clamp(min, max));
                }};
            }
            let mutable = self._get_inner_mut();
            downcast_as_macro_arg_physical_mut!(mutable, apply_clip);
            Ok(self)
        } else {
            Err(PolarsError::SchemaMisMatch(
                format!("Cannot use 'clip' on dtype {:?}, consider using a when -> then -> otherwise expression", self.dtype()).into(),
            ))
        }
    }

    /// Clamp underlying values to the `max` value.
    pub fn clip_max(mut self, max: AnyValue<'_>) -> PolarsResult<Self> {
        use num::traits::clamp_max;
        if self.dtype().is_numeric() {
            macro_rules! apply_clip {
                ($pl_type:ty, $ca:expr) => {{
                    let max = max
                        .extract::<<$pl_type as PolarsNumericType>::Native>()
                        .unwrap();

                    $ca.apply_mut(|val| clamp_max(val, max));
                }};
            }
            let mutable = self._get_inner_mut();
            downcast_as_macro_arg_physical_mut!(mutable, apply_clip);
            Ok(self)
        } else {
            Err(PolarsError::SchemaMisMatch(
                format!("Cannot use 'clip' on dtype {:?}, consider using a when -> then -> otherwise expression", self.dtype()).into(),
            ))
        }
    }

    /// Clamp underlying values to the `min` value.
    pub fn clip_min(mut self, min: AnyValue<'_>) -> PolarsResult<Self> {
        use num::traits::clamp_min;

        if self.dtype().is_numeric() {
            macro_rules! apply_clip {
                ($pl_type:ty, $ca:expr) => {{
                    let min = min
                        .extract::<<$pl_type as PolarsNumericType>::Native>()
                        .unwrap();

                    $ca.apply_mut(|val| clamp_min(val, min));
                }};
            }
            let mutable = self._get_inner_mut();
            downcast_as_macro_arg_physical_mut!(mutable, apply_clip);
            Ok(self)
        } else {
            Err(PolarsError::SchemaMisMatch(
                format!("Cannot use 'clip' on dtype {:?}, consider using a when -> then -> otherwise expression", self.dtype()).into(),
            ))
        }
    }
}
