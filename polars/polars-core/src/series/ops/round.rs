use num_traits::pow::Pow;
use num_traits::{clamp_max, clamp_min};

use crate::prelude::*;

impl Series {
    /// Round underlying floating point array to given decimal.
    pub fn round(&self, decimals: u32) -> PolarsResult<Self> {
        if let Ok(ca) = self.f32() {
            if decimals == 0 {
                let s = ca.apply(|val| val.round()).into_series();
                return Ok(s);
            } else {
                // Note we do the computation on f64 floats to not lose precision
                // when the computation is done, we cast to f32
                let multiplier = 10.0.pow(decimals as f64);
                let s = ca
                    .apply(|val| ((val as f64 * multiplier).round() / multiplier) as f32)
                    .into_series();
                return Ok(s);
            }
        }
        if let Ok(ca) = self.f64() {
            if decimals == 0 {
                let s = ca.apply(|val| val.round()).into_series();
                return Ok(s);
            } else {
                let multiplier = 10.0.pow(decimals as f64);
                let s = ca
                    .apply(|val| (val * multiplier).round() / multiplier)
                    .into_series();
                return Ok(s);
            }
        }
        polars_bail!(opq = round, self.dtype());
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
        polars_bail!(opq = floor, self.dtype());
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
        polars_bail!(opq = ceil, self.dtype());
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
            polars_bail!(opq = clip, self.dtype());
        }
    }

    /// Clamp underlying values to the `max` value.
    pub fn clip_max(mut self, max: AnyValue<'_>) -> PolarsResult<Self> {
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
            polars_bail!(opq = clip_max, self.dtype());
        }
    }

    /// Clamp underlying values to the `min` value.
    pub fn clip_min(mut self, min: AnyValue<'_>) -> PolarsResult<Self> {
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
            polars_bail!(opq = clip_min, self.dtype());
        }
    }
}
