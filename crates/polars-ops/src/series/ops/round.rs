use num_traits::pow::Pow;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

use crate::series::ops::SeriesSealed;

pub trait RoundSeries: SeriesSealed {
    /// Round underlying floating point array to given decimal.
    fn round(&self, decimals: u32) -> PolarsResult<Series> {
        let s = self.as_series();

        if let Ok(ca) = s.f32() {
            return if decimals == 0 {
                let s = ca.apply_values(|val| val.round()).into_series();
                Ok(s)
            } else {
                // Note we do the computation on f64 floats to not lose precision
                // when the computation is done, we cast to f32
                let multiplier = 10.0.pow(decimals as f64);
                let s = ca
                    .apply_values(|val| ((val as f64 * multiplier).round() / multiplier) as f32)
                    .into_series();
                Ok(s)
            };
        }
        if let Ok(ca) = s.f64() {
            return if decimals == 0 {
                let s = ca.apply_values(|val| val.round()).into_series();
                Ok(s)
            } else {
                let multiplier = 10.0.pow(decimals as f64);
                let s = ca
                    .apply_values(|val| (val * multiplier).round() / multiplier)
                    .into_series();
                Ok(s)
            };
        }

        polars_ensure!(s.dtype().is_numeric(), InvalidOperation: "round can only be used on numeric types" );
        Ok(s.clone())
    }

    fn round_sig_figs(&self, digits: i32) -> PolarsResult<Series> {
        let s = self.as_series();
        polars_ensure!(digits >= 1, InvalidOperation: "digits must be an integer >= 1");
        polars_ensure!(s.dtype().is_numeric(), InvalidOperation: "round_sig_figs can only be used on numeric types" );
        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            let s = ca.apply_values(|value| {
                let value = value as f64;
                if value == 0.0 {
                    return value as <$T as PolarsNumericType>::Native;
                }
                let magnitude = 10.0_f64.powi(digits - 1 - value.abs().log10().floor() as i32);
                ((value * magnitude).round() / magnitude) as <$T as PolarsNumericType>::Native
            }).into_series();
            return Ok(s);
        });
    }

    /// Floor underlying floating point array to the lowest integers smaller or equal to the float value.
    fn floor(&self) -> PolarsResult<Series> {
        let s = self.as_series();

        if let Ok(ca) = s.f32() {
            let s = ca.apply_values(|val| val.floor()).into_series();
            return Ok(s);
        }
        if let Ok(ca) = s.f64() {
            let s = ca.apply_values(|val| val.floor()).into_series();
            return Ok(s);
        }

        polars_ensure!(s.dtype().is_numeric(), InvalidOperation: "floor can only be used on numeric types" );
        Ok(s.clone())
    }

    /// Ceil underlying floating point array to the highest integers smaller or equal to the float value.
    fn ceil(&self) -> PolarsResult<Series> {
        let s = self.as_series();

        if let Ok(ca) = s.f32() {
            let s = ca.apply_values(|val| val.ceil()).into_series();
            return Ok(s);
        }
        if let Ok(ca) = s.f64() {
            let s = ca.apply_values(|val| val.ceil()).into_series();
            return Ok(s);
        }

        polars_ensure!(s.dtype().is_numeric(), InvalidOperation: "ceil can only be used on numeric types" );
        Ok(s.clone())
    }
}

impl RoundSeries for Series {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_round_series() {
        let series = Series::new("a", &[1.003, 2.23222, 3.4352]);
        let out = series.round(2).unwrap();
        let ca = out.f64().unwrap();
        assert_eq!(ca.get(0), Some(1.0));
    }
}
