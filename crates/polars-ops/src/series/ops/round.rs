use num_traits::pow::Pow;
use polars_core::prelude::*;

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
        polars_bail!(opq = round, s.dtype());
    }

    fn round_sf(&self, significant_figures: u32) -> PolarsResult<Series> {
        let s = self.as_series();
        let significant_figures = if significant_figures == 0 {
            1
        } else {
            significant_figures
        };

        if let Ok(ca) = s.f64() {
            let s = ca
                .zip_with(
                    &ca.equal(0),
                    &ca.apply_values(|val| round_sf(val, significant_figures)),
                )?
                .into_series();
            return Ok(s);
        }
        // Note we do the computation on f64 floats to not lose precision
        // when the computation is done we cast back
        if let Ok(ca) = s.f32() {
            let s = ca
                .zip_with(
                    &ca.equal(0),
                    &ca.apply_values(|val| round_sf(val as f64, significant_figures) as f32),
                )?
                .into_series();
            return Ok(s);
        }
        if let Ok(ca) = s.i32() {
            let s = ca
                .zip_with(
                    &ca.equal(0),
                    &ca.apply_values(|val| round_sf(val as f64, significant_figures) as i32),
                )?
                .into_series();
            return Ok(s);
        }
        if let Ok(ca) = s.i64() {
            let s = ca
                .zip_with(
                    &ca.equal(0),
                    &ca.apply_values(|val| round_sf(val as f64, significant_figures) as i64),
                )?
                .into_series();
            return Ok(s);
        }
        polars_bail!(opq = round_sf, s.dtype());
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
        polars_bail!(opq = floor, s.dtype());
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
        polars_bail!(opq = ceil, s.dtype());
    }
}

impl RoundSeries for Series {}

fn get_magnitude(value: f64, significant_figures: u32) -> f64 {
    10.0.pow(significant_figures as f64 - 1.0 - ((value).log10().floor()))
}
fn round_sf(value: f64, significant_figures: u32) -> f64 {
    (value * get_magnitude(value.abs(), significant_figures)).round()
        / get_magnitude(value.abs(), significant_figures)
}

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
