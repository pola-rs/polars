use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use crate::series::ops::SeriesSealed;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "snake_case")]
#[derive(Default)]
pub enum RoundMode {
    #[default]
    HalfToEven,
    HalfAwayFromZero,
}

pub trait RoundSeries: SeriesSealed {
    /// Round underlying floating point array to given decimal.
    fn round(&self, decimals: u32, mode: RoundMode) -> PolarsResult<Series> {
        let s = self.as_series();

        if let Ok(ca) = s.f32() {
            match mode {
                RoundMode::HalfToEven => {
                    return if decimals == 0 {
                        let s = ca.apply_values(|val| val.round_ties_even()).into_series();
                        Ok(s)
                    } else if decimals >= 326 {
                        // More precise than smallest denormal.
                        Ok(s.clone())
                    } else {
                        // Note we do the computation on f64 floats to not lose precision
                        // when the computation is done, we cast to f32
                        let multiplier = 10.0_f64.powi(decimals as i32);
                        let s = ca
                            .apply_values(|val| {
                                let ret = ((val as f64 * multiplier).round_ties_even() / multiplier)
                                    as f32;
                                if ret.is_finite() {
                                    ret
                                } else {
                                    // We return the original value which is correct both for overflows and non-finite inputs.
                                    val
                                }
                            })
                            .into_series();
                        Ok(s)
                    };
                },
                RoundMode::HalfAwayFromZero => {
                    return if decimals == 0 {
                        let s = ca.apply_values(|val| val.round()).into_series();
                        Ok(s)
                    } else if decimals >= 326 {
                        // More precise than smallest denormal.
                        Ok(s.clone())
                    } else {
                        // Note we do the computation on f64 floats to not lose precision
                        // when the computation is done, we cast to f32
                        let multiplier = 10.0_f64.powi(decimals as i32);
                        let s = ca
                            .apply_values(|val| {
                                let ret = ((val as f64 * multiplier).round_ties_even() / multiplier)
                                    as f32;
                                if ret.is_finite() {
                                    ret
                                } else {
                                    // We return the original value which is correct both for overflows and non-finite inputs.
                                    val
                                }
                            })
                            .into_series();
                        Ok(s)
                    };
                },
            }
        }
        if let Ok(ca) = s.f64() {
            match mode {
                RoundMode::HalfToEven => {
                    return if decimals == 0 {
                        let s = ca.apply_values(|val| val.round_ties_even()).into_series();
                        Ok(s)
                    } else if decimals >= 326 {
                        // More precise than smallest denormal.
                        Ok(s.clone())
                    } else if decimals >= 300 {
                        // We're getting into unrepresentable territory for the multiplier
                        // here, split up the 10^n multiplier into 2^n and 5^n.
                        let mul2 = libm::scalbn(1.0, decimals as i32);
                        let invmul2 = 1.0 / mul2; // Still exact for any valid value of decimals.
                        let mul5 = 5.0_f64.powi(decimals as i32);
                        let s = ca
                            .apply_values(|val| {
                                let ret = (val * mul2 * mul5).round_ties_even() / mul5 * invmul2;
                                if ret.is_finite() {
                                    ret
                                } else {
                                    // We return the original value which is correct both for overflows and non-finite inputs.
                                    val
                                }
                            })
                            .into_series();
                        Ok(s)
                    } else {
                        let multiplier = 10.0_f64.powi(decimals as i32);
                        let s = ca
                            .apply_values(|val| {
                                let ret = (val * multiplier).round_ties_even() / multiplier;
                                if ret.is_finite() {
                                    ret
                                } else {
                                    // We return the original value which is correct both for overflows and non-finite inputs.
                                    val
                                }
                            })
                            .into_series();
                        Ok(s)
                    };
                },
                RoundMode::HalfAwayFromZero => {
                    return if decimals == 0 {
                        let s = ca.apply_values(|val| val.round()).into_series();
                        Ok(s)
                    } else if decimals >= 326 {
                        // More precise than smallest denormal.
                        Ok(s.clone())
                    } else if decimals >= 300 {
                        // We're getting into unrepresentable territory for the multiplier
                        // here, split up the 10^n multiplier into 2^n and 5^n.
                        let mul2 = libm::scalbn(1.0, decimals as i32);
                        let invmul2 = 1.0 / mul2; // Still exact for any valid value of decimals.
                        let mul5 = 5.0_f64.powi(decimals as i32);
                        let s = ca
                            .apply_values(|val| {
                                let ret = (val * mul2 * mul5).round() / mul5 * invmul2;
                                if ret.is_finite() {
                                    ret
                                } else {
                                    // We return the original value which is correct both for overflows and non-finite inputs.
                                    val
                                }
                            })
                            .into_series();
                        Ok(s)
                    } else {
                        let multiplier = 10.0_f64.powi(decimals as i32);
                        let s = ca
                            .apply_values(|val| {
                                let ret = (val * multiplier).round() / multiplier;
                                if ret.is_finite() {
                                    ret
                                } else {
                                    // We return the original value which is correct both for overflows and non-finite inputs.
                                    val
                                }
                            })
                            .into_series();
                        Ok(s)
                    };
                },
            }
        }
        #[cfg(feature = "dtype-decimal")]
        if let Some(ca) = s.try_decimal() {
            let scale = ca.scale() as u32;

            if scale <= decimals {
                return Ok(ca.clone().into_series());
            }

            let decimal_delta = scale - decimals;
            let multiplier = 10i128.pow(decimal_delta);
            let threshold = multiplier / 2;

            let res = match mode {
                RoundMode::HalfToEven => ca.apply_values(|v| {
                    let rem_big = v % (2 * multiplier);
                    let is_v_floor_even = rem_big.abs() < multiplier;
                    let rem = if is_v_floor_even {
                        rem_big
                    } else if rem_big > 0 {
                        rem_big - multiplier
                    } else {
                        rem_big + multiplier
                    };

                    let threshold = threshold + i128::from(is_v_floor_even);
                    let round_offset = if rem.abs() >= threshold {
                        if v < 0 { -multiplier } else { multiplier }
                    } else {
                        0
                    };
                    v - rem + round_offset
                }),
                RoundMode::HalfAwayFromZero => ca.apply_values(|v| {
                    let rem = v % multiplier;
                    let round_offset = if rem.abs() >= threshold {
                        if v < 0 { -multiplier } else { multiplier }
                    } else {
                        0
                    };
                    v - rem + round_offset
                }),
            };
            return Ok(res
                .into_decimal_unchecked(ca.precision(), scale as usize)
                .into_series());
        }

        polars_ensure!(s.dtype().is_integer(), InvalidOperation: "round can only be used on numeric types" );
        Ok(s.clone())
    }

    fn round_sig_figs(&self, digits: i32) -> PolarsResult<Series> {
        let s = self.as_series();
        polars_ensure!(digits >= 1, InvalidOperation: "digits must be an integer >= 1");

        #[cfg(feature = "dtype-decimal")]
        if let Some(ca) = s.try_decimal() {
            let precision = ca.precision();
            let scale = ca.scale() as u32;

            let s = ca
                .apply_values(|v| {
                    if v == 0 {
                        return 0;
                    }

                    let mut magnitude = v.abs().ilog10();
                    let magnitude_mult = 10i128.pow(magnitude); // @Q? It might be better to do this with a
                    // LUT.
                    if v.abs() > magnitude_mult {
                        magnitude += 1;
                    }
                    let decimals = magnitude.saturating_sub(digits as u32);
                    let multiplier = 10i128.pow(decimals); // @Q? It might be better to do this with a
                    // LUT.
                    let threshold = multiplier / 2;

                    // We use rounding=ROUND_HALF_EVEN
                    let rem = v % multiplier;
                    let is_v_floor_even = decimals <= scale && ((v - rem) / multiplier) % 2 == 0;
                    let threshold = threshold + i128::from(is_v_floor_even);
                    let round_offset = if rem.abs() >= threshold {
                        multiplier
                    } else {
                        0
                    };
                    let round_offset = if v < 0 { -round_offset } else { round_offset };
                    v - rem + round_offset
                })
                .into_decimal_unchecked(precision, scale as usize)
                .into_series();

            return Ok(s);
        }

        polars_ensure!(s.dtype().is_primitive_numeric(), InvalidOperation: "round_sig_figs can only be used on numeric types" );
        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            let s = ca.apply_values(|value| {
                let value = value as f64;
                if value == 0.0 {
                    return value as <$T as PolarsNumericType>::Native;
                }
                // To deal with very large/small numbers we split up 10^n in 5^n and 2^n.
                // The scaling by 2^n is almost always lossless.
                let exp = digits - 1 - value.abs().log10().floor() as i32;
                let pow5 = 5.0_f64.powi(exp);
                let scaled = libm::scalbn(value, exp) * pow5;
                let descaled = libm::scalbn(scaled.round() / pow5, -exp);
                if descaled.is_finite() {
                    descaled as <$T as PolarsNumericType>::Native
                } else {
                    value as <$T as PolarsNumericType>::Native
                }
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
        #[cfg(feature = "dtype-decimal")]
        if let Some(ca) = s.try_decimal() {
            let precision = ca.precision();
            let scale = ca.scale() as u32;
            if scale == 0 {
                return Ok(ca.clone().into_series());
            }

            let decimal_delta = scale;
            let multiplier = 10i128.pow(decimal_delta);

            let ca = ca
                .apply_values(|v| {
                    let rem = v % multiplier;
                    let round_offset = if v < 0 { multiplier + rem } else { rem };
                    let round_offset = if rem == 0 { 0 } else { round_offset };
                    v - round_offset
                })
                .into_decimal_unchecked(precision, scale as usize);

            return Ok(ca.into_series());
        }

        polars_ensure!(s.dtype().is_primitive_numeric(), InvalidOperation: "floor can only be used on numeric types" );
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
        #[cfg(feature = "dtype-decimal")]
        if let Some(ca) = s.try_decimal() {
            let precision = ca.precision();
            let scale = ca.scale() as u32;
            if scale == 0 {
                return Ok(ca.clone().into_series());
            }

            let decimal_delta = scale;
            let multiplier = 10i128.pow(decimal_delta);

            let ca = ca
                .apply_values(|v| {
                    let rem = v % multiplier;
                    let round_offset = if v < 0 { -rem } else { multiplier - rem };
                    let round_offset = if rem == 0 { 0 } else { round_offset };
                    v + round_offset
                })
                .into_decimal_unchecked(precision, scale as usize);

            return Ok(ca.into_series());
        }

        polars_ensure!(s.dtype().is_primitive_numeric(), InvalidOperation: "ceil can only be used on numeric types" );
        Ok(s.clone())
    }
}

impl RoundSeries for Series {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_round_series() {
        let series = Series::new("a".into(), &[1.003, 2.23222, 3.4352]);
        let out = series.round(2, RoundMode::default()).unwrap();
        let ca = out.f64().unwrap();
        assert_eq!(ca.get(0), Some(1.0));
    }
}
