use num_traits::AsPrimitive;
use polars_core::prelude::*;
use polars_core::{
    with_match_physical_integer_polars_type, with_match_physical_numeric_polars_type,
};
use polars_defs::expr::RoundMode;
use polars_utils::float16::pf16;

use crate::series::ops::SeriesSealed;

/// The precision of a rounded `Decimal(precision, _)`: rounding away from zero can carry
/// into one more integer digit.
#[cfg(feature = "dtype-decimal")]
fn rounded_decimal_precision(precision: usize) -> usize {
    (precision + 1).min(polars_compute::decimal::DEC128_MAX_PREC)
}

/// Wraps rounded decimal values, raising if a carry doesn't fit the widened precision
/// (only possible at the maximum precision).
#[cfg(feature = "dtype-decimal")]
fn rounded_decimal(
    values: Int128Chunked,
    precision: usize,
    scale: usize,
    op: &str,
) -> PolarsResult<Series> {
    let precision = rounded_decimal_precision(precision);
    if precision == polars_compute::decimal::DEC128_MAX_PREC
        && values
            .iter()
            .flatten()
            .any(|v| !polars_compute::decimal::dec128_fits(v, precision))
    {
        polars_bail!(
            ComputeError: "overflow in decimal {op}: result doesn't fit Decimal({precision}, {scale})"
        );
    }
    Ok(values
        .into_decimal_unchecked(precision, scale)
        .into_series())
}

/// Apply the given rounding operation across f16/f32/f64 types.
fn apply_float_rounding(
    s: &Series,
    decimals: u32,
    f32_op: fn(f32) -> f32,
    f64_op: fn(f64) -> f64,
) -> Option<PolarsResult<Series>> {
    #[cfg(feature = "dtype-f16")]
    if let Ok(ca) = s.f16() {
        return Some(if decimals == 0 {
            let s = ca
                .apply_values(|val| f32_op(f32::from(val)).into())
                .into_series();
            Ok(s)
        } else if decimals >= 11 {
            // More precise than smallest denormal.
            Ok(s.clone())
        } else {
            let multiplier = 10.0_f32.powi(decimals as i32);
            let s = ca
                .apply_values(|val| {
                    let val_f32: f32 = val.into();
                    let ret: pf16 = (f32_op(val_f32 * multiplier) / multiplier).into();
                    if ret.is_finite() {
                        ret
                    } else {
                        // We return the original value which is correct both for overflows and non-finite inputs.
                        val
                    }
                })
                .into_series();
            Ok(s)
        });
    }
    if let Ok(ca) = s.f32() {
        return Some(if decimals == 0 {
            let s = ca.apply_values(f32_op).into_series();
            Ok(s)
        } else if decimals >= 47 {
            // More precise than smallest denormal.
            Ok(s.clone())
        } else {
            // Note we do the computation on f64 floats to not lose precision
            // when the computation is done, we cast to f32
            let multiplier = 10.0_f64.powi(decimals as i32);
            let s = ca
                .apply_values(|val| {
                    let ret = (f64_op(val as f64 * multiplier) / multiplier) as f32;
                    if ret.is_finite() {
                        ret
                    } else {
                        // We return the original value which is correct both for overflows and non-finite inputs.
                        val
                    }
                })
                .into_series();
            Ok(s)
        });
    }
    if let Ok(ca) = s.f64() {
        return Some(if decimals == 0 {
            let s = ca.apply_values(f64_op).into_series();
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
                    let ret = f64_op(val * mul2 * mul5) / mul5 * invmul2;
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
                    let ret = f64_op(val * multiplier) / multiplier;
                    if ret.is_finite() { ret } else { val }
                })
                .into_series();
            Ok(s)
        });
    }
    None
}

pub trait RoundSeries: SeriesSealed {
    /// Round underlying floating point array to the given number of decimals.
    fn round(&self, decimals: u32, mode: RoundMode) -> PolarsResult<Series> {
        let s = self.as_series();

        #[allow(clippy::type_complexity)]
        let (f32_op, f64_op): (fn(f32) -> f32, fn(f64) -> f64) = match mode {
            RoundMode::HalfToEven => (f32::round_ties_even, f64::round_ties_even),
            RoundMode::HalfAwayFromZero => (f32::round, f64::round),
            RoundMode::ToZero => (f32::trunc, f64::trunc),
        };

        if let Some(result) = apply_float_rounding(s, decimals, f32_op, f64_op) {
            return result;
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
                RoundMode::HalfToEven => ca.physical().apply_values(|v| {
                    // Compares the remainder with half the multiplier without doubling
                    // either, which could overflow i128 at scale 38.
                    let rem = v % multiplier;
                    let r = rem.unsigned_abs();
                    let rest = multiplier as u128 - r;
                    let odd = (v / multiplier) % 2 != 0;
                    let round_away = r > rest || (r == rest && odd);
                    let round_offset = if round_away {
                        if v < 0 { -multiplier } else { multiplier }
                    } else {
                        0
                    };
                    v - rem + round_offset
                }),
                RoundMode::HalfAwayFromZero => ca.physical().apply_values(|v| {
                    let rem = v % multiplier;
                    let round_offset = if rem.abs() >= threshold {
                        if v < 0 { -multiplier } else { multiplier }
                    } else {
                        0
                    };
                    v - rem + round_offset
                }),
                RoundMode::ToZero => {
                    let res = ca.physical().apply_values(|v| v - (v % multiplier));
                    return Ok(res
                        .into_decimal_unchecked(ca.precision(), scale as usize)
                        .into_series());
                },
            };
            return rounded_decimal(res, ca.precision(), scale as usize, "rounding");
        }

        let op = match mode {
            RoundMode::ToZero => "truncation ('to_zero')",
            RoundMode::HalfToEven => "rounding ('half_to_even')",
            RoundMode::HalfAwayFromZero => "rounding ('half_away_from_zero')",
        };
        polars_ensure!(
            s.dtype().is_integer(),
            op = op,
            got = s.dtype(),
            expected = "numeric"
        );
        Ok(s.clone())
    }

    /// Round underlying floating point array to the given number of significant digits.
    fn round_sig_figs(&self, digits: i32) -> PolarsResult<Series> {
        let s = self.as_series();
        polars_ensure!(
            digits >= 1,
            InvalidOperation: "digits must be an integer >= 1, got {}", digits
        );

        #[cfg(feature = "dtype-decimal")]
        if let Some(ca) = s.try_decimal() {
            let precision = ca.precision();
            let scale = ca.scale() as u32;

            let s = ca.physical().apply_values(|v| {
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
                // An exact value doesn't round, even when the threshold is 0.
                let round_offset = if rem != 0 && rem.abs() >= threshold {
                    multiplier
                } else {
                    0
                };
                let round_offset = if v < 0 { -round_offset } else { round_offset };
                v - rem + round_offset
            });

            return rounded_decimal(s, precision, scale as usize, "rounding");
        }

        polars_ensure!(
            s.dtype().is_primitive_numeric(),
            opq = round_sig_figs,
            got = s.dtype(),
            expected = "numeric"
        );
        // Integers are rounded in integer arithmetic: a round trip through f64 loses
        // the low digits of values past 2^53 and leaves results like 123499999.
        // 128-bit integers keep the float path, since a rounded magnitude can exceed i128.
        let is_int_up_to_64_bits = matches!(
            s.dtype(),
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
        );
        if is_int_up_to_64_bits {
            with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let min: i128 = <$T as PolarsNumericType>::Native::MIN.as_();
                let max: i128 = <$T as PolarsNumericType>::Native::MAX.as_();
                let s = ca.apply_values(|value| {
                    let rounded = round_int_sig_figs(AsPrimitive::<i128>::as_(value), digits);
                    AsPrimitive::<<$T as PolarsNumericType>::Native>::as_(rounded.clamp(min, max))
                }).into_series();
                return Ok(s);
            });
        }
        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            let s = ca.apply_values(|value| {
                let value = AsPrimitive::<f64>::as_(value);
                if value == 0.0 {
                    return AsPrimitive::<<$T as PolarsNumericType>::Native>::as_(value);
                }
                // To deal with very large/small numbers we split up 10^n in 5^n and 2^n.
                // The scaling by 2^n is almost always lossless.
                let exp = digits - 1 - value.abs().log10().floor() as i32;
                let pow5 = 5.0_f64.powi(exp);
                let scaled = libm::scalbn(value, exp) * pow5;
                let descaled = libm::scalbn(scaled.round() / pow5, -exp);
                AsPrimitive::<<$T as PolarsNumericType>::Native>::as_(
                    if descaled.is_finite() { descaled } else { value }
                )
            }).into_series();
            return Ok(s);
        });
    }

    /// Truncate underlying floating point array toward zero to the given number of decimals.
    fn truncate(&self, decimals: u32) -> PolarsResult<Series> {
        // Note: we make `truncate` available as its own function for discoverability
        // as this is how it is commonly known in other libraries/languages.
        self.round(decimals, RoundMode::ToZero)
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

            let ca = ca.physical().apply_values(|v| {
                let rem = v % multiplier;
                let round_offset = if v < 0 { multiplier + rem } else { rem };
                let round_offset = if rem == 0 { 0 } else { round_offset };
                v - round_offset
            });

            return rounded_decimal(ca, precision, scale as usize, "floor");
        }

        polars_ensure!(
            s.dtype().is_primitive_numeric(),
            opq = floor,
            got = s.dtype(),
            expected = "numeric"
        );
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

            let ca = ca.physical().apply_values(|v| {
                let rem = v % multiplier;
                let round_offset = if v < 0 { -rem } else { multiplier - rem };
                let round_offset = if rem == 0 { 0 } else { round_offset };
                v + round_offset
            });

            return rounded_decimal(ca, precision, scale as usize, "ceil");
        }

        polars_ensure!(
            s.dtype().is_primitive_numeric(),
            opq = ceil,
            got = s.dtype(),
            expected = "numeric"
        );
        Ok(s.clone())
    }
}

impl RoundSeries for Series {}

/// Round an integer to `digits` significant digits, half away from zero, without
/// going through floating point. `digits` is at least 1.
fn round_int_sig_figs(value: i128, digits: i32) -> i128 {
    if value == 0 {
        return 0;
    }
    let magnitude = value.unsigned_abs().ilog10() as i32 + 1;
    if magnitude <= digits {
        return value;
    }
    let multiplier = 10u128.pow((magnitude - digits) as u32);
    let rounded = (value.unsigned_abs() + multiplier / 2) / multiplier * multiplier;
    // At most one digit above u64::MAX, so this fits in i128.
    let rounded = rounded as i128;
    if value < 0 { -rounded } else { rounded }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_round_int_sig_figs() {
        assert_eq!(round_int_sig_figs(123456789, 4), 123500000);
        assert_eq!(round_int_sig_figs(234567890, 4), 234600000);
        assert_eq!(round_int_sig_figs(-123456789, 4), -123500000);
        assert_eq!(round_int_sig_figs(12345, 4), 12350);
        assert_eq!(round_int_sig_figs(-12345, 4), -12350);
        assert_eq!(round_int_sig_figs(99, 1), 100);
        assert_eq!(round_int_sig_figs(42, 4), 42);
        assert_eq!(round_int_sig_figs(0, 3), 0);
        assert_eq!(
            round_int_sig_figs(i64::MAX as i128, 2),
            9_200_000_000_000_000_000
        );
        assert_eq!(
            round_int_sig_figs(u64::MAX as i128, 3),
            18_400_000_000_000_000_000
        );
    }

    #[test]
    fn test_round_series() {
        let series = Series::new("a".into(), &[1.003, 2.23222, 3.4352]);
        let out = series.round(2, RoundMode::default()).unwrap();
        let ca = out.f64().unwrap();
        assert_eq!(ca.get(0), Some(1.0));
    }
}
