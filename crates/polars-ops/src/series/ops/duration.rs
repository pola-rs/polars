use polars_core::datatypes::{AnyValue, DataType, TimeUnit, TimeUnitRescale};
use polars_core::prelude::{ChunkAgg, Column, Scalar};
use polars_error::PolarsResult;
#[cfg(not(feature = "dtype-i128"))]
use polars_error::polars_bail;
use polars_utils::broadcast::broadcast_len;

/// Inputs and their rescaling to the target time unit.
type Terms<'a> = Vec<(&'a Column, TimeUnitRescale)>;

pub fn impl_duration(s: &[Column], time_unit: TimeUnit) -> PolarsResult<Column> {
    let name = s[0].name().clone();
    let dtype = DataType::Duration(time_unit);
    if s.iter().any(|s| s.is_empty()) {
        return Ok(Column::new_empty(name, &dtype));
    }
    let len = broadcast_len(s.iter())?;

    // The non-zero (weeks, days, hours, minutes, seconds, milliseconds, microseconds,
    // nanoseconds) inputs; type coercion has made these either Int64 or Float64
    let is_zero_scalar = |c: &Column| {
        c.len() == 1
            && match c.get(0).unwrap() {
                v if v.is_integer() => v == AnyValue::Int64(0),
                v if v.is_float() => v == AnyValue::Float64(0.0),
                _ => false,
            }
    };
    let (ints, floats): (Terms, Terms) = s
        .iter()
        .zip(TimeUnitRescale::duration_components(time_unit))
        .filter(|(c, _)| !is_zero_scalar(c))
        .partition(|(c, _)| c.dtype().is_integer());

    // Integer arithmetic wraps on overflow, so integers are summed in i128 if the result might
    // overflow i64 (going by an upper bound on its magnitude), in which case out-of-range values
    // raise when cast; floats are then added
    let (mut duration, ints) = if !ints.is_empty() && may_overflow(&ints)? {
        (Some(wide_integer_sum(&ints, time_unit)?), vec![])
    } else {
        (None, ints)
    };
    for (c, rescale) in ints.iter().chain(&floats) {
        let term = rescaled(c, *rescale);
        duration = Some(match duration {
            Some(d) => (d + term)?,
            None => term,
        });
    }
    let duration =
        duration.unwrap_or_else(|| Column::new_scalar(name.clone(), Scalar::from(0i64), len));

    // (out-of-range integers raise, while non-finite or out-of-range floats become null)
    let mut duration = if duration.dtype().is_integer() {
        duration.strict_cast(&dtype)?
    } else {
        duration.cast(&dtype)?
    };
    duration.broadcast_in_place_to(len)?;
    Ok(duration.with_name(name))
}

/// The input rescaled with column arithmetic (truncating divisions).
fn rescaled(c: &Column, rescale: TimeUnitRescale) -> Column {
    match rescale {
        TimeUnitRescale::Unchanged => c.clone(),
        TimeUnitRescale::Multiply(k) => c * k,
        TimeUnitRescale::Divide(k) => c.wrapping_trunc_div_scalar(k),
    }
}

/// Whether the sum of the rescaled integer inputs might overflow i64, going by an upper
/// bound on its magnitude.
fn may_overflow(terms: &[(&Column, TimeUnitRescale)]) -> PolarsResult<bool> {
    let mut bound = 0i128;
    for (c, rescale) in terms {
        let c = c.strict_cast(&DataType::Int64)?;
        let s = c.as_materialized_series_maintain_scalar();
        let abs = |v: i64| (v as i128).abs();
        let max_abs = s
            .i64()?
            .min_max()
            .map_or(0, |(min, max)| abs(min).max(abs(max)));
        let factor = match rescale {
            TimeUnitRescale::Multiply(k) => *k as i128,
            _ => 1,
        };
        bound += factor * max_abs;
    }
    Ok(bound > i64::MAX as i128)
}

/// The exact (i128) sum of the rescaled integer inputs.
#[cfg(feature = "dtype-i128")]
fn wide_integer_sum(terms: &[(&Column, TimeUnitRescale)], _: TimeUnit) -> PolarsResult<Column> {
    let mut sum: Option<Column> = None;
    for (c, rescale) in terms {
        let term = rescaled(&c.cast(&DataType::Int128)?, *rescale);
        sum = Some(match sum {
            Some(s) => (s + term)?,
            None => term,
        });
    }
    Ok(sum.expect("non-empty terms"))
}

#[cfg(not(feature = "dtype-i128"))]
fn wide_integer_sum(_: &[(&Column, TimeUnitRescale)], time_unit: TimeUnit) -> PolarsResult<Column> {
    polars_bail!(InvalidOperation: "duration does not fit into time unit '{time_unit}'")
}
