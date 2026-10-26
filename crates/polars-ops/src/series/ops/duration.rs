use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_core::datatypes::{AnyValue, DataType, TimeUnit, TimeUnitRescale};
use polars_core::prelude::{ChunkFullNull, Column, Int64Chunked, IntoColumn, Scalar};
use polars_error::PolarsResult;
#[cfg(not(feature = "dtype-i128"))]
use polars_error::polars_bail;
use polars_utils::broadcast::broadcast_len;
use polars_utils::pl_str::PlSmallStr;

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

    // Integer arithmetic wraps on overflow, so integers are summed with overflow checks (falling
    // back to i128, in which case out-of-range values raise when cast); floats are then added
    let mut duration = if ints.is_empty() {
        None
    } else {
        Some(match checked_integer_sum(&ints, len)? {
            Some(sum) => sum,
            #[cfg(feature = "dtype-i128")]
            None => wide_integer_sum(&ints)?,
            #[cfg(not(feature = "dtype-i128"))]
            None => {
                polars_bail!(InvalidOperation: "duration does not fit into time unit '{time_unit}'")
            },
        })
    };
    for (c, rescale) in &floats {
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

/// The sum of the (rescaled) integer inputs, using overflow-checked i64 arithmetic in a
/// single in-place pass per input; `None` if it overflows.
fn checked_integer_sum(
    terms: &[(&Column, TimeUnitRescale)],
    len: usize,
) -> PolarsResult<Option<Column>> {
    // (a lone unscaled input needs no arithmetic)
    if let [(c, TimeUnitRescale::Unchanged)] = terms {
        return c.strict_cast(&DataType::Int64).map(Some);
    }
    let mut sum: Option<Vec<i64>> = None;
    let mut validity: Option<Bitmap> = None;
    let mut overflow = false;
    for (c, rescale) in terms {
        let c = c.strict_cast(&DataType::Int64)?;
        // (divisions cannot overflow, and the existing kernel also strength-reduces them)
        let (c, k) = match rescale {
            TimeUnitRescale::Multiply(k) => (c, *k),
            TimeUnitRescale::Divide(k) => (c.wrapping_trunc_div_scalar(*k), 1),
            TimeUnitRescale::Unchanged => (c, 1),
        };
        let s = c.as_materialized_series_maintain_scalar();
        let ca = s.i64()?.rechunk();
        let arr = ca.downcast_as_array();
        if arr.len() == 1 && len != 1 {
            // (a broadcast scalar)
            if ca.get(0).is_none() {
                return Ok(Some(
                    Int64Chunked::full_null(PlSmallStr::EMPTY, len).into_column(),
                ));
            }
        } else {
            validity = combine_validities_and(validity.as_ref(), arr.validity());
        }
        let values = arr.values().as_slice();
        overflow |= if k == 1 {
            accumulate(&mut sum, values, len, |v| (v, false))
        } else {
            // (bounds-based overflow check, as `overflowing_mul` does not vectorise)
            let (max, min) = (i64::MAX / k, i64::MIN / k);
            accumulate(&mut sum, values, len, move |v| {
                (v.wrapping_mul(k), (v > max) | (v < min))
            })
        };
    }
    if overflow {
        return Ok(None);
    }
    let sum = sum.unwrap_or_else(|| vec![0; len]);
    let sum = Int64Chunked::from_vec_validity(PlSmallStr::EMPTY, sum, validity);
    Ok(Some(sum.into_column()))
}

/// Add the (`f`) rescaled `values` (a single value is broadcast to `len`) into `sum`,
/// returning whether any operation overflowed.
#[inline(always)]
fn accumulate(
    sum: &mut Option<Vec<i64>>,
    values: &[i64],
    len: usize,
    f: impl Fn(i64) -> (i64, bool),
) -> bool {
    // (explicit loops with a local overflow flag, so that they vectorise)
    let mut overflow = false;
    match (sum.as_mut(), values) {
        (None, &[v]) => {
            let (v, o) = f(v);
            *sum = Some(vec![v; len]);
            overflow = o;
        },
        (None, values) => {
            let mut out = Vec::with_capacity(len);
            for (out, &v) in out.spare_capacity_mut().iter_mut().zip(values) {
                let (v, o) = f(v);
                out.write(v);
                overflow |= o;
            }
            // SAFETY: all `values.len() == len` elements were initialised above.
            unsafe { out.set_len(values.len()) };
            *sum = Some(out);
        },
        (Some(sum), &[v]) => {
            let (v, o) = f(v);
            overflow = o;
            for s in sum.iter_mut() {
                let (r, o) = s.overflowing_add(v);
                *s = r;
                overflow |= o;
            }
        },
        (Some(sum), values) => {
            for (s, &v) in sum.iter_mut().zip(values) {
                let (v, o1) = f(v);
                let (r, o2) = s.overflowing_add(v);
                *s = r;
                overflow |= o1 | o2;
            }
        },
    }
    overflow
}

/// The exact (i128) sum of the rescaled integer inputs.
#[cfg(feature = "dtype-i128")]
fn wide_integer_sum(terms: &[(&Column, TimeUnitRescale)]) -> PolarsResult<Column> {
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
