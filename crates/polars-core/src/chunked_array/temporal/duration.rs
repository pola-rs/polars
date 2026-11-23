use std::cmp::Ordering;

use chrono::Duration as ChronoDuration;
use polars_arrow::array::PrimitiveArray;
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::temporal_conversions::{NANOSECONDS, NANOSECONDS_IN_DAY};
use polars_compute::min_max::MinMaxKernel;
use polars_utils::broadcast::broadcast_len;

use crate::fmt::{fmt_duration_string, iso_duration_string};
use crate::prelude::DataType::Duration;
use crate::prelude::*;

impl DurationChunked {
    pub fn time_unit(&self) -> TimeUnit {
        match &self.dtype {
            DataType::Duration(tu) => *tu,
            _ => unreachable!(),
        }
    }

    /// Change the underlying [`TimeUnit`]. And update the data accordingly
    /// (values that overflow the new time unit become null).
    #[must_use]
    pub fn cast_time_unit(&self, tu: TimeUnit) -> Self {
        let current_unit = self.time_unit();
        let mut out = self.clone();
        out.set_time_unit(tu);

        match current_unit.rescale_to(tu) {
            TimeUnitRescale::Unchanged => {},
            TimeUnitRescale::Multiply(k) => out.phys = (&self.phys).checked_mul_scalar(k),
            TimeUnitRescale::Divide(k) => out.phys = (&self.phys).wrapping_trunc_div_scalar(k),
        }
        out
    }

    /// Change the underlying [`TimeUnit`]. This does not modify the data.
    pub fn set_time_unit(&mut self, tu: TimeUnit) {
        self.dtype = Duration(tu);
    }

    /// Convert from [`Duration`] to String; note that `strftime` format
    /// strings are not supported, only the specifiers 'iso' and 'polars'.
    pub fn to_string(&self, format: &str) -> PolarsResult<StringChunked> {
        match format {
            "iso" | "iso:strict" => Ok(self.phys.apply_into_string_amortized(|v, buf| {
                iso_duration_string(buf, v, self.time_unit())
            })),
            "polars" => Ok(self.phys.apply_into_string_amortized(|v, buf| {
                fmt_duration_string(buf, v, self.time_unit()).expect("failed to format duration")
            })),
            _ => {
                polars_bail!(
                    InvalidOperation: "format {:?} not supported for Duration type (expected one of 'iso' or 'polars')",
                    format
                )
            },
        }
    }

    /// Construct a new [`DurationChunked`] from an iterator over [`ChronoDuration`].
    pub fn from_duration<I: IntoIterator<Item = ChronoDuration>>(
        name: PlSmallStr,
        v: I,
        tu: TimeUnit,
    ) -> Self {
        let func = match tu {
            TimeUnit::Nanoseconds => |v: ChronoDuration| v.num_nanoseconds().unwrap(),
            TimeUnit::Microseconds => |v: ChronoDuration| v.num_microseconds().unwrap(),
            TimeUnit::Milliseconds => |v: ChronoDuration| v.num_milliseconds(),
        };
        let vals = v.into_iter().map(func).collect::<Vec<_>>();
        Int64Chunked::from_vec(name, vals).into_duration(tu)
    }

    /// Construct a new [`DurationChunked`] from an iterator over optional [`ChronoDuration`].
    pub fn from_duration_options<I: IntoIterator<Item = Option<ChronoDuration>>>(
        name: PlSmallStr,
        v: I,
        tu: TimeUnit,
    ) -> Self {
        let func = match tu {
            TimeUnit::Nanoseconds => |v: ChronoDuration| v.num_nanoseconds().unwrap(),
            TimeUnit::Microseconds => |v: ChronoDuration| v.num_microseconds().unwrap(),
            TimeUnit::Milliseconds => |v: ChronoDuration| v.num_milliseconds(),
        };
        let vals = v.into_iter().map(|opt| opt.map(func));
        Int64Chunked::from_iter_options(name, vals).into_duration(tu)
    }
}

/// The nanoseconds in each (weeks, days, hours, minutes, seconds, milliseconds, microseconds,
/// nanoseconds) input of [`impl_duration`].
const COMPONENT_NANOSECONDS: [i64; 8] = [
    7 * NANOSECONDS_IN_DAY,
    NANOSECONDS_IN_DAY,
    3_600 * NANOSECONDS,
    60 * NANOSECONDS,
    NANOSECONDS,
    1_000_000,
    1_000,
    1,
];

/// The duration in `time_unit` of the (weeks, days, hours, minutes, seconds, milliseconds,
/// microseconds, nanoseconds) inputs, summed exactly (floats to float precision) and truncated
/// once; null where the total does not fit.
pub fn impl_duration(s: &[Column], time_unit: TimeUnit) -> PolarsResult<Column> {
    polars_ensure!(
        s.len() == 8 && s.iter().all(|c| c.dtype().is_integer() || c.dtype().is_float()),
        InvalidOperation: "duration expects 8 integer or float inputs"
    );
    let name = s[0].name().clone();
    let dtype = Duration(time_unit);
    if s.iter().any(|c| c.is_empty()) {
        return Ok(Column::new_empty(name, &dtype));
    }
    let len = broadcast_len(s.iter())?;

    // (a null scalar makes every result null)
    let is_scalar = |c: &Column| c.len() == 1 || matches!(c, Column::Scalar(_));
    if s.iter().any(|c| is_scalar(c) && c.has_nulls()) {
        return Ok(Column::full_null(name, len, &dtype));
    }

    // The non-zero inputs, with the nanoseconds in each of their units
    let is_zero = |c: &Column| is_scalar(c) && c.get(0).is_ok_and(|v| v.extract() == Some(0.0));
    let (ints, floats): (Vec<_>, Vec<_>) = s
        .iter()
        .zip(COMPONENT_NANOSECONDS)
        .filter(|(c, _)| !is_zero(c))
        .partition(|(c, _)| c.dtype().is_integer());

    // Floats are rescaled to `time_unit` (dividing exactly, unlike `Column / f64`) and summed
    let unit = time_unit.nanoseconds();
    let mut float_sum = None;
    for (c, ns) in floats {
        let c = &c.cast(&DataType::Float64)?;
        let term = match ns.cmp(&unit) {
            Ordering::Greater => c * (ns / unit) as f64,
            Ordering::Equal => c.clone(),
            Ordering::Less => {
                let k = (unit / ns) as f64;
                let c = c.as_materialized_series_maintain_scalar();
                c.f64()?.apply_values(|v| v / k).into_column()
            },
        };
        float_sum = Some(match float_sum {
            Some(sum) => (sum + term)?,
            None => term,
        });
    }

    // (floats alone are truncated by the cast, making non-finite or out-of-range values null)
    let int_len = if ints.iter().all(|(c, _)| is_scalar(c)) {
        1
    } else {
        len
    };
    let total = match (ints.is_empty(), float_sum) {
        (false, float_sum) => integer_sum(&ints, unit, int_len, float_sum.as_ref())?,
        (true, Some(float_sum)) => float_sum,
        (true, None) => Column::new_scalar(name.clone(), 0i64.into(), len),
    };
    let mut duration = total.cast(&dtype)?;
    duration.broadcast_in_place_to(len)?;
    Ok(duration.with_name(name))
}

/// The exact sum of the integer `terms` (given the nanoseconds in each of their units), in
/// the finest unit that they need, plus any `floats` (in `unit` nanoseconds), truncated into
/// `unit` nanoseconds; null if a value does not fit.
fn integer_sum(
    terms: &[(&Column, i64)],
    unit: i64,
    len: usize,
    floats: Option<&Column>,
) -> PolarsResult<Column> {
    let sum_unit = terms.iter().map(|(_, ns)| *ns).fold(unit, i64::min);
    let k = unit / sum_unit;
    let finish = |sum: Column| -> PolarsResult<Column> {
        Ok(match floats {
            None if k == 1 => sum,
            None => sum.wrapping_trunc_div_scalar(k),
            Some(floats) => add_floats(&sum, floats, k)?,
        })
    };

    // (a lone unscaled input needs no arithmetic)
    if let &[(c, ns)] = terms
        && ns == sum_unit
    {
        return finish(c.strict_cast(&DataType::Int64)?);
    }
    let mut arrays = Vec::with_capacity(terms.len());
    let mut validity = None;
    for &(c, ns) in terms {
        let c = c.strict_cast(&DataType::Int64)?;
        let s = c.as_materialized_series_maintain_scalar();
        let arr = s.i64()?.rechunk().downcast_as_array().clone();
        if arr.len() == len {
            validity = combine_validities_and(validity.as_ref(), arr.validity());
        }
        arrays.push((arr, ns / sum_unit));
    }

    // Sum in i64, flagging overflow (in any slot, including null ones); bounds-check the
    // multiplications, as `overflowing_mul` does not vectorise
    let mut sum = None;
    let mut overflow = false;
    for (arr, m) in &arrays {
        // (NEON and AVX2 have no 64-bit vector multiply, so unscaled terms skip it)
        let (m, max, min) = (*m, i64::MAX / m, i64::MIN / m);
        overflow |= if m == 1 {
            accumulate(&mut sum, arr.values(), len, |v| (v, false))
        } else {
            accumulate(&mut sum, arr.values(), len, move |v| {
                (v.wrapping_mul(m), (v > max) | (v < min))
            })
        };
    }
    if overflow {
        // (the wrapped sum is still exact if the non-null values cannot overflow, e.g. if only
        // masked null slots did)
        let (lo, hi) = arrays.iter().fold((0, 0), |(lo, hi), (arr, m)| {
            let (min, max) = arr.min_max_ignore_nan_kernel().unwrap_or((0, 0));
            (lo + min as i128 * *m as i128, hi + max as i128 * *m as i128)
        });
        if lo < i64::MIN as i128 || hi > i64::MAX as i128 {
            return exact_integer_sum(&arrays, validity, len, k, floats);
        }
    }
    let sum = Int64Chunked::from_vec_validity(PlSmallStr::EMPTY, sum.unwrap(), validity);
    finish(sum.into_column())
}

/// [`integer_sum`] when it overflows i64: summed row by row in i128 (which cannot overflow,
/// as there are at most eight terms, each below 2^113).
#[cold]
fn exact_integer_sum(
    arrays: &[(PrimitiveArray<i64>, i64)],
    validity: Option<Bitmap>,
    len: usize,
    k: i64,
    floats: Option<&Column>,
) -> PolarsResult<Column> {
    let at = |i: usize, len: usize| i.min(len - 1);
    let sum = |i: usize| -> i128 {
        let value = |arr: &PrimitiveArray<i64>| arr.value(at(i, arr.len())) as i128;
        arrays.iter().map(|(arr, m)| value(arr) * *m as i128).sum()
    };
    let floats = floats.map(|f| f.as_materialized_series_maintain_scalar().f64().cloned());
    let floats = floats
        .transpose()?
        .map(|f| f.rechunk().downcast_as_array().clone());
    let len = floats.as_ref().map_or(len, |f| len.max(f.len()));
    let k = k as i128;
    let out: Int64Chunked = (0..len)
        .map(|i| {
            if validity.as_ref().is_some_and(|valid| !valid.get_bit(i)) {
                return None;
            }
            let v = sum(i);
            match &floats {
                None => i64::try_from(v / k).ok(),
                Some(f) => {
                    let f = f.get(at(i, f.len()))?;
                    add_float(v / k, (v % k) as f64 / k as f64, f)
                },
            }
        })
        .collect();
    Ok(out.into_column())
}

/// The truncation (towards zero) of `v / k + f` for the integer sums `v` and float sums `f`
/// (either may be a single value, which is broadcast), as [`add_float`]; null if either is.
fn add_floats(ints: &Column, floats: &Column, k: i64) -> PolarsResult<Column> {
    let (ints, floats) = (
        ints.as_materialized_series_maintain_scalar(),
        floats.as_materialized_series_maintain_scalar(),
    );
    let (ints, floats) = (ints.i64()?.rechunk(), floats.f64()?.rechunk());
    let (ints, floats) = (ints.downcast_as_array(), floats.downcast_as_array());
    let len = ints.len().max(floats.len());
    let (vs, fs) = (ints.values().as_slice(), floats.values().as_slice());
    assert!([vs.len(), fs.len()].iter().all(|&n| n == len || n == 1));
    let (si, sf) = ((vs.len() > 1) as usize, (fs.len() > 1) as usize);
    let split = |v: i64| match k {
        1 => (v, 0.0),
        k => (v / k, (v % k) as f64 / k as f64),
    };
    // (as `add_float` in i64, flagging values that this cannot represent)
    let fast = |v: i64, f: f64| {
        let (q, r) = split(v);
        let (z, o1) = q.overflowing_add(f.trunc() as i64);
        let c = r + f.fract();
        let c = if z as f64 + c >= 0.0 {
            c.floor()
        } else {
            c.ceil()
        };
        let (value, o2) = z.overflowing_add(c as i64);
        (value, !(o1 | o2) & (f.abs() < 9e18))
    };

    // (a branch-free pass, with non-finite floats then made null, and any other rows that it
    // cannot represent redone in i128)
    let (mut fits, mut finite) = (true, true);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        // SAFETY: each input has length `len`, or length 1 (with a stride of 0), as asserted.
        let (v, f) = unsafe { (*vs.get_unchecked(i * si), *fs.get_unchecked(i * sf)) };
        let (value, ok) = fast(v, f);
        fits &= ok | !f.is_finite();
        finite &= f.is_finite();
        out.push(value);
    }
    let int_validity = ints.validity().filter(|_| ints.len() == len);
    let float_validity = floats.validity().filter(|_| floats.len() == len);
    let mut validity = combine_validities_and(int_validity, float_validity);
    if !finite {
        let finite = (0..len).map(|i| fs[i * sf].is_finite()).collect::<Bitmap>();
        validity = combine_validities_and(validity.as_ref(), Some(&finite));
    }
    if !fits {
        let mut valid = BitmapBuilder::with_capacity(len);
        for (i, value) in out.iter_mut().enumerate() {
            let (v, f) = (vs[i * si], fs[i * sf]);
            let exact = if fast(v, f).1 {
                Some(*value)
            } else {
                let (q, r) = split(v);
                add_float(q as i128, r, f)
            };
            valid.push(exact.is_some());
            *value = exact.unwrap_or(0);
        }
        validity = combine_validities_and(validity.as_ref(), valid.into_opt_validity().as_ref());
    }
    Ok(Int64Chunked::from_vec_validity(PlSmallStr::EMPTY, out, validity).into_column())
}

/// The truncation (towards zero) of `q + r + f`, for an integer `q` and a fraction `r` (in
/// (-1, 1)), exact but for the rounding of the fractions; `None` if it does not fit into i64
/// (or `f` is not finite).
#[inline]
fn add_float(q: i128, r: f64, f: f64) -> Option<i64> {
    // (larger floats cannot give a total in range, as |q| < 2^113)
    if f.is_nan() || f.abs() >= 1e35 {
        return None;
    }
    let (z, c) = (q + f.trunc() as i128, r + f.fract());
    let c = if z as f64 + c >= 0.0 {
        c.floor()
    } else {
        c.ceil()
    };
    i64::try_from(z + c as i128).ok()
}

/// Add the (`scale`d) `values` (a single value is broadcast to `len`) into `sum`, returning
/// whether any operation overflowed.
#[inline(always)]
fn accumulate(
    sum: &mut Option<Vec<i64>>,
    values: &[i64],
    len: usize,
    scale: impl Fn(i64) -> (i64, bool),
) -> bool {
    // (explicit loops with a local overflow flag, so that they vectorise)
    let mut overflow = false;
    match (sum.as_mut(), values) {
        (None, &[v]) => {
            let (v, o) = scale(v);
            *sum = Some(vec![v; len]);
            overflow = o;
        },
        (None, values) => {
            assert_eq!(values.len(), len);
            let mut out = Vec::with_capacity(len);
            for (out, &v) in out.spare_capacity_mut().iter_mut().zip(values) {
                let (v, o) = scale(v);
                out.write(v);
                overflow |= o;
            }
            // SAFETY: all `len` elements (as many as there are values, asserted above) were
            // initialised above.
            unsafe { out.set_len(len) };
            *sum = Some(out);
        },
        (Some(sum), &[v]) => {
            let (v, o) = scale(v);
            overflow = o;
            for s in sum.iter_mut() {
                let (r, o) = s.overflowing_add(v);
                *s = r;
                overflow |= o;
            }
        },
        (Some(sum), values) => {
            for (s, &v) in sum.iter_mut().zip(values) {
                let (v, o1) = scale(v);
                let (r, o2) = s.overflowing_add(v);
                *s = r;
                overflow |= o1 | o2;
            }
        },
    }
    overflow
}
