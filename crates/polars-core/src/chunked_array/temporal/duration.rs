use chrono::Duration as ChronoDuration;
use polars_arrow::array::PrimitiveArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::temporal_conversions::{MICROSECONDS, MILLISECONDS, NANOSECONDS};
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

/// The unit of each (weeks, days, hours, minutes, seconds, milliseconds, microseconds,
/// nanoseconds) input of [`impl_duration`]: a number of seconds, or a time unit.
#[derive(Clone, Copy)]
enum Component {
    Seconds(i64),
    Unit(TimeUnit),
}

const COMPONENTS: [Component; 8] = [
    Component::Seconds(7 * 86_400),
    Component::Seconds(86_400),
    Component::Seconds(3_600),
    Component::Seconds(60),
    Component::Seconds(1),
    Component::Unit(TimeUnit::Milliseconds),
    Component::Unit(TimeUnit::Microseconds),
    Component::Unit(TimeUnit::Nanoseconds),
];

impl Component {
    /// How values of this component are rescaled to `time_unit`.
    fn rescale_to(self, time_unit: TimeUnit) -> TimeUnitRescale {
        match self {
            Component::Seconds(n) => TimeUnitRescale::Multiply(
                n * match time_unit {
                    TimeUnit::Nanoseconds => NANOSECONDS,
                    TimeUnit::Microseconds => MICROSECONDS,
                    TimeUnit::Milliseconds => MILLISECONDS,
                },
            ),
            Component::Unit(unit) => unit.rescale_to(time_unit),
        }
    }
}

/// The duration in `time_unit` of the (weeks, days, hours, minutes, seconds, milliseconds,
/// microseconds, nanoseconds) inputs: integers summed exactly and truncated once, with floats
/// then added (to float precision); null where the total does not fit.
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

    // The non-zero inputs, with their units
    let is_zero = |c: &Column| is_scalar(c) && c.get(0).is_ok_and(|v| v.extract() == Some(0.0));
    let (ints, floats): (Vec<_>, Vec<_>) = s
        .iter()
        .zip(COMPONENTS)
        .filter(|(c, _)| !is_zero(c))
        .partition(|(c, _)| c.dtype().is_integer());

    // Floats are rescaled to `time_unit` (dividing exactly, unlike `Column / f64`) and summed
    let mut float_sum = None;
    for (c, component) in floats {
        let c = &c.cast(&DataType::Float64)?;
        let term = match component.rescale_to(time_unit) {
            TimeUnitRescale::Unchanged => c.clone(),
            TimeUnitRescale::Multiply(k) => c * k as f64,
            TimeUnitRescale::Divide(k) => {
                let c = c.as_materialized_series_maintain_scalar();
                c.f64()?.apply_values(|v| v / k as f64).into_column()
            },
        };
        float_sum = Some(match float_sum {
            Some(sum) => (sum + term)?,
            None => term,
        });
    }

    // Integers are summed exactly and truncated to `time_unit`, then the floats are added (with
    // the cast truncating them, and making non-finite or out-of-range values null)
    let int_len = if ints.iter().all(|(c, _)| is_scalar(c)) {
        1
    } else {
        len
    };
    let int_sum = (!ints.is_empty())
        .then(|| integer_sum(&ints, time_unit, int_len))
        .transpose()?;
    let total = match (int_sum, float_sum) {
        (Some(int_sum), Some(float_sum)) => (int_sum.cast(&DataType::Float64)? + float_sum)?,
        (Some(sum), None) | (None, Some(sum)) => sum,
        (None, None) => Column::new_scalar(name.clone(), 0i64.into(), len),
    };
    let mut duration = total.cast(&dtype)?;
    duration.broadcast_in_place_to(len)?;
    Ok(duration.with_name(name))
}

/// The exact sum of the integer `terms`, in the finest unit that they need (so that rescaling
/// them only multiplies), truncated to `time_unit`; null where it does not fit.
fn integer_sum(
    terms: &[(&Column, Component)],
    time_unit: TimeUnit,
    len: usize,
) -> PolarsResult<Column> {
    let sum_unit = terms
        .iter()
        .filter_map(|(_, component)| match component {
            Component::Unit(unit) => Some(*unit),
            Component::Seconds(_) => None,
        })
        .fold(time_unit, Ord::min);
    let scale = |component: Component| match component.rescale_to(sum_unit) {
        TimeUnitRescale::Unchanged => 1,
        TimeUnitRescale::Multiply(m) => m,
        TimeUnitRescale::Divide(_) => unreachable!("integers are summed in the finest unit"),
    };
    let k = match sum_unit.rescale_to(time_unit) {
        TimeUnitRescale::Unchanged => 1,
        TimeUnitRescale::Divide(k) => k,
        TimeUnitRescale::Multiply(_) => unreachable!("integers are summed in the finest unit"),
    };
    let finish = |sum: Column| match k {
        1 => sum,
        k => sum.wrapping_trunc_div_scalar(k),
    };

    // (a lone unscaled input needs no arithmetic)
    if let &[(c, component)] = terms
        && scale(component) == 1
    {
        return Ok(finish(c.strict_cast(&DataType::Int64)?));
    }
    let mut arrays = Vec::with_capacity(terms.len());
    let mut validity = None;
    for &(c, component) in terms {
        let c = c.strict_cast(&DataType::Int64)?;
        let s = c.as_materialized_series_maintain_scalar();
        let arr = s.i64()?.rechunk().downcast_as_array().clone();
        if arr.len() == len {
            validity = combine_validities_and(validity.as_ref(), arr.validity());
        }
        arrays.push((arr, scale(component)));
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
            return Ok(exact_integer_sum(&arrays, validity, len, k));
        }
    }
    let sum = Int64Chunked::from_vec_validity(PlSmallStr::EMPTY, sum.unwrap(), validity);
    Ok(finish(sum.into_column()))
}

/// [`integer_sum`] when it overflows i64: summed row by row in i128 (which cannot overflow,
/// as there are at most eight terms, each below 2^113).
#[cold]
fn exact_integer_sum(
    arrays: &[(PrimitiveArray<i64>, i64)],
    validity: Option<Bitmap>,
    len: usize,
    k: i64,
) -> Column {
    let sum = |i: usize| -> i128 {
        let value = |arr: &PrimitiveArray<i64>| arr.value(i.min(arr.len() - 1)) as i128;
        arrays.iter().map(|(arr, m)| value(arr) * *m as i128).sum()
    };
    let is_valid = |i: usize| validity.as_ref().is_none_or(|valid| valid.get_bit(i));
    let out: Int64Chunked = (0..len)
        .map(|i| is_valid(i).then(|| i64::try_from(sum(i) / k as i128).ok())?)
        .collect();
    out.into_column()
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
