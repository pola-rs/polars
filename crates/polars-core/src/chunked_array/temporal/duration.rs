use chrono::Duration as ChronoDuration;
use num_traits::ToPrimitive;
use polars_arrow::array::{Array, ArrayRef, PrimitiveArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::temporal_conversions::{MICROSECONDS, MILLISECONDS, NANOSECONDS};
use polars_compute::min_max::MinMaxKernel;
use polars_utils::broadcast::broadcast_len;

use crate::chunked_array::ops::arity::{broadcast_binary_elementwise, unary_elementwise};
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
    // (an empty input only broadcasts against scalars)
    let len = broadcast_len(s.iter())?;
    if len == 0 {
        return Ok(Column::new_empty(name, &dtype));
    }

    // (a null scalar makes every result null)
    let is_scalar = |c: &Column| c.len() == 1 || matches!(c, Column::Scalar(_));
    if s.iter().any(|c| is_scalar(c) && c.has_nulls()) {
        return Ok(Column::full_null(name, len, &dtype));
    }

    // The inputs other than zero scalars, with their units
    let is_zero = |c: &Column| is_scalar(c) && c.get(0).is_ok_and(|v| v.extract() == Some(0.0));
    let (ints, floats): (Vec<_>, Vec<_>) = s
        .iter()
        .zip(COMPONENTS)
        .filter(|(c, _)| !is_zero(c))
        .partition(|(c, _)| c.dtype().is_integer());

    // Floats are rescaled to `time_unit` (dividing exactly, unlike `Column / f64`) and summed
    let mut float_sum = None;
    for (c, component) in floats {
        let c = c.cast(&DataType::Float64)?;
        let c = c.as_materialized_series_maintain_scalar().f64()?.clone();
        let term = match component.rescale_to(time_unit) {
            TimeUnitRescale::Unchanged => c,
            TimeUnitRescale::Multiply(k) => &c * k as f64,
            TimeUnitRescale::Divide(k) => c.apply_values(|v| v / k as f64),
        };
        float_sum = Some(match float_sum {
            Some(sum) => &sum + &term,
            None => term,
        });
    }

    // Integers are summed exactly and truncated to `time_unit` before floats are added.
    let int_len = if ints.iter().all(|(c, _)| is_scalar(c)) {
        1
    } else {
        len
    };
    let total = match (ints.is_empty(), float_sum) {
        (false, float_sum) => integer_sum(&ints, time_unit, int_len, float_sum.as_ref())?,
        (true, Some(sum)) => sum.into_column(),
        (true, None) => Column::new_scalar(name.clone(), 0i64.into(), len),
    };
    let mut duration = total.cast(&dtype)?;
    duration.broadcast_in_place_to(len)?;
    Ok(duration.with_name(name))
}

/// The exact sum of the integer `terms` in the finest unit that they need (so that rescaling
/// them only multiplies), truncated to `time_unit` before adding `floats`; null where the final
/// duration does not fit.
fn integer_sum(
    terms: &[(&Column, Component)],
    time_unit: TimeUnit,
    len: usize,
    floats: Option<&Float64Chunked>,
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

    // Compute scalar integer inputs once, then broadcast over the float column.
    if len == 1
        && let Some(floats) = floats
    {
        let mut total = 0i128;
        for &(c, component) in terms {
            let c = c.strict_cast(&DataType::Int64)?;
            total += c.get(0)?.extract::<i64>().unwrap() as i128 * scale(component) as i128;
        }
        let q = total / k as i128;
        if q as f64 as i128 == q {
            // (the non-strict cast nulls totals that do not fit, as `add_float` does)
            return (floats + q as f64).into_column().cast(&DataType::Int64);
        }
        let out: Int64Chunked = unary_elementwise(floats, |f| add_float(q, f?));
        return Ok(out.into_column());
    }
    // (a lone unscaled input needs no arithmetic; a scalar stays a single value)
    if let &[(c, component)] = terms
        && scale(component) == 1
    {
        let c = c.strict_cast(&DataType::Int64)?;
        let c = c.as_materialized_series_maintain_scalar().into_column();
        return finish_integer_sum(c, k, floats);
    }

    let cast_terms: Vec<_> = terms
        .iter()
        .map(|&(c, component)| {
            let c = c.strict_cast(&DataType::Int64)?;
            Ok((
                c.as_materialized_series_maintain_scalar().i64()?.clone(),
                scale(component),
            ))
        })
        .collect::<PolarsResult<_>>()?;
    // Walk every input's chunk boundary. Arrow slices share their buffers, so the
    // overflow fallback only materializes the span where it is needed.
    let mut cursors: Vec<_> = cast_terms
        .iter()
        .map(|(c, _)| ChunkCursor::new(c.chunks(), c.len() == 1))
        .collect();
    let mut float_cursor = floats.map(|f| ChunkCursor::new(f.chunks(), f.len() == 1));
    let mut remaining = len;
    let mut out: Option<Column> = None;
    while remaining > 0 {
        let mut span_len = cursors
            .iter_mut()
            .map(ChunkCursor::remaining)
            .min()
            .unwrap()
            .min(remaining);
        if let Some(cursor) = &mut float_cursor {
            span_len = span_len.min(cursor.remaining());
        }

        let arrays: Vec<(PrimitiveArray<i64>, i64)> = cursors
            .iter_mut()
            .zip(&cast_terms)
            .map(|(cursor, (_, m))| (cursor.take(span_len), *m))
            .collect();
        let sliced_floats = float_cursor.as_mut().map(|cursor| {
            Float64Chunked::from_chunk_iter(PlSmallStr::EMPTY, [cursor.take(span_len)])
        });
        let part = integer_sum_chunk(&arrays, span_len, k, sliced_floats.as_ref())?;
        if span_len == len {
            return Ok(part);
        }
        match &mut out {
            Some(out) => {
                out.append_owned(part)?;
            },
            None => out = Some(part),
        }
        remaining -= span_len;
    }
    Ok(out.unwrap())
}

/// A cursor over the zero-copy slices of one input's Arrow chunks.
struct ChunkCursor<'a> {
    chunks: &'a [ArrayRef],
    index: usize,
    offset: usize,
    scalar: bool,
}

impl<'a> ChunkCursor<'a> {
    fn new(chunks: &'a [ArrayRef], scalar: bool) -> Self {
        Self {
            chunks,
            index: 0,
            offset: 0,
            scalar,
        }
    }

    fn remaining(&mut self) -> usize {
        while self.chunks[self.index].len() == self.offset {
            self.index += 1;
            self.offset = 0;
        }
        if self.scalar {
            usize::MAX
        } else {
            self.chunks[self.index].len() - self.offset
        }
    }

    fn take<A: Array + Clone>(&mut self, len: usize) -> A {
        self.remaining();
        let mut arr = self.chunks[self.index]
            .as_any()
            .downcast_ref::<A>()
            .unwrap()
            .clone();
        if !self.scalar {
            arr.slice(self.offset, len);
            self.offset += len;
        }
        arr
    }
}

/// Sum a contiguous span of integer input arrays, with a row-local i128 fallback.
fn integer_sum_chunk(
    arrays: &[(PrimitiveArray<i64>, i64)],
    len: usize,
    k: i64,
    floats: Option<&Float64Chunked>,
) -> PolarsResult<Column> {
    let mut validity = None;
    for (arr, _) in arrays {
        if arr.len() == len {
            validity = combine_validities_and(validity.as_ref(), arr.validity());
        }
    }

    // Sum in i64, flagging overflow (in any slot, including null ones); bounds-check the
    // multiplications, as `overflowing_mul` does not vectorise
    let mut sum = None;
    let mut overflow = false;
    for (arr, m) in arrays {
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
            return Ok(exact_integer_sum(arrays, validity, len, k, floats));
        }
    }
    let sum = Int64Chunked::from_vec_validity(PlSmallStr::EMPTY, sum.unwrap(), validity);
    finish_integer_sum(sum.into_column(), k, floats)
}

fn finish_integer_sum(
    sum: Column,
    k: i64,
    floats: Option<&Float64Chunked>,
) -> PolarsResult<Column> {
    let sum = if k == 1 {
        sum
    } else {
        sum.wrapping_trunc_div_scalar(k)
    };
    match floats {
        Some(floats) => {
            let sum = sum.as_materialized_series_maintain_scalar();
            let out: Int64Chunked =
                broadcast_binary_elementwise(sum.i64()?, floats, |v, f| add_float(v? as i128, f?));
            Ok(out.into_column())
        },
        None => Ok(sum),
    }
}

/// [`integer_sum`] when it overflows i64: summed row by row in i128 (which cannot overflow,
/// as there are at most eight terms, each below 2^113).
#[cold]
fn exact_integer_sum(
    arrays: &[(PrimitiveArray<i64>, i64)],
    validity: Option<Bitmap>,
    len: usize,
    k: i64,
    floats: Option<&Float64Chunked>,
) -> Column {
    let sum = |i: usize| -> i128 {
        let value = |arr: &PrimitiveArray<i64>| arr.value(i.min(arr.len() - 1)) as i128;
        arrays.iter().map(|(arr, m)| value(arr) * *m as i128).sum()
    };
    let floats = floats.map(|f| f.downcast_as_array());
    let is_valid = |i: usize| validity.as_ref().is_none_or(|valid| valid.get_bit(i));
    let out: Int64Chunked = (0..len)
        .map(|i| {
            if !is_valid(i) {
                return None;
            }
            let v = sum(i) / k as i128;
            match &floats {
                Some(f) => add_float(v, f.get(i.min(f.len() - 1))?),
                None => i64::try_from(v).ok(),
            }
        })
        .collect();
    out.into_column()
}

#[inline]
fn add_float(v: i128, f: f64) -> Option<i64> {
    if f == 0.0 {
        return i64::try_from(v).ok();
    }
    (v as f64 + f).to_i64()
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
            let out = values
                .iter()
                .map(|&v| {
                    let (v, o) = scale(v);
                    overflow |= o;
                    v
                })
                .collect();
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
