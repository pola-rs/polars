use arrow::legacy::time_zone::Tz;
use arrow::temporal_conversions::*;
#[cfg(not(feature = "timezones"))]
use chrono::NaiveDateTime;
#[cfg(feature = "timezones")]
use chrono::{Datelike, LocalResult, NaiveDateTime, TimeZone};
use now::DateTimeNow;
use polars_core::prelude::*;

use crate::prelude::*;
#[cfg(feature = "timezones")]
use crate::utils::unlocalize_datetime;

// Probe forward to find first valid time after a DST gap.
// Most DST gaps are 60 minutes, but time zones may use 30-minute offsets.
#[cfg(feature = "timezones")]
macro_rules! impl_resolve_nonexistent {
    ($name:ident, $datetime_to_timestamp:expr, $fallback_offset:expr) => {
        fn $name(local_ndt: NaiveDateTime, tz: &Tz) -> i64 {
            for minutes in [30i64, 60] {
                let probe = local_ndt + chrono::Duration::minutes(minutes);
                match tz.from_local_datetime(&probe) {
                    LocalResult::Single(dt) => return $datetime_to_timestamp(dt.naive_utc()),
                    LocalResult::Ambiguous(earliest, _) => {
                        return $datetime_to_timestamp(earliest.naive_utc());
                    },
                    LocalResult::None => continue,
                }
            }
            $datetime_to_timestamp(local_ndt) + $fallback_offset
        }
    };
}

#[cfg(feature = "timezones")]
impl_resolve_nonexistent!(
    resolve_nonexistent_ns,
    datetime_to_timestamp_ns,
    3_600_000_000_000
);
#[cfg(feature = "timezones")]
impl_resolve_nonexistent!(
    resolve_nonexistent_us,
    datetime_to_timestamp_us,
    3_600_000_000
);
#[cfg(feature = "timezones")]
impl_resolve_nonexistent!(resolve_nonexistent_ms, datetime_to_timestamp_ms, 3_600_000);

#[cfg(feature = "timezones")]
use crate::windows::calendar::days_in_month;

#[cfg(feature = "timezones")]
fn add_duration_in_local_time(local_dt: NaiveDateTime, duration: &Duration) -> NaiveDateTime {
    if duration.months() == 0 && duration.weeks() == 0 && duration.days() == 0 {
        local_dt + chrono::Duration::nanoseconds(duration.duration_ns())
    } else {
        let days_to_add = duration.days() + duration.weeks() * 7;
        let months_to_add = duration.months();

        let mut result = local_dt;

        if months_to_add != 0 {
            let month = result.month() as i64;
            let year = result.year() as i64;
            let total_months = year * 12 + month - 1 + months_to_add;
            let new_year = (total_months / 12) as i32;
            let new_month = ((total_months % 12) + 1) as u32;
            let day = result
                .day()
                .min(days_in_month(new_year, new_month as u8) as u32);
            result = result
                .with_year(new_year)
                .and_then(|d| d.with_month(new_month))
                .and_then(|d| d.with_day(day))
                .unwrap_or(result);
        }

        if days_to_add != 0 {
            result += chrono::Duration::days(days_to_add);
        }

        result + chrono::Duration::nanoseconds(duration.nanoseconds())
    }
}

#[cfg(feature = "timezones")]
fn is_dst_error(err: &PolarsError) -> bool {
    match err {
        PolarsError::ComputeError(msg) => {
            let msg = msg.as_ref();
            msg.contains("is ambiguous in time zone")
                || msg.contains("is non-existent in time zone")
        },
        _ => false,
    }
}

// Add duration, handling DST transitions:
// - Ambiguous times (fall back): use earliest
// - Non-existent times (spring forward): find next valid time
macro_rules! impl_add_duration_with_dst_handling {
    ($name:ident, $add_method:ident, $duration_method:ident,
     $timestamp_to_datetime:expr, $datetime_to_timestamp:expr, $resolve_nonexistent:ident) => {
        fn $name(duration: &Duration, t: i64, tz: Option<&Tz>) -> i64 {
            match duration.$add_method(t, tz) {
                Ok(result) => result,
                Err(e) => {
                    #[cfg(feature = "timezones")]
                    if is_dst_error(&e) {
                        if let Some(tz) = tz {
                            let utc_dt = $timestamp_to_datetime(t);
                            let local_dt = unlocalize_datetime(utc_dt, tz);
                            let result_local = add_duration_in_local_time(local_dt, duration);

                            return match tz.from_local_datetime(&result_local) {
                                LocalResult::Single(dt) => $datetime_to_timestamp(dt.naive_utc()),
                                LocalResult::Ambiguous(earliest, _) => {
                                    $datetime_to_timestamp(earliest.naive_utc())
                                },
                                LocalResult::None => $resolve_nonexistent(result_local, tz),
                            };
                        }
                    }
                    // For non-DST errors or when timezones feature is disabled, fall back
                    t + duration.$duration_method()
                },
            }
        }
    };
}

impl_add_duration_with_dst_handling!(
    add_duration_ns_with_dst_handling,
    add_ns,
    duration_ns,
    timestamp_ns_to_datetime,
    datetime_to_timestamp_ns,
    resolve_nonexistent_ns
);
impl_add_duration_with_dst_handling!(
    add_duration_us_with_dst_handling,
    add_us,
    duration_us,
    timestamp_us_to_datetime,
    datetime_to_timestamp_us,
    resolve_nonexistent_us
);
impl_add_duration_with_dst_handling!(
    add_duration_ms_with_dst_handling,
    add_ms,
    duration_ms,
    timestamp_ms_to_datetime,
    datetime_to_timestamp_ms,
    resolve_nonexistent_ms
);

// DST-safe wrappers returning PolarsResult for use as offset_fn.
macro_rules! impl_add_dst_safe {
    ($name:ident, $inner:ident) => {
        fn $name(duration: &Duration, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
            Ok($inner(duration, t, tz))
        }
    };
}

impl_add_dst_safe!(add_ns_dst_safe, add_duration_ns_with_dst_handling);
impl_add_dst_safe!(add_us_dst_safe, add_duration_us_with_dst_handling);
impl_add_dst_safe!(add_ms_dst_safe, add_duration_ms_with_dst_handling);

/// Add duration to timestamp (ns), handling DST transitions.
pub fn duration_add_ns_dst_safe(duration: &Duration, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
    Ok(add_duration_ns_with_dst_handling(duration, t, tz))
}

/// Add duration to timestamp (us), handling DST transitions.
pub fn duration_add_us_dst_safe(duration: &Duration, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
    Ok(add_duration_us_with_dst_handling(duration, t, tz))
}

/// Add duration to timestamp (ms), handling DST transitions.
pub fn duration_add_ms_dst_safe(duration: &Duration, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
    Ok(add_duration_ms_with_dst_handling(duration, t, tz))
}

/// Ensure that earliest datapoint (`t`) is in, or in front of, first window.
///
/// For example, if we have:
///
/// - first datapoint is `2020-01-01 01:00`
/// - `every` is `'1d'`
/// - `period` is `'2d'`
/// - `offset` is `'6h'`
///
/// then truncating the earliest datapoint by `every` and adding `offset` results
/// in the window `[2020-01-01 06:00, 2020-01-03 06:00)`. To give the earliest datapoint
/// a chance of being included, we then shift the window back by `every` to
/// `[2019-12-31 06:00, 2020-01-02 06:00)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ensure_t_in_or_in_front_of_window(
    mut every: Duration,
    t: i64,
    offset_fn: fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>,
    nte_duration_fn: fn(&Duration) -> i64,
    period: Duration,
    mut start: i64,
    closed_window: ClosedWindow,
    tz: Option<&Tz>,
) -> PolarsResult<Bounds> {
    every.negative = !every.negative;
    let mut stop = offset_fn(&period, start, tz)?;

    while Bounds::new(start, stop).is_past(t, closed_window) {
        let mut gap = start - t;
        if matches!(closed_window, ClosedWindow::Right | ClosedWindow::None) {
            gap += 1;
        }
        debug_assert!(gap >= 1);

        // Ceil division
        let stride = (gap + nte_duration_fn(&every) - 1) / nte_duration_fn(&every);
        debug_assert!(stride >= 1);
        let stride = std::cmp::max(stride, 1);

        start = offset_fn(&(every * stride), start, tz)?;
        stop = offset_fn(&period, start, tz)?;
    }
    Ok(Bounds::new_checked(start, stop))
}

/// Represents a window in time
#[derive(Copy, Clone)]
pub struct Window {
    // The ith window start is expressed via this equation:
    //   window_start_i = zero + every * i
    //   window_stop_i = zero + every * i + period
    pub(crate) every: Duration,
    pub(crate) period: Duration,
    pub offset: Duration,
}

impl Window {
    pub fn new(every: Duration, period: Duration, offset: Duration) -> Self {
        debug_assert!(!every.negative);
        Self {
            every,
            period,
            offset,
        }
    }

    /// Truncate the given ns timestamp by the window boundary.
    pub fn truncate_ns(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        self.every.truncate_ns(t, tz)
    }

    /// Truncate the given us timestamp by the window boundary.
    pub fn truncate_us(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        self.every.truncate_us(t, tz)
    }

    /// Truncate the given ms timestamp by the window boundary.
    pub fn truncate_ms(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        self.every.truncate_ms(t, tz)
    }

    /// Round the given ns timestamp by the window boundary.
    pub fn round_ns(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        let t = t + self.every.duration_ns() / 2_i64;
        self.truncate_ns(t, tz)
    }

    /// Round the given us timestamp by the window boundary.
    pub fn round_us(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        let t = t + self.every.duration_ns()
            / (2 * timeunit_scale(ArrowTimeUnit::Nanosecond, ArrowTimeUnit::Microsecond) as i64);
        self.truncate_us(t, tz)
    }

    /// Round the given ms timestamp by the window boundary.
    pub fn round_ms(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        let t = t + self.every.duration_ns()
            / (2 * timeunit_scale(ArrowTimeUnit::Nanosecond, ArrowTimeUnit::Millisecond) as i64);
        self.truncate_ms(t, tz)
    }

    /// returns the bounds for the earliest window bounds
    /// that contains the given time t.  For underlapping windows that
    /// do not contain time t, the window directly after time t will be returned.
    pub fn get_earliest_bounds_ns(
        &self,
        t: i64,
        closed_window: ClosedWindow,
        tz: Option<&Tz>,
    ) -> PolarsResult<Bounds> {
        let start = self.truncate_ns(t, tz)?;
        let start = add_duration_ns_with_dst_handling(&self.offset, start, tz);
        ensure_t_in_or_in_front_of_window(
            self.every,
            t,
            add_ns_dst_safe,
            Duration::nte_duration_ns,
            self.period,
            start,
            closed_window,
            tz,
        )
    }

    pub fn get_earliest_bounds_us(
        &self,
        t: i64,
        closed_window: ClosedWindow,
        tz: Option<&Tz>,
    ) -> PolarsResult<Bounds> {
        let start = self.truncate_us(t, tz)?;
        let start = add_duration_us_with_dst_handling(&self.offset, start, tz);
        ensure_t_in_or_in_front_of_window(
            self.every,
            t,
            add_us_dst_safe,
            Duration::nte_duration_us,
            self.period,
            start,
            closed_window,
            tz,
        )
    }

    pub fn get_earliest_bounds_ms(
        &self,
        t: i64,
        closed_window: ClosedWindow,
        tz: Option<&Tz>,
    ) -> PolarsResult<Bounds> {
        let start = self.truncate_ms(t, tz)?;
        let start = add_duration_ms_with_dst_handling(&self.offset, start, tz);
        ensure_t_in_or_in_front_of_window(
            self.every,
            t,
            add_ms_dst_safe,
            Duration::nte_duration_ms,
            self.period,
            start,
            closed_window,
            tz,
        )
    }

    pub(crate) fn estimate_overlapping_bounds_ns(&self, boundary: Bounds) -> usize {
        (boundary.duration() / self.every.duration_ns()
            + self.period.duration_ns() / self.every.duration_ns()) as usize
    }

    pub(crate) fn estimate_overlapping_bounds_us(&self, boundary: Bounds) -> usize {
        (boundary.duration() / self.every.duration_us()
            + self.period.duration_us() / self.every.duration_us()) as usize
    }

    pub(crate) fn estimate_overlapping_bounds_ms(&self, boundary: Bounds) -> usize {
        (boundary.duration() / self.every.duration_ms()
            + self.period.duration_ms() / self.every.duration_ms()) as usize
    }

    pub fn get_overlapping_bounds_iter<'a>(
        &'a self,
        boundary: Bounds,
        closed_window: ClosedWindow,
        tu: TimeUnit,
        tz: Option<&'a Tz>,
        start_by: StartBy,
    ) -> PolarsResult<BoundsIter<'a>> {
        BoundsIter::new(*self, closed_window, boundary, tu, tz, start_by)
    }
}

pub struct BoundsIter<'a> {
    window: Window,
    // wrapping boundary
    boundary: Bounds,
    // boundary per window iterator
    bi: Bounds,
    tu: TimeUnit,
    tz: Option<&'a Tz>,
}
impl<'a> BoundsIter<'a> {
    fn new(
        window: Window,
        closed_window: ClosedWindow,
        boundary: Bounds,
        tu: TimeUnit,
        tz: Option<&'a Tz>,
        start_by: StartBy,
    ) -> PolarsResult<Self> {
        let bi = match start_by {
            StartBy::DataPoint => {
                let mut boundary = boundary;
                let offset_fn = match tu {
                    TimeUnit::Nanoseconds => add_ns_dst_safe,
                    TimeUnit::Microseconds => add_us_dst_safe,
                    TimeUnit::Milliseconds => add_ms_dst_safe,
                };
                boundary.stop = offset_fn(&window.period, boundary.start, tz)?;
                boundary
            },
            StartBy::WindowBound => match tu {
                TimeUnit::Nanoseconds => {
                    window.get_earliest_bounds_ns(boundary.start, closed_window, tz)?
                },
                TimeUnit::Microseconds => {
                    window.get_earliest_bounds_us(boundary.start, closed_window, tz)?
                },
                TimeUnit::Milliseconds => {
                    window.get_earliest_bounds_ms(boundary.start, closed_window, tz)?
                },
            },
            _ => {
                {
                    #[allow(clippy::type_complexity)]
                    let (from, to, offset_fn, nte_duration_fn): (
                        fn(i64) -> NaiveDateTime,
                        fn(NaiveDateTime) -> i64,
                        fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>,
                        fn(&Duration) -> i64,
                    ) = match tu {
                        TimeUnit::Nanoseconds => (
                            timestamp_ns_to_datetime,
                            datetime_to_timestamp_ns,
                            add_ns_dst_safe,
                            Duration::nte_duration_ns,
                        ),
                        TimeUnit::Microseconds => (
                            timestamp_us_to_datetime,
                            datetime_to_timestamp_us,
                            add_us_dst_safe,
                            Duration::nte_duration_us,
                        ),
                        TimeUnit::Milliseconds => (
                            timestamp_ms_to_datetime,
                            datetime_to_timestamp_ms,
                            add_ms_dst_safe,
                            Duration::nte_duration_ms,
                        ),
                    };
                    // find beginning of the week.
                    let dt = from(boundary.start);
                    match tz {
                        #[cfg(feature = "timezones")]
                        Some(tz) => {
                            let dt = tz.from_utc_datetime(&dt);
                            let dt = dt.beginning_of_week();
                            let dt = dt.naive_utc();
                            let start = to(dt);
                            // adjust start of the week based on given day of the week
                            let start = offset_fn(
                                &Duration::parse(&format!("{}d", start_by.weekday().unwrap())),
                                start,
                                Some(tz),
                            )?;
                            // apply the 'offset'
                            let start = offset_fn(&window.offset, start, Some(tz))?;
                            // make sure the first datapoint has a chance to be included
                            // and compute the end of the window defined by the 'period'
                            ensure_t_in_or_in_front_of_window(
                                window.every,
                                boundary.start,
                                offset_fn,
                                nte_duration_fn,
                                window.period,
                                start,
                                closed_window,
                                Some(tz),
                            )?
                        },
                        _ => {
                            let tz = chrono::Utc;
                            let dt = dt.and_local_timezone(tz).unwrap();
                            let dt = dt.beginning_of_week();
                            let dt = dt.naive_utc();
                            let start = to(dt);
                            // adjust start of the week based on given day of the week
                            let start = offset_fn(
                                &Duration::parse(&format!("{}d", start_by.weekday().unwrap())),
                                start,
                                None,
                            )
                            .unwrap();
                            // apply the 'offset'
                            let start = offset_fn(&window.offset, start, None).unwrap();
                            // make sure the first datapoint has a chance to be included
                            // and compute the end of the window defined by the 'period'
                            ensure_t_in_or_in_front_of_window(
                                window.every,
                                boundary.start,
                                offset_fn,
                                nte_duration_fn,
                                window.period,
                                start,
                                closed_window,
                                None,
                            )?
                        },
                    }
                }
            },
        };
        Ok(Self {
            window,
            boundary,
            bi,
            tu,
            tz,
        })
    }
}

impl Iterator for BoundsIter<'_> {
    type Item = Bounds;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bi.start < self.boundary.stop {
            let out = self.bi;
            match self.tu {
                TimeUnit::Nanoseconds => {
                    self.bi.start = add_duration_ns_with_dst_handling(
                        &self.window.every,
                        self.bi.start,
                        self.tz,
                    );
                    self.bi.stop = add_duration_ns_with_dst_handling(
                        &self.window.period,
                        self.bi.start,
                        self.tz,
                    );
                },
                TimeUnit::Microseconds => {
                    self.bi.start = add_duration_us_with_dst_handling(
                        &self.window.every,
                        self.bi.start,
                        self.tz,
                    );
                    self.bi.stop = add_duration_us_with_dst_handling(
                        &self.window.period,
                        self.bi.start,
                        self.tz,
                    );
                },
                TimeUnit::Milliseconds => {
                    self.bi.start = add_duration_ms_with_dst_handling(
                        &self.window.every,
                        self.bi.start,
                        self.tz,
                    );
                    self.bi.stop = add_duration_ms_with_dst_handling(
                        &self.window.period,
                        self.bi.start,
                        self.tz,
                    );
                },
            }
            Some(out)
        } else {
            None
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let n: i64 = n.try_into().unwrap();
        if self.bi.start < self.boundary.stop {
            match self.tu {
                TimeUnit::Nanoseconds => {
                    self.bi.start = add_duration_ns_with_dst_handling(
                        &(self.window.every * n),
                        self.bi.start,
                        self.tz,
                    );
                    self.bi.stop = add_duration_ns_with_dst_handling(
                        &self.window.period,
                        self.bi.start,
                        self.tz,
                    );
                },
                TimeUnit::Microseconds => {
                    self.bi.start = add_duration_us_with_dst_handling(
                        &(self.window.every * n),
                        self.bi.start,
                        self.tz,
                    );
                    self.bi.stop = add_duration_us_with_dst_handling(
                        &self.window.period,
                        self.bi.start,
                        self.tz,
                    );
                },
                TimeUnit::Milliseconds => {
                    self.bi.start = add_duration_ms_with_dst_handling(
                        &(self.window.every * n),
                        self.bi.start,
                        self.tz,
                    );
                    self.bi.stop = add_duration_ms_with_dst_handling(
                        &self.window.period,
                        self.bi.start,
                        self.tz,
                    );
                },
            }
            self.next()
        } else {
            None
        }
    }
}

impl<'a> BoundsIter<'a> {
    /// Number of iterations to advance, such that the bounds are on target; or, in
    /// the case of non-constant duration, close to target.
    /// Follows the `nth()` convention on Iterator indexing, i.e., a return value of 0
    /// implies advancing 1 iteration.
    pub fn get_stride(&self, target: i64) -> usize {
        let mut stride = 0;
        if self.bi.start < self.boundary.stop && target > self.bi.start {
            let gap = target - self.bi.start;
            match self.tu {
                TimeUnit::Nanoseconds => {
                    if gap
                        > self.window.every.nte_duration_ns() + self.window.period.nte_duration_ns()
                    {
                        stride = ((gap - self.window.period.nte_duration_ns()) as usize)
                            / (self.window.every.nte_duration_ns() as usize);
                    }
                },
                TimeUnit::Microseconds => {
                    if gap
                        > self.window.every.nte_duration_us() + self.window.period.nte_duration_us()
                    {
                        stride = ((gap - self.window.period.nte_duration_us()) as usize)
                            / (self.window.every.nte_duration_us() as usize);
                    }
                },
                TimeUnit::Milliseconds => {
                    if gap
                        > self.window.every.nte_duration_ms() + self.window.period.nte_duration_ms()
                    {
                        stride = ((gap - self.window.period.nte_duration_ms()) as usize)
                            / (self.window.every.nte_duration_ms() as usize);
                    }
                },
            }
        }
        stride
    }
}
