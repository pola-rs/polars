use polars_arrow::time_zone::Tz;
use polars_arrow::trusted_len::TrustedLen;
use polars_core::export::rayon::prelude::*;
use polars_core::prelude::*;
use polars_core::utils::_split_offsets;
use polars_core::utils::flatten::flatten_par;
use polars_core::POOL;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ClosedWindow {
    Left,
    Right,
    Both,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StartBy {
    WindowBound,
    DataPoint,
    /// only useful if periods are weekly
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

impl Default for StartBy {
    fn default() -> Self {
        Self::WindowBound
    }
}

impl StartBy {
    pub fn weekday(&self) -> Option<u32> {
        match self {
            StartBy::Monday => Some(0),
            StartBy::Tuesday => Some(1),
            StartBy::Wednesday => Some(2),
            StartBy::Thursday => Some(3),
            StartBy::Friday => Some(4),
            StartBy::Saturday => Some(5),
            StartBy::Sunday => Some(6),
            _ => None,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn update_groups_and_bounds(
    bounds_iter: BoundsIter<'_>,
    mut start_offset: usize,
    time: &[i64],
    closed_window: ClosedWindow,
    include_lower_bound: bool,
    include_upper_bound: bool,
    lower_bound: &mut Vec<i64>,
    upper_bound: &mut Vec<i64>,
    groups: &mut Vec<[IdxSize; 2]>,
) {
    for bi in bounds_iter {
        let mut skip_window = false;
        // find starting point of window
        while start_offset < time.len() {
            let t = time[start_offset];
            if bi.is_future(t, closed_window) {
                // the window is behind the time values.
                skip_window = true;
                break;
            }
            if bi.is_member(t, closed_window) {
                break;
            }
            start_offset += 1;
        }
        if skip_window {
            start_offset = start_offset.saturating_sub(1);
            continue;
        }
        if start_offset == time.len() {
            start_offset = start_offset.saturating_sub(1);
        }

        // find members of this window
        let mut i = start_offset;
        // start next iteration 1 index back because of boundary conditions.
        // e.g. "closed left" could match the next iteration, but did not this one.
        start_offset = start_offset.saturating_sub(1);

        // last value
        if i == time.len() - 1 {
            let t = time[i];
            if bi.is_member(t, closed_window) {
                if include_lower_bound {
                    lower_bound.push(bi.start);
                }
                if include_upper_bound {
                    upper_bound.push(bi.stop);
                }
                groups.push([i as IdxSize, 1])
            }
            continue;
        }

        let first = i as IdxSize;

        while i < time.len() {
            let t = time[i];
            if !bi.is_member(t, closed_window) {
                break;
            }
            i += 1;
        }
        let len = (i as IdxSize) - first;

        if include_lower_bound {
            lower_bound.push(bi.start);
        }
        if include_upper_bound {
            upper_bound.push(bi.stop);
        }
        groups.push([first, len])
    }
}

/// Based on the given `Window`, which has an
/// - every
/// - period
/// - offset
/// window boundaries are created. And every window boundary we search for the values
/// that fit that window by the given `ClosedWindow`. The groups are return as `GroupTuples`
/// together with the lower bound and upper bound timestamps. These timestamps indicate the start (lower)
/// and end (upper) of the window of that group.
///
/// If `include_boundaries` is `false` those `lower` and `upper` vectors will be empty.
#[allow(clippy::too_many_arguments)]
pub fn groupby_windows(
    window: Window,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: &Option<TimeZone>,
    include_lower_bound: bool,
    include_upper_bound: bool,
    start_by: StartBy,
) -> (GroupsSlice, Vec<i64>, Vec<i64>) {
    let start = time[0];
    // the boundary we define here is not yet correct. It doesn't take 'period' into account
    // and it doesn't have the proper starting point. This boundary is used as a proxy to find
    // the proper 'boundary' in  'window.get_overlapping_bounds_iter'.
    let boundary = if time.len() > 1 {
        // +1 because left or closed boundary could match the next window if it is on the boundary
        let stop = time[time.len() - 1] + 1;
        Bounds::new_checked(start, stop)
    } else {
        let stop = start + 1;
        Bounds::new_checked(start, stop)
    };

    let size = {
        match tu {
            TimeUnit::Nanoseconds => window.estimate_overlapping_bounds_ns(boundary),
            TimeUnit::Microseconds => window.estimate_overlapping_bounds_us(boundary),
            TimeUnit::Milliseconds => window.estimate_overlapping_bounds_ms(boundary),
        }
    };
    let size_lower = if include_lower_bound { size } else { 0 };
    let size_upper = if include_upper_bound { size } else { 0 };
    let mut lower_bound = Vec::with_capacity(size_lower);
    let mut upper_bound = Vec::with_capacity(size_upper);

    let mut groups = Vec::with_capacity(size);
    let start_offset = 0;

    match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => {
            update_groups_and_bounds(
                window
                    .get_overlapping_bounds_iter(
                        boundary,
                        tu,
                        tz.parse::<Tz>().ok().as_ref(),
                        start_by,
                    )
                    .unwrap(),
                start_offset,
                time,
                closed_window,
                include_lower_bound,
                include_upper_bound,
                &mut lower_bound,
                &mut upper_bound,
                &mut groups,
            );
        }
        _ => {
            update_groups_and_bounds(
                window
                    .get_overlapping_bounds_iter(boundary, tu, None, start_by)
                    .unwrap(),
                start_offset,
                time,
                closed_window,
                include_lower_bound,
                include_upper_bound,
                &mut lower_bound,
                &mut upper_bound,
                &mut groups,
            );
        }
    };

    (groups, lower_bound, upper_bound)
}

// this assumes that the given time point is the right endpoint of the window
pub(crate) fn groupby_values_iter_lookbehind(
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<Tz>,
    start_offset: usize,
) -> impl Iterator<Item = PolarsResult<(IdxSize, IdxSize)>> + TrustedLen + '_ {
    debug_assert!(offset.duration_ns() == period.duration_ns());
    debug_assert!(offset.negative);
    let add = match tu {
        TimeUnit::Nanoseconds => Duration::add_ns,
        TimeUnit::Microseconds => Duration::add_us,
        TimeUnit::Milliseconds => Duration::add_ms,
    };

    let mut last_lookbehind_i = 0;
    time[start_offset..]
        .iter()
        .enumerate()
        .map(move |(mut i, lower)| {
            i += start_offset;
            let lower = add(&offset, *lower, tz.as_ref())?;
            let upper = add(&period, lower, tz.as_ref())?;

            let b = Bounds::new(lower, upper);

            // we have a complete lookbehind so we know that `i` is the upper bound.
            // Safety
            // we are in bounds
            let slice = {
                #[cfg(debug_assertions)]
                {
                    &time[last_lookbehind_i..i]
                }
                #[cfg(not(debug_assertions))]
                {
                    unsafe { time.get_unchecked(last_lookbehind_i..i) }
                }
            };
            let offset = slice.partition_point(|v| !b.is_member(*v, closed_window));

            let lookbehind_i = offset + last_lookbehind_i;
            // -1 for window boundary effects
            last_lookbehind_i = lookbehind_i.saturating_sub(1);

            let mut len = i - lookbehind_i;
            if matches!(closed_window, ClosedWindow::Right | ClosedWindow::Both) {
                len += 1;
            }

            Ok((lookbehind_i as IdxSize, len as IdxSize))
        })
}

// this one is correct for all lookbehind/lookaheads, but is slower
pub(crate) fn groupby_values_iter_window_behind_t(
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<Tz>,
) -> impl Iterator<Item = PolarsResult<(IdxSize, IdxSize)>> + TrustedLen + '_ {
    let add = match tu {
        TimeUnit::Nanoseconds => Duration::add_ns,
        TimeUnit::Microseconds => Duration::add_us,
        TimeUnit::Milliseconds => Duration::add_ms,
    };

    let mut lagging_offset = 0;
    time.iter().enumerate().map(move |(i, lower)| {
        let lower = add(&offset, *lower, tz.as_ref())?;
        let upper = add(&period, lower, tz.as_ref())?;

        let b = Bounds::new(lower, upper);
        if b.is_future(time[0], closed_window) {
            Ok((0, 0))
        } else {
            // find starting point of window
            // we can start searching from lagging offset as that is the minimum boundary because data is sorted
            // and every iteration this boundary shifts right
            // we cannot use binary search as a window is not binary,
            // it is false left from the window, true inside, and false right of the window
            let mut count = 0;
            for &t in &time[lagging_offset..] {
                if b.is_member(t, closed_window) || lagging_offset + count == i {
                    break;
                }
                count += 1
            }
            if lagging_offset + count != i {
                lagging_offset += count;
            }

            // Safety
            // we just iterated over value i.
            let slice = unsafe { time.get_unchecked(lagging_offset..) };
            let len = slice.partition_point(|v| b.is_member(*v, closed_window));

            Ok((lagging_offset as IdxSize, len as IdxSize))
        }
    })
}

// this one is correct for all lookbehind/lookaheads, but is slower
pub(crate) fn groupby_values_iter_partial_lookbehind(
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<Tz>,
) -> impl Iterator<Item = PolarsResult<(IdxSize, IdxSize)>> + TrustedLen + '_ {
    let add = match tu {
        TimeUnit::Nanoseconds => Duration::add_ns,
        TimeUnit::Microseconds => Duration::add_us,
        TimeUnit::Milliseconds => Duration::add_ms,
    };

    let mut lagging_offset = 0;
    time.iter().enumerate().map(move |(i, lower)| {
        let lower = add(&offset, *lower, tz.as_ref())?;
        let upper = add(&period, lower, tz.as_ref())?;

        let b = Bounds::new(lower, upper);

        for &t in &time[lagging_offset..] {
            if b.is_member(t, closed_window) || lagging_offset == i {
                break;
            }
            lagging_offset += 1;
        }

        // Safety
        // we just iterated over value i.
        let slice = unsafe { time.get_unchecked(lagging_offset..) };
        let len = slice.partition_point(|v| b.is_member(*v, closed_window));

        Ok((lagging_offset as IdxSize, len as IdxSize))
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn groupby_values_iter_partial_lookahead(
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<Tz>,
    start_offset: usize,
    upper_bound: Option<usize>,
) -> impl Iterator<Item = PolarsResult<(IdxSize, IdxSize)>> + TrustedLen + '_ {
    let upper_bound = upper_bound.unwrap_or(time.len());
    debug_assert!(!offset.negative);

    let add = match tu {
        TimeUnit::Nanoseconds => Duration::add_ns,
        TimeUnit::Microseconds => Duration::add_us,
        TimeUnit::Milliseconds => Duration::add_ms,
    };

    time[start_offset..upper_bound]
        .iter()
        .enumerate()
        .map(move |(mut i, lower)| {
            i += start_offset;
            let lower = add(&offset, *lower, tz.as_ref())?;
            let upper = add(&period, lower, tz.as_ref())?;

            let b = Bounds::new(lower, upper);

            debug_assert!(i < time.len());
            let slice = unsafe { time.get_unchecked(i..) };
            let len = slice.partition_point(|v| b.is_member(*v, closed_window));

            Ok((i as IdxSize, len as IdxSize))
        })
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn groupby_values_iter_full_lookahead(
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<Tz>,
    start_offset: usize,
    upper_bound: Option<usize>,
) -> impl Iterator<Item = PolarsResult<(IdxSize, IdxSize)>> + TrustedLen + '_ {
    let upper_bound = upper_bound.unwrap_or(time.len());
    debug_assert!(!offset.negative);

    let add = match tu {
        TimeUnit::Nanoseconds => Duration::add_ns,
        TimeUnit::Microseconds => Duration::add_us,
        TimeUnit::Milliseconds => Duration::add_ms,
    };

    time[start_offset..upper_bound]
        .iter()
        .enumerate()
        .map(move |(mut i, lower)| {
            i += start_offset;
            let lower = add(&offset, *lower, tz.as_ref())?;
            let upper = add(&period, lower, tz.as_ref())?;

            let b = Bounds::new(lower, upper);

            // find starting point of window
            for &t in &time[i..] {
                if b.is_member(t, closed_window) {
                    break;
                }
                i += 1;
            }
            if i >= time.len() {
                return Ok((i as IdxSize, 0));
            }

            let slice = unsafe { time.get_unchecked(i..) };
            let len = slice.partition_point(|v| b.is_member(*v, closed_window));

            Ok((i as IdxSize, len as IdxSize))
        })
}

#[cfg(feature = "rolling_window")]
pub(crate) fn groupby_values_iter<'a>(
    period: Duration,
    time: &'a [i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<Tz>,
) -> Box<dyn TrustedLen<Item = PolarsResult<(IdxSize, IdxSize)>> + 'a> {
    let mut offset = period;
    offset.negative = true;
    // t is at the right endpoint of the window
    let iter = groupby_values_iter_lookbehind(period, offset, time, closed_window, tu, tz, 0);
    Box::new(iter)
}

/// Different from `groupby_windows`, where define window buckets and search which values fit that
/// pre-defined bucket, this function defines every window based on the:
///     - timestamp (lower bound)
///     - timestamp + period (upper bound)
/// where timestamps are the individual values in the array `time`
pub fn groupby_values(
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<Tz>,
) -> PolarsResult<GroupsSlice> {
    let thread_offsets = _split_offsets(time.len(), POOL.current_num_threads());

    // we have a (partial) lookbehind window
    if offset.negative {
        // lookbehind
        if offset.duration_ns() == period.duration_ns() {
            // t is right at the end of the window
            // ------t---
            // [------]
            POOL.install(|| {
                let vals = thread_offsets
                    .par_iter()
                    .copied()
                    .map(|(base_offset, len)| {
                        let upper_bound = base_offset + len;
                        let iter = groupby_values_iter_lookbehind(
                            period,
                            offset,
                            &time[..upper_bound],
                            closed_window,
                            tu,
                            tz,
                            base_offset,
                        );
                        iter.map(|result| result.map(|(offset, len)| [offset, len]))
                            .collect::<PolarsResult<Vec<_>>>()
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                Ok(flatten_par(&vals))
            })
        } else if ((offset.duration_ns() >= period.duration_ns())
            && matches!(closed_window, ClosedWindow::Left | ClosedWindow::None))
            || ((offset.duration_ns() > period.duration_ns())
                && matches!(closed_window, ClosedWindow::Right | ClosedWindow::Both))
        {
            // window is completely behind t and t itself is not a member
            // ---------------t---
            //  [---]
            let iter =
                groupby_values_iter_window_behind_t(period, offset, time, closed_window, tu, tz);
            iter.map(|result| result.map(|(offset, len)| [offset, len]))
                .collect::<PolarsResult<_>>()
        }
        // partial lookbehind
        // this one is still single threaded
        // can make it parallel later, its a bit more complicated because the boundaries are unknown
        // window is with -1 periods of t
        // ----t---
        //  [---]
        else {
            let iter =
                groupby_values_iter_partial_lookbehind(period, offset, time, closed_window, tu, tz);
            iter.map(|result| result.map(|(offset, len)| [offset, len]))
                .collect::<PolarsResult<_>>()
        }
    } else if offset != Duration::parse("0ns")
        || closed_window == ClosedWindow::Right
        || closed_window == ClosedWindow::None
    {
        // window is completely ahead of t and t itself is not a member
        // --t-----------
        //        [---]
        POOL.install(|| {
            let vals = thread_offsets
                .par_iter()
                .copied()
                .map(|(base_offset, len)| {
                    let lower_bound = base_offset;
                    let upper_bound = base_offset + len;
                    let iter = groupby_values_iter_full_lookahead(
                        period,
                        offset,
                        time,
                        closed_window,
                        tu,
                        tz,
                        lower_bound,
                        Some(upper_bound),
                    );
                    iter.map(|result| result.map(|(offset, len)| [offset as IdxSize, len]))
                        .collect::<PolarsResult<Vec<_>>>()
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            Ok(flatten_par(&vals))
        })
    } else {
        // Offset is 0 and window is closed on the left:
        // it must be that the window starts at t and t is a member
        // --t-----------
        //  [---]
        POOL.install(|| {
            let vals = thread_offsets
                .par_iter()
                .copied()
                .map(|(base_offset, len)| {
                    let lower_bound = base_offset;
                    let upper_bound = base_offset + len;
                    let iter = groupby_values_iter_partial_lookahead(
                        period,
                        offset,
                        time,
                        closed_window,
                        tu,
                        tz,
                        lower_bound,
                        Some(upper_bound),
                    );
                    iter.map(|result| result.map(|(offset, len)| [offset as IdxSize, len]))
                        .collect::<PolarsResult<Vec<_>>>()
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            Ok(flatten_par(&vals))
        })
    }
}
