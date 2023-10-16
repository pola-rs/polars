use arrow::legacy::time_zone::Tz;
use arrow::legacy::trusted_len::TrustedLen;
use polars_core::export::rayon::prelude::*;
use polars_core::prelude::*;
use polars_core::utils::_split_offsets;
use polars_core::utils::flatten::flatten_par;
use polars_core::POOL;
use polars_utils::slice::GetSaferUnchecked;
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
pub enum Label {
    Left,
    Right,
    DataPoint,
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
    mut start: usize,
    time: &[i64],
    closed_window: ClosedWindow,
    include_lower_bound: bool,
    include_upper_bound: bool,
    lower_bound: &mut Vec<i64>,
    upper_bound: &mut Vec<i64>,
    groups: &mut Vec<[IdxSize; 2]>,
) {
    'bounds: for bi in bounds_iter {
        // find starting point of window
        for &t in &time[start..time.len().saturating_sub(1)] {
            // the window is behind the time values.
            if bi.is_future(t, closed_window) {
                continue 'bounds;
            }
            if bi.is_member_entry(t, closed_window) {
                break;
            }
            start += 1;
        }

        // find members of this window
        let mut end = start;

        // last value isn't always added
        if end == time.len() - 1 {
            let t = time[end];
            if bi.is_member(t, closed_window) {
                if include_lower_bound {
                    lower_bound.push(bi.start);
                }
                if include_upper_bound {
                    upper_bound.push(bi.stop);
                }
                groups.push([end as IdxSize, 1])
            }
            continue;
        }
        for &t in &time[end..] {
            if !bi.is_member_exit(t, closed_window) {
                break;
            }
            end += 1;
        }
        let len = end - start;

        if include_lower_bound {
            lower_bound.push(bi.start);
        }
        if include_upper_bound {
            upper_bound.push(bi.stop);
        }
        groups.push([start as IdxSize, len as IdxSize])
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
pub fn group_by_windows(
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
        },
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
        },
    };

    (groups, lower_bound, upper_bound)
}

// t is right at the end of the window
// ------t---
// [------]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn group_by_values_iter_lookbehind(
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<Tz>,
    start_offset: usize,
    upper_bound: Option<usize>,
) -> impl Iterator<Item = PolarsResult<(IdxSize, IdxSize)>> + TrustedLen + '_ {
    debug_assert!(offset.duration_ns() == period.duration_ns());
    debug_assert!(offset.negative);
    let add = match tu {
        TimeUnit::Nanoseconds => Duration::add_ns,
        TimeUnit::Microseconds => Duration::add_us,
        TimeUnit::Milliseconds => Duration::add_ms,
    };

    let upper_bound = upper_bound.unwrap_or(time.len());
    // Use binary search to find the initial start as that is behind.
    let mut start = if let Some(&t) = time.get(start_offset) {
        let lower = add(&offset, t, tz.as_ref()).unwrap();
        let upper = add(&period, lower, tz.as_ref()).unwrap();
        let b = Bounds::new(lower, upper);
        let slice = &time[..start_offset];
        slice.partition_point(|v| !b.is_member(*v, closed_window))
    } else {
        0
    };
    let mut end = start;
    time[start_offset..upper_bound]
        .iter()
        .enumerate()
        .map(move |(mut i, t)| {
            i += start_offset;
            let lower = add(&offset, *t, tz.as_ref())?;
            let upper = add(&period, lower, tz.as_ref())?;

            let b = Bounds::new(lower, upper);

            for &t in unsafe { time.get_unchecked_release(start..i) } {
                if b.is_member_entry(t, closed_window) {
                    break;
                }
                start += 1;
            }

            // faster path, check if `i` is member.
            if b.is_member_exit(*t, closed_window) {
                end = i;
            } else {
                end = std::cmp::max(end, start);
            }
            // we still must loop to consume duplicates
            for &t in unsafe { time.get_unchecked_release(end..) } {
                if !b.is_member_exit(t, closed_window) {
                    break;
                }
                end += 1;
            }

            let len = end - start;
            let offset = start as IdxSize;

            Ok((offset, len as IdxSize))
        })
}

// this one is correct for all lookbehind/lookaheads, but is slower
// window is completely behind t and t itself is not a member
// ---------------t---
//  [---]
pub(crate) fn group_by_values_iter_window_behind_t(
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

    let mut start = 0;
    let mut end = start;
    time.iter().map(move |lower| {
        let lower = add(&offset, *lower, tz.as_ref())?;
        let upper = add(&period, lower, tz.as_ref())?;

        let b = Bounds::new(lower, upper);
        if b.is_future(time[0], closed_window) {
            Ok((0, 0))
        } else {
            for &t in &time[start..] {
                if b.is_member_entry(t, closed_window) {
                    break;
                }
                start += 1;
            }

            end = std::cmp::max(start, end);
            for &t in &time[end..] {
                if !b.is_member_exit(t, closed_window) {
                    break;
                }
                end += 1;
            }

            let len = end - start;
            let offset = start as IdxSize;

            Ok((offset, len as IdxSize))
        }
    })
}

// window is with -1 periods of t
// ----t---
//  [---]
pub(crate) fn group_by_values_iter_partial_lookbehind(
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

    let mut start = 0;
    let mut end = start;
    time.iter().enumerate().map(move |(i, lower)| {
        let lower = add(&offset, *lower, tz.as_ref())?;
        let upper = add(&period, lower, tz.as_ref())?;

        let b = Bounds::new(lower, upper);

        for &t in &time[start..] {
            if b.is_member_entry(t, closed_window) || start == i {
                break;
            }
            start += 1;
        }

        end = std::cmp::max(start, end);
        for &t in &time[end..] {
            if !b.is_member_exit(t, closed_window) {
                break;
            }
            end += 1;
        }

        let len = end - start;
        let offset = start as IdxSize;

        Ok((offset, len as IdxSize))
    })
}

#[allow(clippy::too_many_arguments)]
// window is completely ahead of t and t itself is not a member
// --t-----------
//        [---]
pub(crate) fn group_by_values_iter_lookahead(
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

    let add = match tu {
        TimeUnit::Nanoseconds => Duration::add_ns,
        TimeUnit::Microseconds => Duration::add_us,
        TimeUnit::Milliseconds => Duration::add_ms,
    };
    let mut start = start_offset;
    let mut end = start;

    time[start_offset..upper_bound].iter().map(move |lower| {
        let lower = add(&offset, *lower, tz.as_ref())?;
        let upper = add(&period, lower, tz.as_ref())?;

        let b = Bounds::new(lower, upper);

        for &t in &time[start..] {
            if b.is_member_entry(t, closed_window) {
                break;
            }
            start += 1;
        }

        end = std::cmp::max(start, end);
        for &t in &time[end..] {
            if !b.is_member_exit(t, closed_window) {
                break;
            }
            end += 1;
        }

        let len = end - start;
        let offset = start as IdxSize;

        Ok((offset, len as IdxSize))
    })
}

#[cfg(feature = "rolling_window")]
#[inline]
pub(crate) fn group_by_values_iter(
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<Tz>,
) -> impl Iterator<Item = PolarsResult<(IdxSize, IdxSize)>> + TrustedLen + '_ {
    let mut offset = period;
    offset.negative = true;
    // t is at the right endpoint of the window
    group_by_values_iter_lookbehind(period, offset, time, closed_window, tu, tz, 0, None)
}

/// Checks if the boundary elements don't split on duplicates
fn check_splits(time: &[i64], thread_offsets: &[(usize, usize)]) -> bool {
    if time.is_empty() {
        return true;
    }
    let mut valid = true;
    for window in thread_offsets.windows(2) {
        let left_block_end = window[0].0 + window[0].1;
        let right_block_start = window[1].0;

        valid &= time[left_block_end] != time[right_block_start];
    }
    valid
}

/// Different from `group_by_windows`, where define window buckets and search which values fit that
/// pre-defined bucket, this function defines every window based on the:
///     - timestamp (lower bound)
///     - timestamp + period (upper bound)
/// where timestamps are the individual values in the array `time`
pub fn group_by_values(
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<Tz>,
) -> PolarsResult<GroupsSlice> {
    let mut thread_offsets = _split_offsets(time.len(), POOL.current_num_threads());
    // there are duplicates in the splits, so we opt for a single partition
    if !check_splits(time, &thread_offsets) {
        thread_offsets = _split_offsets(time.len(), 1)
    }

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
                        let iter = group_by_values_iter_lookbehind(
                            period,
                            offset,
                            time,
                            closed_window,
                            tu,
                            tz,
                            base_offset,
                            Some(upper_bound),
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
                group_by_values_iter_window_behind_t(period, offset, time, closed_window, tu, tz);
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
            let iter = group_by_values_iter_partial_lookbehind(
                period,
                offset,
                time,
                closed_window,
                tu,
                tz,
            );
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
                    let iter = group_by_values_iter_lookahead(
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
                    let iter = group_by_values_iter_lookahead(
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
