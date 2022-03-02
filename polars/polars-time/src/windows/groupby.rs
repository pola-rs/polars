use crate::prelude::*;
use polars_arrow::utils::CustomIterTools;
use polars_core::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum ClosedWindow {
    Left,
    Right,
    Both,
    None,
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
pub fn groupby_windows(
    window: Window,
    time: &[i64],
    include_boundaries: bool,
    closed_window: ClosedWindow,
    tu: TimeUnit,
) -> (GroupsSlice, Vec<i64>, Vec<i64>) {
    let start = time[0];
    let boundary = if time.len() > 1 {
        // +1 because left or closed boundary could match the next window if it is on the boundary
        let stop = time[time.len() - 1] + 1;
        Bounds::new_checked(start, stop)
    } else {
        let stop = start + 1;
        Bounds::new_checked(start, stop)
    };

    let size = if include_boundaries {
        match tu {
            TimeUnit::Nanoseconds => window.estimate_overlapping_bounds_ns(boundary),
            TimeUnit::Microseconds => window.estimate_overlapping_bounds_us(boundary),
            TimeUnit::Milliseconds => window.estimate_overlapping_bounds_ms(boundary),
        }
    } else {
        0
    };
    let mut lower_bound = Vec::with_capacity(size);
    let mut upper_bound = Vec::with_capacity(size);

    let mut groups = match tu {
        TimeUnit::Nanoseconds => {
            Vec::with_capacity(window.estimate_overlapping_bounds_ns(boundary))
        }
        TimeUnit::Microseconds => {
            Vec::with_capacity(window.estimate_overlapping_bounds_us(boundary))
        }
        TimeUnit::Milliseconds => {
            Vec::with_capacity(window.estimate_overlapping_bounds_ms(boundary))
        }
    };
    let mut start_offset = 0;

    for bi in window.get_overlapping_bounds_iter(boundary, tu) {
        let mut skip_window = false;
        // find starting point of window
        while start_offset < time.len() {
            let t = time[start_offset];
            if bi.is_future(t) {
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
                if include_boundaries {
                    lower_bound.push(bi.start);
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

        if include_boundaries {
            lower_bound.push(bi.start);
            upper_bound.push(bi.stop);
        }
        groups.push([first, len])
    }
    (groups, lower_bound, upper_bound)
}

fn find_offset(time: &[i64], b: Bounds, closed: ClosedWindow) -> Option<usize> {
    time.iter()
        .enumerate()
        .find_map(|(i, t)| match b.is_member(*t, closed) {
            true => None,
            false => Some(i),
        })
}

/// Different from `groupby_windows`, where define window buckets and search which values fit that
/// pre-defined bucket, this function defines every window based on the:
///     - timestamp (lower bound)
///     - timestamp + period (upper bound)
/// where timestamps are the individual values in the array `time`
///
pub fn groupby_values(
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
) -> GroupsSlice {
    let add = match tu {
        TimeUnit::Nanoseconds => Duration::add_ns,
        TimeUnit::Microseconds => Duration::add_us,
        TimeUnit::Milliseconds => Duration::add_ms,
    };

    // the offset can be lagging if we have a negative offset duration
    let mut lagging_offset = 0;
    time.iter()
        .enumerate()
        .map(|(i, lower)| {
            let lower = add(&offset, *lower);
            let upper = add(&period, lower);

            let b = Bounds::new(lower, upper);

            for &t in &time[lagging_offset..] {
                if b.is_member(t, closed_window) || lagging_offset == i {
                    break;
                }
                lagging_offset += 1;
            }

            let slice = &time[lagging_offset..];
            let len = find_offset(slice, b, closed_window).unwrap_or(slice.len());

            [lagging_offset as IdxSize, len as IdxSize]
        })
        .collect_trusted()
}
