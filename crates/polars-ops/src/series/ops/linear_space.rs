use polars_core::prelude::*;
use polars_core::series::IsSorted;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Default, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
pub enum ClosedInterval {
    #[default]
    Both,
    Left,
    Right,
    None,
}

// The enumeration for a moving window is identical to ClosedInterval.
pub type ClosedWindow = ClosedInterval;

// Primarily used for date/datetime logical types.
pub fn new_linear_space_i64(
    start: i64,
    end: i64,
    n: i64,
    closed: ClosedInterval,
    name: PlSmallStr,
) -> Int64Chunked {
    let values = if n == 0 {
        Vec::<i64>::new()
    } else {
        // The bin width depends on the interval closure.
        let divisor = match closed {
            ClosedInterval::None => n + 1,
            ClosedInterval::Left => n,
            ClosedInterval::Right => n,
            ClosedInterval::Both => n - 1,
        };
        let bin_width = (end - start) as f64 / (divisor as f64);

        // For left-open intervals, increase the left by one interval.
        let start = if closed == ClosedInterval::None || closed == ClosedInterval::Right {
            start as f64 + bin_width
        } else {
            start as f64
        };

        let right_closed = closed == ClosedInterval::Right || closed == ClosedInterval::Both;
        let n = if right_closed { n - 1 } else { n };
        let values = (0..n).map(move |x| (x as f64 * bin_width + start) as i64);

        // For right-closed and fully-closed interval, ensure the last point is exact.
        if right_closed {
            // ensures floating point accuracy of final value
            values.chain(std::iter::once(end)).collect()
        } else {
            values.collect()
        }
    };
    let mut ca = Int64Chunked::new_vec(name, values);
    let is_sorted = if end < start {
        IsSorted::Descending
    } else {
        IsSorted::Ascending
    };
    ca.set_sorted_flag(is_sorted);
    ca
}

pub fn new_linear_space_f32(
    start: f32,
    end: f32,
    n: u64,
    closed: ClosedInterval,
    name: PlSmallStr,
) -> PolarsResult<Float32Chunked> {
    let mut ca = match n {
        0 => Float32Chunked::full_null(name, 0),
        1 => match closed {
            ClosedInterval::None => Float32Chunked::from_slice(name, &[(end + start) * 0.5]),
            ClosedInterval::Left | ClosedInterval::Both => {
                Float32Chunked::from_slice(name, &[start])
            },
            ClosedInterval::Right => Float32Chunked::from_slice(name, &[end]),
        },
        _ => Float32Chunked::from_iter_values(name, {
            let span = end - start;

            let (start, d, end) = match closed {
                ClosedInterval::None => {
                    let d = span / (n + 1) as f32;
                    (start + d, d, end - d)
                },
                ClosedInterval::Left => (start, span / n as f32, end - span / n as f32),
                ClosedInterval::Right => (start + span / n as f32, span / n as f32, end),
                ClosedInterval::Both => (start, span / (n - 1) as f32, end),
            };
            (0..n - 1)
                .map(move |v| (v as f32 * d) + start)
                .chain(std::iter::once(end)) // ensures floating point accuracy of final value
        }),
    };

    let is_sorted = if end < start {
        IsSorted::Descending
    } else {
        IsSorted::Ascending
    };
    ca.set_sorted_flag(is_sorted);
    Ok(ca)
}

pub fn new_linear_space_f64(
    start: f64,
    end: f64,
    n: u64,
    closed: ClosedInterval,
    name: PlSmallStr,
) -> PolarsResult<Float64Chunked> {
    let mut ca = match n {
        0 => Float64Chunked::full_null(name, 0),
        1 => match closed {
            ClosedInterval::None => Float64Chunked::from_slice(name, &[(end + start) * 0.5]),
            ClosedInterval::Left | ClosedInterval::Both => {
                Float64Chunked::from_slice(name, &[start])
            },
            ClosedInterval::Right => Float64Chunked::from_slice(name, &[end]),
        },
        _ => Float64Chunked::from_iter_values(name, {
            let span = end - start;

            let (start, d, end) = match closed {
                ClosedInterval::None => {
                    let d = span / (n + 1) as f64;
                    (start + d, d, end - d)
                },
                ClosedInterval::Left => (start, span / n as f64, end - span / n as f64),
                ClosedInterval::Right => (start + span / n as f64, span / n as f64, end),
                ClosedInterval::Both => (start, span / (n - 1) as f64, end),
            };
            (0..n - 1)
                .map(move |v| (v as f64 * d) + start)
                .chain(std::iter::once(end)) // ensures floating point accuracy of final value
        }),
    };

    let is_sorted = if end < start {
        IsSorted::Descending
    } else {
        IsSorted::Ascending
    };
    ca.set_sorted_flag(is_sorted);
    Ok(ca)
}
