use polars_core::prelude::*;
use polars_core::series::IsSorted;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Default, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "snake_case")]
pub enum ClosedInterval {
    #[default]
    Both,
    Left,
    Right,
    None,
}

pub fn new_linear_space_f32(
    start: f32,
    end: f32,
    n: u64,
    closed: ClosedInterval,
    name: PlSmallStr,
) -> PolarsResult<Series> {
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

    Ok(ca.into_series())
}

pub fn new_linear_space_f64(
    start: f64,
    end: f64,
    n: u64,
    closed: ClosedInterval,
    name: PlSmallStr,
) -> PolarsResult<Series> {
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

    Ok(ca.into_series())
}
