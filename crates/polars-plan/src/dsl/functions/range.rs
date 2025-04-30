use polars_ops::series::ClosedInterval;
#[cfg(feature = "temporal")]
use polars_time::ClosedWindow;

use super::*;

/// Generate a range of integers.
///
/// Alias for `int_range`.
pub fn arange(start: Expr, end: Expr, step: i64, dtype: DataType) -> Expr {
    int_range(start, end, step, dtype)
}

/// Generate a range of integers.
pub fn int_range(start: Expr, end: Expr, step: i64, dtype: DataType) -> Expr {
    Expr::n_ary(RangeFunction::IntRange { step, dtype }, vec![start, end])
}

/// Generate a range of integers for each row of the input columns.
pub fn int_ranges(start: Expr, end: Expr, step: Expr) -> Expr {
    Expr::n_ary(RangeFunction::IntRanges, vec![start, end, step])
}

/// Create a date range from a `start` and `stop` expression.
#[cfg(feature = "temporal")]
pub fn date_range(start: Expr, end: Expr, interval: Duration, closed: ClosedWindow) -> Expr {
    Expr::n_ary(
        RangeFunction::DateRange { interval, closed },
        vec![start, end],
    )
}

/// Create a column of date ranges from a `start` and `stop` expression.
#[cfg(feature = "temporal")]
pub fn date_ranges(start: Expr, end: Expr, interval: Duration, closed: ClosedWindow) -> Expr {
    Expr::n_ary(
        RangeFunction::DateRanges { interval, closed },
        vec![start, end],
    )
}

/// Create a datetime range from a `start` and `stop` expression.
#[cfg(feature = "dtype-datetime")]
pub fn datetime_range(
    start: Expr,
    end: Expr,
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> Expr {
    Expr::n_ary(
        RangeFunction::DatetimeRange {
            interval,
            closed,
            time_unit,
            time_zone,
        },
        vec![start, end],
    )
}

/// Create a column of datetime ranges from a `start` and `stop` expression.
#[cfg(feature = "dtype-datetime")]
pub fn datetime_ranges(
    start: Expr,
    end: Expr,
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> Expr {
    Expr::n_ary(
        RangeFunction::DatetimeRanges {
            interval,
            closed,
            time_unit,
            time_zone,
        },
        vec![start, end],
    )
}

/// Generate a time range.
#[cfg(feature = "dtype-time")]
pub fn time_range(start: Expr, end: Expr, interval: Duration, closed: ClosedWindow) -> Expr {
    Expr::n_ary(
        RangeFunction::TimeRange { interval, closed },
        vec![start, end],
    )
}

/// Create a column of time ranges from a `start` and `stop` expression.
#[cfg(feature = "dtype-time")]
pub fn time_ranges(start: Expr, end: Expr, interval: Duration, closed: ClosedWindow) -> Expr {
    Expr::n_ary(
        RangeFunction::TimeRanges { interval, closed },
        vec![start, end],
    )
}

/// Generate a series of equally-spaced points.
pub fn linear_space(start: Expr, end: Expr, num_samples: Expr, closed: ClosedInterval) -> Expr {
    Expr::n_ary(
        RangeFunction::LinearSpace { closed },
        vec![start, end, num_samples],
    )
}

/// Create a column of linearly-spaced sequences from 'start', 'end', and 'num_samples' expressions.
pub fn linear_spaces(
    start: Expr,
    end: Expr,
    num_samples: Expr,
    closed: ClosedInterval,
    as_array: bool,
) -> PolarsResult<Expr> {
    let mut input = Vec::<Expr>::with_capacity(3);
    input.push(start);
    input.push(end);
    let array_width = if as_array {
        Some(num_samples.extract_usize().map_err(|_| {
            polars_err!(InvalidOperation: "'as_array' is only valid when 'num_samples' is a constant integer")
        })?)
    } else {
        input.push(num_samples);
        None
    };

    Ok(Expr::n_ary(
        RangeFunction::LinearSpaces {
            closed,
            array_width,
        },
        input,
    ))
}
