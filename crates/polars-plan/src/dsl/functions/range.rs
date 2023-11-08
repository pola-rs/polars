use super::*;

/// Generate a range of integers.
///
/// Alias for `int_range`.
pub fn arange(start: Expr, end: Expr, step: i64) -> Expr {
    int_range(start, end, step)
}

/// Generate a range of integers.
pub fn int_range(start: Expr, end: Expr, step: i64) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::Range(RangeFunction::IntRange { step }),
        options: FunctionOptions {
            allow_rename: true,
            ..Default::default()
        },
    }
}

/// Generate a range of integers for each row of the input columns.
pub fn int_ranges(start: Expr, end: Expr, step: i64) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::Range(RangeFunction::IntRanges { step }),
        options: FunctionOptions {
            allow_rename: true,
            ..Default::default()
        },
    }
}

/// Create a date range from a `start` and `stop` expression.
#[cfg(feature = "temporal")]
pub fn date_range(
    start: Expr,
    end: Expr,
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::Range(RangeFunction::DateRange {
            interval,
            closed,
            time_unit,
            time_zone,
        }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            cast_to_supertypes: true,
            allow_rename: true,
            ..Default::default()
        },
    }
}

/// Create a column of date ranges from a `start` and `stop` expression.
#[cfg(feature = "temporal")]
pub fn date_ranges(
    start: Expr,
    end: Expr,
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::Range(RangeFunction::DateRanges {
            interval,
            closed,
            time_unit,
            time_zone,
        }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            cast_to_supertypes: true,
            allow_rename: true,
            ..Default::default()
        },
    }
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
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::Range(RangeFunction::DatetimeRange {
            interval,
            closed,
            time_unit,
            time_zone,
        }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            cast_to_supertypes: true,
            allow_rename: true,
            ..Default::default()
        },
    }
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
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::Range(RangeFunction::DatetimeRanges {
            interval,
            closed,
            time_unit,
            time_zone,
        }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            cast_to_supertypes: true,
            allow_rename: true,
            ..Default::default()
        },
    }
}

/// Generate a time range.
#[cfg(feature = "dtype-time")]
pub fn time_range(start: Expr, end: Expr, interval: Duration, closed: ClosedWindow) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::Range(RangeFunction::TimeRange { interval, closed }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            allow_rename: true,
            ..Default::default()
        },
    }
}

/// Create a column of time ranges from a `start` and `stop` expression.
#[cfg(feature = "dtype-time")]
pub fn time_ranges(start: Expr, end: Expr, interval: Duration, closed: ClosedWindow) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::Range(RangeFunction::TimeRanges { interval, closed }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            allow_rename: true,
            ..Default::default()
        },
    }
}
