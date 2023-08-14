use super::*;

/// Create list entries that are range arrays
/// - if `start` and `end` are a column, every element will expand into an array in a list column.
/// - if `start` and `end` are literals the output will be of `Int64`.
#[cfg(feature = "range")]
pub fn arange(start: Expr, end: Expr, step: i64) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::Range(RangeFunction::ARange { step }),
        options: FunctionOptions {
            allow_rename: true,
            ..Default::default()
        },
    }
}

#[cfg(feature = "range")]
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

#[cfg(feature = "range")]
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

pub trait Range<T> {
    fn into_range(self, high: T) -> Expr;
}

macro_rules! impl_into_range {
    ($dt: ty) => {
        impl Range<$dt> for $dt {
            fn into_range(self, high: $dt) -> Expr {
                Expr::Literal(LiteralValue::Range {
                    low: self as i64,
                    high: high as i64,
                    data_type: DataType::Int32,
                })
            }
        }
    };
}

impl_into_range!(i32);
impl_into_range!(i64);
impl_into_range!(u32);

/// Create a range literal.
pub fn range<T: Range<T>>(low: T, high: T) -> Expr {
    low.into_range(high)
}

/// Create a date range from a `start` and `stop` expression.
#[cfg(feature = "temporal")]
pub fn date_range(
    start: Expr,
    end: Expr,
    every: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::TemporalExpr(TemporalFunction::DateRange {
            every,
            closed,
            time_unit,
            time_zone,
        }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
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
    every: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::TemporalExpr(TemporalFunction::DateRanges {
            every,
            closed,
            time_unit,
            time_zone,
        }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            cast_to_supertypes: true,
            allow_rename: true,
            ..Default::default()
        },
    }
}

/// Generate a time range.
#[cfg(feature = "temporal")]
pub fn time_range(start: Expr, end: Expr, every: Duration, closed: ClosedWindow) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::TemporalExpr(TemporalFunction::TimeRange { every, closed }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            cast_to_supertypes: false,
            allow_rename: true,
            ..Default::default()
        },
    }
}

/// Create a column of time ranges from a `start` and `stop` expression.
#[cfg(feature = "temporal")]
pub fn time_ranges(start: Expr, end: Expr, every: Duration, closed: ClosedWindow) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::TemporalExpr(TemporalFunction::TimeRanges { every, closed }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            cast_to_supertypes: false,
            allow_rename: true,
            ..Default::default()
        },
    }
}

/// Create a column of length `n` containing `n` copies of the literal `value`. Generally you won't need this function,
/// as `lit(value)` already represents a column containing only `value` whose length is automatically set to the correct
/// number of rows.
pub fn repeat<E: Into<Expr>>(value: E, n: Expr) -> Expr {
    let function = |s: Series, n: Series| {
        polars_ensure!(
            n.dtype().is_integer(),
            SchemaMismatch: "expected expression of dtype 'integer', got '{}'", n.dtype()
        );
        let first_value = n.get(0)?;
        let n = first_value.extract::<usize>().ok_or_else(
            || polars_err!(ComputeError: "could not parse value '{}' as a size.", first_value),
        )?;
        Ok(Some(s.new_from_index(0, n)))
    };
    apply_binary(value.into(), n, function, GetOutput::same_type()).alias("repeat")
}
