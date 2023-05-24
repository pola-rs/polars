//! # Functions
//!
//! Functions on expressions that might be useful.
//!
use std::ops::{BitAnd, BitOr};

#[cfg(feature = "temporal")]
use polars_core::export::arrow::temporal_conversions::NANOSECONDS;
#[cfg(feature = "temporal")]
use polars_core::utils::arrow::temporal_conversions::SECONDS_IN_DAY;
#[cfg(feature = "dtype-struct")]
use polars_core::utils::get_supertype;

#[cfg(feature = "arg_where")]
use crate::dsl::function_expr::FunctionExpr;
use crate::dsl::function_expr::ListFunction;
#[cfg(all(feature = "concat_str", feature = "strings"))]
use crate::dsl::function_expr::StringFunction;
use crate::dsl::*;
use crate::prelude::*;

/// Compute the covariance between two columns.
pub fn cov(a: Expr, b: Expr) -> Expr {
    let name = "cov";
    let function = move |a: Series, b: Series| {
        let s = match a.dtype() {
            DataType::Float32 => {
                let ca_a = a.f32().unwrap();
                let ca_b = b.f32().unwrap();
                Series::new(name, &[polars_core::functions::cov_f(ca_a, ca_b)])
            }
            DataType::Float64 => {
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::cov_f(ca_a, ca_b)])
            }
            DataType::Int32 => {
                let ca_a = a.i32().unwrap();
                let ca_b = b.i32().unwrap();
                Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
            }
            DataType::Int64 => {
                let ca_a = a.i64().unwrap();
                let ca_b = b.i64().unwrap();
                Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
            }
            DataType::UInt32 => {
                let ca_a = a.u32().unwrap();
                let ca_b = b.u32().unwrap();
                Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
            }
            DataType::UInt64 => {
                let ca_a = a.u64().unwrap();
                let ca_b = b.u64().unwrap();
                Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
            }
            _ => {
                let a = a.cast(&DataType::Float64)?;
                let b = b.cast(&DataType::Float64)?;
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::cov_f(ca_a, ca_b)])
            }
        };
        Ok(Some(s))
    };
    apply_binary(
        a,
        b,
        function,
        GetOutput::map_dtype(|dt| {
            if matches!(dt, DataType::Float32) {
                DataType::Float32
            } else {
                DataType::Float64
            }
        }),
    )
    .with_function_options(|mut options| {
        options.auto_explode = true;
        options.fmt_str = "cov";
        options
    })
}

/// Compute the pearson correlation between two columns.
pub fn pearson_corr(a: Expr, b: Expr, ddof: u8) -> Expr {
    let name = "pearson_corr";
    let function = move |a: Series, b: Series| {
        let s = match a.dtype() {
            DataType::Float32 => {
                let ca_a = a.f32().unwrap();
                let ca_b = b.f32().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_f(ca_a, ca_b, ddof)],
                )
            }
            DataType::Float64 => {
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_f(ca_a, ca_b, ddof)],
                )
            }
            DataType::Int32 => {
                let ca_a = a.i32().unwrap();
                let ca_b = b.i32().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_i(ca_a, ca_b, ddof)],
                )
            }
            DataType::Int64 => {
                let ca_a = a.i64().unwrap();
                let ca_b = b.i64().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_i(ca_a, ca_b, ddof)],
                )
            }
            DataType::UInt32 => {
                let ca_a = a.u32().unwrap();
                let ca_b = b.u32().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_i(ca_a, ca_b, ddof)],
                )
            }
            DataType::UInt64 => {
                let ca_a = a.u64().unwrap();
                let ca_b = b.u64().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_i(ca_a, ca_b, ddof)],
                )
            }
            _ => {
                let a = a.cast(&DataType::Float64)?;
                let b = b.cast(&DataType::Float64)?;
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_f(ca_a, ca_b, ddof)],
                )
            }
        };
        Ok(Some(s))
    };
    apply_binary(
        a,
        b,
        function,
        GetOutput::map_dtype(|dt| {
            if matches!(dt, DataType::Float32) {
                DataType::Float32
            } else {
                DataType::Float64
            }
        }),
    )
    .with_function_options(|mut options| {
        options.auto_explode = true;
        options.fmt_str = "pearson_corr";
        options
    })
}

/// Compute the spearman rank correlation between two columns.
/// Missing data will be excluded from the computation.
/// # Arguments
/// * ddof
///     Delta degrees of freedom
/// * propagate_nans
///     If `true` any `NaN` encountered will lead to `NaN` in the output.
///     If to `false` then `NaN` are regarded as larger than any finite number
///     and thus lead to the highest rank.
#[cfg(all(feature = "rank", feature = "propagate_nans"))]
pub fn spearman_rank_corr(a: Expr, b: Expr, ddof: u8, propagate_nans: bool) -> Expr {
    use polars_core::utils::coalesce_nulls_series;
    use polars_ops::prelude::nan_propagating_aggregate::nan_max_s;

    let function = move |a: Series, b: Series| {
        let (a, b) = coalesce_nulls_series(&a, &b);

        let name = "spearman_rank_correlation";
        if propagate_nans && a.dtype().is_float() {
            for s in [&a, &b] {
                if nan_max_s(s, "")
                    .get(0)
                    .unwrap()
                    .extract::<f64>()
                    .unwrap()
                    .is_nan()
                {
                    return Ok(Some(Series::new(name, &[f64::NAN])));
                }
            }
        }

        // drop nulls so that they are excluded
        let a = a.drop_nulls();
        let b = b.drop_nulls();

        let a_idx = a.rank(
            RankOptions {
                method: RankMethod::Min,
                ..Default::default()
            },
            None,
        );
        let b_idx = b.rank(
            RankOptions {
                method: RankMethod::Min,
                ..Default::default()
            },
            None,
        );
        let a_idx = a_idx.idx().unwrap();
        let b_idx = b_idx.idx().unwrap();

        Ok(Some(Series::new(
            name,
            &[polars_core::functions::pearson_corr_i(a_idx, b_idx, ddof)],
        )))
    };

    apply_binary(a, b, function, GetOutput::from_type(DataType::Float64)).with_function_options(
        |mut options| {
            options.auto_explode = true;
            options.fmt_str = "spearman_rank_correlation";
            options
        },
    )
}

/// Find the indexes that would sort these series in order of appearance.
/// That means that the first `Series` will be used to determine the ordering
/// until duplicates are found. Once duplicates are found, the next `Series` will
/// be used and so on.
#[cfg(feature = "arange")]
pub fn arg_sort_by<E: AsRef<[Expr]>>(by: E, descending: &[bool]) -> Expr {
    let e = &by.as_ref()[0];
    let name = expr_output_name(e).unwrap();
    arange(lit(0 as IdxSize), count().cast(IDX_DTYPE), 1)
        .sort_by(by, descending)
        .alias(name.as_ref())
}

#[cfg(all(feature = "concat_str", feature = "strings"))]
/// Horizontally concat string columns in linear time
pub fn concat_str<E: AsRef<[Expr]>>(s: E, separator: &str) -> Expr {
    let input = s.as_ref().to_vec();
    let separator = separator.to_string();

    Expr::Function {
        input,
        function: StringFunction::ConcatHorizontal(separator).into(),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: true,
            auto_explode: true,
            ..Default::default()
        },
    }
}

#[cfg(all(feature = "concat_str", feature = "strings"))]
/// Format the results of an array of expressions using a format string
pub fn format_str<E: AsRef<[Expr]>>(format: &str, args: E) -> PolarsResult<Expr> {
    let mut args: std::collections::VecDeque<Expr> = args.as_ref().to_vec().into();

    // Parse the format string, and separate substrings between placeholders
    let segments: Vec<&str> = format.split("{}").collect();

    polars_ensure!(
        segments.len() - 1 == args.len(),
        ShapeMismatch: "number of placeholders should equal the number of arguments"
    );

    let mut exprs: Vec<Expr> = Vec::new();

    for (i, s) in segments.iter().enumerate() {
        if i > 0 {
            if let Some(arg) = args.pop_front() {
                exprs.push(arg);
            }
        }

        if !s.is_empty() {
            exprs.push(lit(s.to_string()))
        }
    }

    Ok(concat_str(exprs, ""))
}

/// Concat lists entries.
pub fn concat_list<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(s: E) -> PolarsResult<Expr> {
    let s: Vec<_> = s.as_ref().iter().map(|e| e.clone().into()).collect();

    polars_ensure!(!s.is_empty(), ComputeError: "`concat_list` needs one or more expressions");

    Ok(Expr::Function {
        input: s,
        function: FunctionExpr::ListExpr(ListFunction::Concat),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            input_wildcard_expansion: true,
            fmt_str: "concat_list",
            ..Default::default()
        },
    })
}

#[cfg(feature = "arange")]
fn arange_impl<T>(start: T::Native, end: T::Native, step: i64) -> PolarsResult<Option<Series>>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
    std::ops::Range<T::Native>: Iterator<Item = T::Native>,
    std::ops::RangeInclusive<T::Native>: DoubleEndedIterator<Item = T::Native>,
{
    let mut ca = match step {
        1 => ChunkedArray::<T>::from_iter_values("arange", start..end),
        2.. => ChunkedArray::<T>::from_iter_values("arange", (start..end).step_by(step as usize)),
        _ => {
            polars_ensure!(start > end, InvalidOperation: "range must be decreasing if 'step' is negative");
            ChunkedArray::<T>::from_iter_values(
                "arange",
                (end..=start).rev().step_by(step.unsigned_abs() as usize),
            )
        }
    };
    let is_sorted = if end < start {
        IsSorted::Descending
    } else {
        IsSorted::Ascending
    };
    ca.set_sorted_flag(is_sorted);
    Ok(Some(ca.into_series()))
}

// TODO! rewrite this with the apply_private architecture
/// Create list entries that are range arrays
/// - if `start` and `end` are a column, every element will expand into an array in a list column.
/// - if `start` and `end` are literals the output will be of `Int64`.
#[cfg(feature = "arange")]
pub fn arange(start: Expr, end: Expr, step: i64) -> Expr {
    let has_col_without_agg = |e: &Expr| {
        has_expr(e, |ae| matches!(ae, Expr::Column(_)))
            &&
            // check if there is no aggregation
            !has_expr(e, |ae| {
                matches!(
                    ae,
                    Expr::Agg(_)
                        | Expr::Count
                        | Expr::AnonymousFunction {
                            options: FunctionOptions {
                                collect_groups: ApplyOptions::ApplyGroups,
                                ..
                            },
                            ..
                        }
                        | Expr::Function {
                            options: FunctionOptions {
                                collect_groups: ApplyOptions::ApplyGroups,
                                ..
                            },
                            ..
                        },
                )
            })
    };
    let has_lit = |e: &Expr| {
        (matches!(e, Expr::Literal(_)) && !matches!(e, Expr::Literal(LiteralValue::Series(_))))
    };

    let any_column_no_agg = has_col_without_agg(&start) || has_col_without_agg(&end);
    let literal_start = has_lit(&start);
    let literal_end = has_lit(&end);

    if (literal_start || literal_end) && !any_column_no_agg {
        let f = move |sa: Series, sb: Series| {
            polars_ensure!(step != 0, InvalidOperation: "step must not be zero");

            match sa.dtype() {
                dt if dt == &IDX_DTYPE => {
                    let start = sa
                        .idx()?
                        .get(0)
                        .ok_or_else(|| polars_err!(NoData: "no data in `start` evaluation"))?;
                    let sb = sb.cast(&IDX_DTYPE)?;
                    let end = sb
                        .idx()?
                        .get(0)
                        .ok_or_else(|| polars_err!(NoData: "no data in `end` evaluation"))?;
                    #[cfg(feature = "bigidx")]
                    {
                        arange_impl::<UInt64Type>(start, end, step)
                    }
                    #[cfg(not(feature = "bigidx"))]
                    {
                        arange_impl::<UInt32Type>(start, end, step)
                    }
                }
                _ => {
                    let sa = sa.cast(&DataType::Int64)?;
                    let sb = sb.cast(&DataType::Int64)?;
                    let start = sa
                        .i64()?
                        .get(0)
                        .ok_or_else(|| polars_err!(NoData: "no data in `start` evaluation"))?;
                    let end = sb
                        .i64()?
                        .get(0)
                        .ok_or_else(|| polars_err!(NoData: "no data in `end` evaluation"))?;
                    arange_impl::<Int64Type>(start, end, step)
                }
            }
        };
        apply_binary(
            start,
            end,
            f,
            GetOutput::map_field(|input| {
                let dtype = if input.data_type() == &IDX_DTYPE {
                    IDX_DTYPE
                } else {
                    DataType::Int64
                };
                Field::new("arange", dtype)
            }),
        )
    } else {
        let f = move |sa: Series, sb: Series| {
            polars_ensure!(step != 0, InvalidOperation: "step must not be zero");
            let mut sa = sa.cast(&DataType::Int64)?;
            let mut sb = sb.cast(&DataType::Int64)?;

            if sa.len() != sb.len() {
                if sa.len() == 1 {
                    sa = sa.new_from_index(0, sb.len())
                } else if sb.len() == 1 {
                    sb = sb.new_from_index(0, sa.len())
                } else {
                    polars_bail!(
                        ComputeError:
                        "lengths of `start`: {} and `end`: {} arguments `\
                        cannot be matched in the `arange` expression",
                        sa.len(), sb.len()
                    );
                }
            }

            let start = sa.i64()?;
            let end = sb.i64()?;
            let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
                "arange",
                start.len(),
                start.len() * 3,
                DataType::Int64,
            );

            for (opt_start, opt_end) in start.into_iter().zip(end.into_iter()) {
                match (opt_start, opt_end) {
                    (Some(start_v), Some(end_v)) => match step {
                        1 => {
                            builder.append_iter_values(start_v..end_v);
                        }
                        2.. => {
                            builder.append_iter_values((start_v..end_v).step_by(step as usize));
                        }
                        _ => {
                            polars_ensure!(start_v > end_v, InvalidOperation: "range must be decreasing if 'step' is negative");
                            builder.append_iter_values(
                                (end_v..=start_v)
                                    .rev()
                                    .step_by(step.unsigned_abs() as usize),
                            )
                        }
                    },
                    _ => builder.append_null(),
                }
            }

            Ok(Some(builder.finish().into_series()))
        };
        apply_binary(
            start,
            end,
            f,
            GetOutput::map_field(|_| Field::new("arange", DataType::List(DataType::Int64.into()))),
        )
    }
}

macro_rules! impl_unit_setter {
    ($fn_name:ident($field:ident)) => {
        #[doc = concat!("Set the ", stringify!($field))]
        pub fn $fn_name(mut self, n: Expr) -> Self {
            self.$field = n.into();
            self
        }
    };
}

/// Arguments used by [`datetime`] in order to produce an `Expr` of `Datetime`
///
/// Construct a `DatetimeArgs` with `DatetimeArgs::new(y, m, d)`. This will set the other time units to `lit(0)`. You
/// can then set the other fields with the `with_*` methods, or use `with_hms` to set `hour`, `minute`, and `second` all
/// at once.
///
/// # Examples
/// ```
/// // construct a DatetimeArgs set to July 20, 1969 at 20:17
/// let args = DatetimeArgs::new(lit(1969), lit(7), lit(20)).with_hms(lit(20), lit(17), lit(0));
/// // or
/// let args = DatetimeArgs::new(lit(1969), lit(7), lit(20)).with_hour(lit(20)).with_minute(lit(17));
///
/// // construct a DatetimeArgs using existing columns
/// let args = DatetimeArgs::new(lit(2023), col("month"), col("day"));
/// ```
#[derive(Debug, Clone)]
pub struct DatetimeArgs {
    pub year: Expr,
    pub month: Expr,
    pub day: Expr,
    pub hour: Expr,
    pub minute: Expr,
    pub second: Expr,
    pub microsecond: Expr,
}

impl DatetimeArgs {
    /// Construct a new `DatetimeArgs` set to `year`, `month`, `day`
    ///
    /// Other fields default to `lit(0)`. Use the `with_*` methods to set them.
    pub fn new(year: Expr, month: Expr, day: Expr) -> Self {
        Self {
            year,
            month,
            day,
            hour: lit(0),
            minute: lit(0),
            second: lit(0),
            microsecond: lit(0),
        }
    }

    /// Set `hour`, `minute`, and `second`
    ///
    /// Equivalent to
    /// ```ignore
    /// self.with_hour(hour)
    ///     .with_minute(minute)
    ///     .with_second(second)
    /// ```
    pub fn with_hms(self, hour: Expr, minute: Expr, second: Expr) -> Self {
        Self {
            hour,
            minute,
            second,
            ..self
        }
    }

    impl_unit_setter!(with_year(year));
    impl_unit_setter!(with_month(month));
    impl_unit_setter!(with_day(day));
    impl_unit_setter!(with_hour(hour));
    impl_unit_setter!(with_minute(minute));
    impl_unit_setter!(with_second(second));
    impl_unit_setter!(with_microsecond(microsecond));
}

/// Construct a column of `Datetime` from the provided [`DatetimeArgs`].
#[cfg(feature = "temporal")]
pub fn datetime(args: DatetimeArgs) -> Expr {
    use polars_core::export::chrono::NaiveDate;
    use polars_core::utils::CustomIterTools;

    let year = args.year;
    let month = args.month;
    let day = args.day;
    let hour = args.hour;
    let minute = args.minute;
    let second = args.second;
    let microsecond = args.microsecond;

    let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
        assert_eq!(s.len(), 7);
        let max_len = s.iter().map(|s| s.len()).max().unwrap();
        let mut year = s[0].cast(&DataType::Int32)?;
        if year.len() < max_len {
            year = year.new_from_index(0, max_len)
        }
        let year = year.i32()?;
        let mut month = s[1].cast(&DataType::UInt32)?;
        if month.len() < max_len {
            month = month.new_from_index(0, max_len);
        }
        let month = month.u32()?;
        let mut day = s[2].cast(&DataType::UInt32)?;
        if day.len() < max_len {
            day = day.new_from_index(0, max_len);
        }
        let day = day.u32()?;
        let mut hour = s[3].cast(&DataType::UInt32)?;
        if hour.len() < max_len {
            hour = hour.new_from_index(0, max_len);
        }
        let hour = hour.u32()?;

        let mut minute = s[4].cast(&DataType::UInt32)?;
        if minute.len() < max_len {
            minute = minute.new_from_index(0, max_len);
        }
        let minute = minute.u32()?;

        let mut second = s[5].cast(&DataType::UInt32)?;
        if second.len() < max_len {
            second = second.new_from_index(0, max_len);
        }
        let second = second.u32()?;

        let mut microsecond = s[6].cast(&DataType::UInt32)?;
        if microsecond.len() < max_len {
            microsecond = microsecond.new_from_index(0, max_len);
        }
        let microsecond = microsecond.u32()?;

        let ca: Int64Chunked = year
            .into_iter()
            .zip(month.into_iter())
            .zip(day.into_iter())
            .zip(hour.into_iter())
            .zip(minute.into_iter())
            .zip(second.into_iter())
            .zip(microsecond.into_iter())
            .map(|((((((y, m), d), h), mnt), s), us)| {
                if let (Some(y), Some(m), Some(d), Some(h), Some(mnt), Some(s), Some(us)) =
                    (y, m, d, h, mnt, s, us)
                {
                    NaiveDate::from_ymd_opt(y, m, d)
                        .and_then(|nd| nd.and_hms_micro_opt(h, mnt, s, us))
                        .map(|ndt| ndt.timestamp_micros())
                } else {
                    None
                }
            })
            .collect_trusted();

        Ok(Some(
            ca.into_datetime(TimeUnit::Microseconds, None).into_series(),
        ))
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: vec![year, month, day, hour, minute, second, microsecond],
        function,
        output_type: GetOutput::from_type(DataType::Datetime(TimeUnit::Microseconds, None)),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: true,
            fmt_str: "datetime",
            ..Default::default()
        },
    }
    .alias("datetime")
}

/// Arguments used by [`duration`] in order to produce an `Expr` of `Duration`
///
/// To construct a `DurationArgs`, use struct literal syntax with `..Default::default()` to leave unspecified fields at
/// their default value of `lit(0)`, as demonstrated below.
///
/// ```
/// let args = DurationArgs {
///     days: lit(5),
///     hours: col("num_hours"),
///     minutes: col("num_minutes"),
///     ..Default::default()  // other fields are lit(0)
/// };
/// ```
/// If you prefer builder syntax, `with_*` methods are also available.
/// ```
/// let args = DurationArgs::new().with_weeks(lit(42)).with_hours(lit(84));
/// ```
#[derive(Debug, Clone)]
pub struct DurationArgs {
    pub weeks: Expr,
    pub days: Expr,
    pub hours: Expr,
    pub minutes: Expr,
    pub seconds: Expr,
    pub milliseconds: Expr,
    pub microseconds: Expr,
    pub nanoseconds: Expr,
}

impl Default for DurationArgs {
    fn default() -> Self {
        Self {
            weeks: lit(0),
            days: lit(0),
            hours: lit(0),
            minutes: lit(0),
            seconds: lit(0),
            milliseconds: lit(0),
            microseconds: lit(0),
            nanoseconds: lit(0),
        }
    }
}

impl DurationArgs {
    /// Create a new `DurationArgs` with all fields set to `lit(0)`. Use the `with_*` methods to set the fields.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `hours`, `minutes`, and `seconds`
    ///
    /// Equivalent to
    /// ```ignore
    /// self.with_hours(hours)
    ///     .with_minutes(minutes)
    ///     .with_seconds(seconds)
    /// ```.
    pub fn with_hms(self, hours: Expr, minutes: Expr, seconds: Expr) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            ..self
        }
    }

    /// Set `milliseconds`, `microseconds`, and `nanoseconds`
    ///
    /// Equivalent to
    /// ```ignore
    /// self.with_milliseconds(milliseconds)
    ///     .with_microseconds(microseconds)
    ///     .with_nanoseconds(nanoseconds)
    /// ```
    pub fn with_fractional_seconds(
        self,
        milliseconds: Expr,
        microseconds: Expr,
        nanoseconds: Expr,
    ) -> Self {
        Self {
            milliseconds,
            microseconds,
            nanoseconds,
            ..self
        }
    }

    impl_unit_setter!(with_weeks(weeks));
    impl_unit_setter!(with_days(days));
    impl_unit_setter!(with_hours(hours));
    impl_unit_setter!(with_minutes(minutes));
    impl_unit_setter!(with_seconds(seconds));
    impl_unit_setter!(with_milliseconds(milliseconds));
    impl_unit_setter!(with_microseconds(microseconds));
    impl_unit_setter!(with_nanoseconds(nanoseconds));
}

/// Construct a column of `Duration` from the provided [`DurationArgs`]
#[cfg(feature = "temporal")]
pub fn duration(args: DurationArgs) -> Expr {
    let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
        assert_eq!(s.len(), 8);
        if s.iter().any(|s| s.is_empty()) {
            return Ok(Some(Series::new_empty(
                s[0].name(),
                &DataType::Duration(TimeUnit::Nanoseconds),
            )));
        }

        let days = s[0].cast(&DataType::Int64).unwrap();
        let seconds = s[1].cast(&DataType::Int64).unwrap();
        let mut nanoseconds = s[2].cast(&DataType::Int64).unwrap();
        let microseconds = s[3].cast(&DataType::Int64).unwrap();
        let milliseconds = s[4].cast(&DataType::Int64).unwrap();
        let minutes = s[5].cast(&DataType::Int64).unwrap();
        let hours = s[6].cast(&DataType::Int64).unwrap();
        let weeks = s[7].cast(&DataType::Int64).unwrap();

        let max_len = s.iter().map(|s| s.len()).max().unwrap();

        let condition = |s: &Series| {
            // check if not literal 0 || full column
            (s.len() != max_len && s.get(0).unwrap() != AnyValue::Int64(0)) || s.len() == max_len
        };

        if nanoseconds.len() != max_len {
            nanoseconds = nanoseconds.new_from_index(0, max_len);
        }
        if condition(&microseconds) {
            nanoseconds = nanoseconds + (microseconds * 1_000);
        }
        if condition(&milliseconds) {
            nanoseconds = nanoseconds + (milliseconds * 1_000_000);
        }
        if condition(&seconds) {
            nanoseconds = nanoseconds + (seconds * NANOSECONDS);
        }
        if condition(&days) {
            nanoseconds = nanoseconds + (days * NANOSECONDS * SECONDS_IN_DAY);
        }
        if condition(&minutes) {
            nanoseconds = nanoseconds + minutes * NANOSECONDS * 60;
        }
        if condition(&hours) {
            nanoseconds = nanoseconds + hours * NANOSECONDS * 60 * 60;
        }
        if condition(&weeks) {
            nanoseconds = nanoseconds + weeks * NANOSECONDS * SECONDS_IN_DAY * 7;
        }

        nanoseconds
            .cast(&DataType::Duration(TimeUnit::Nanoseconds))
            .map(Some)
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: vec![
            args.days,
            args.seconds,
            args.nanoseconds,
            args.microseconds,
            args.milliseconds,
            args.minutes,
            args.hours,
            args.weeks,
        ],
        function,
        output_type: GetOutput::from_type(DataType::Duration(TimeUnit::Nanoseconds)),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: true,
            fmt_str: "duration",
            ..Default::default()
        },
    }
    .alias("duration")
}

/// Create a Column Expression based on a column name.
///
/// # Arguments
///
/// * `name` - A string slice that holds the name of the column. If a column with this name does not exist when the
///   LazyFrame is collected, an error is returned.
///
/// # Examples
///
/// ```ignore
/// // select a column name
/// col("foo")
/// ```
///
/// ```ignore
/// // select all columns by using a wildcard
/// col("*")
/// ```
///
/// ```ignore
/// // select specific column by writing a regular expression that starts with `^` and ends with `$`
/// // only if regex features is activated
/// col("^foo.*$")
/// ```
pub fn col(name: &str) -> Expr {
    match name {
        "*" => Expr::Wildcard,
        _ => Expr::Column(Arc::from(name)),
    }
}

/// Selects all columns. Shorthand for `col("*")`.
pub fn all() -> Expr {
    Expr::Wildcard
}

/// Select multiple columns by name.
pub fn cols<I: IntoVec<String>>(names: I) -> Expr {
    let names = names.into_vec();
    Expr::Columns(names)
}

/// Select multiple columns by dtype.
pub fn dtype_col(dtype: &DataType) -> Expr {
    Expr::DtypeColumn(vec![dtype.clone()])
}

/// Select multiple columns by dtype.
pub fn dtype_cols<DT: AsRef<[DataType]>>(dtype: DT) -> Expr {
    let dtypes = dtype.as_ref().to_vec();
    Expr::DtypeColumn(dtypes)
}

/// Sum all the values in the column named `name`. Shorthand for `col(name).sum()`.
pub fn sum(name: &str) -> Expr {
    col(name).sum()
}

/// Find the minimum of all the values in the column named `name`. Shorthand for `col(name).min()`.
pub fn min(name: &str) -> Expr {
    col(name).min()
}

/// Find the maximum of all the values in the column named `name`. Shorthand for `col(name).max()`.
pub fn max(name: &str) -> Expr {
    col(name).max()
}

/// Find the mean of all the values in the column named `name`. Shorthand for `col(name).mean()`.
pub fn mean(name: &str) -> Expr {
    col(name).mean()
}

/// Find the mean of all the values in the column named `name`. Alias for [`mean`].
pub fn avg(name: &str) -> Expr {
    col(name).mean()
}

/// Find the median of all the values in the column named `name`. Shorthand for `col(name).median()`.
pub fn median(name: &str) -> Expr {
    col(name).median()
}

/// Find a specific quantile of all the values in the column named `name`.
pub fn quantile(name: &str, quantile: Expr, interpol: QuantileInterpolOptions) -> Expr {
    col(name).quantile(quantile, interpol)
}

macro_rules! prepare_binary_function {
    ($f:ident) => {
        move |s: &mut [Series]| {
            let s0 = std::mem::take(&mut s[0]);
            let s1 = std::mem::take(&mut s[1]);

            $f(s0, s1)
        }
    };
}

/// Apply a closure on the two columns that are evaluated from `Expr` a and `Expr` b.
///
/// The closure takes two arguments, each a `Series`. `output_type` must be the output dtype of the resulting `Series`.
pub fn map_binary<F: 'static>(a: Expr, b: Expr, f: F, output_type: GetOutput) -> Expr
where
    F: Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync,
{
    let function = prepare_binary_function!(f);
    a.map_many(function, &[b], output_type)
}

/// Like [`map_binary`], but used in a groupby-aggregation context.
///
/// See [`Expr::apply`] for the difference between [`map`](Expr::map) and [`apply`](Expr::apply).
pub fn apply_binary<F: 'static>(a: Expr, b: Expr, f: F, output_type: GetOutput) -> Expr
where
    F: Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync,
{
    let function = prepare_binary_function!(f);
    a.apply_many(function, &[b], output_type)
}

#[cfg(feature = "dtype-struct")]
fn cumfold_dtype() -> GetOutput {
    GetOutput::map_fields(|fields| {
        let mut st = fields[0].dtype.clone();
        for fld in &fields[1..] {
            st = get_supertype(&st, &fld.dtype).unwrap();
        }
        Field::new(
            &fields[0].name,
            DataType::Struct(
                fields
                    .iter()
                    .map(|fld| Field::new(fld.name(), st.clone()))
                    .collect(),
            ),
        )
    })
}

/// Accumulate over multiple columns horizontally / row wise.
pub fn fold_exprs<F: 'static, E: AsRef<[Expr]>>(acc: Expr, f: F, exprs: E) -> Expr
where
    F: Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync + Clone,
{
    let mut exprs = exprs.as_ref().to_vec();
    exprs.push(acc);

    let function = SpecialEq::new(Arc::new(move |series: &mut [Series]| {
        let mut series = series.to_vec();
        let mut acc = series.pop().unwrap();

        for s in series {
            if let Some(a) = f(acc.clone(), s)? {
                acc = a
            }
        }
        Ok(Some(acc))
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: exprs,
        function,
        output_type: GetOutput::super_type(),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            input_wildcard_expansion: true,
            auto_explode: true,
            fmt_str: "fold",
            ..Default::default()
        },
    }
}

/// Analogous to [`Iterator::reduce`](std::iter::Iterator::reduce).
///
/// An accumulator is initialized to the series given by the first expression in `exprs`, and then each subsequent value
/// of the accumulator is computed from `f(acc, next_expr_series)`. If `exprs` is empty, an error is returned when
/// `collect` is called.
pub fn reduce_exprs<F: 'static, E: AsRef<[Expr]>>(f: F, exprs: E) -> Expr
where
    F: Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync + Clone,
{
    let exprs = exprs.as_ref().to_vec();

    let function = SpecialEq::new(Arc::new(move |series: &mut [Series]| {
        let mut s_iter = series.iter();

        match s_iter.next() {
            Some(acc) => {
                let mut acc = acc.clone();

                for s in s_iter {
                    if let Some(a) = f(acc.clone(), s.clone())? {
                        acc = a
                    }
                }
                Ok(Some(acc))
            }
            None => Err(polars_err!(ComputeError: "`reduce` did not have any expressions to fold")),
        }
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: exprs,
        function,
        output_type: GetOutput::super_type(),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            input_wildcard_expansion: true,
            auto_explode: true,
            fmt_str: "reduce",
            ..Default::default()
        },
    }
}

/// Accumulate over multiple columns horizontally / row wise.
#[cfg(feature = "dtype-struct")]
pub fn cumreduce_exprs<F: 'static, E: AsRef<[Expr]>>(f: F, exprs: E) -> Expr
where
    F: Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync + Clone,
{
    let exprs = exprs.as_ref().to_vec();

    let function = SpecialEq::new(Arc::new(move |series: &mut [Series]| {
        let mut s_iter = series.iter();

        match s_iter.next() {
            Some(acc) => {
                let mut acc = acc.clone();
                let mut result = vec![acc.clone()];

                for s in s_iter {
                    let name = s.name().to_string();
                    if let Some(a) = f(acc.clone(), s.clone())? {
                        acc = a;
                    }
                    acc.rename(&name);
                    result.push(acc.clone());
                }

                StructChunked::new(acc.name(), &result).map(|ca| Some(ca.into_series()))
            }
            None => Err(polars_err!(ComputeError: "`reduce` did not have any expressions to fold")),
        }
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: exprs,
        function,
        output_type: cumfold_dtype(),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            input_wildcard_expansion: true,
            auto_explode: true,
            fmt_str: "cumreduce",
            ..Default::default()
        },
    }
}

/// Accumulate over multiple columns horizontally / row wise.
#[cfg(feature = "dtype-struct")]
pub fn cumfold_exprs<F: 'static, E: AsRef<[Expr]>>(
    acc: Expr,
    f: F,
    exprs: E,
    include_init: bool,
) -> Expr
where
    F: Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync + Clone,
{
    let mut exprs = exprs.as_ref().to_vec();
    exprs.push(acc);

    let function = SpecialEq::new(Arc::new(move |series: &mut [Series]| {
        let mut series = series.to_vec();
        let mut acc = series.pop().unwrap();

        let mut result = vec![];
        if include_init {
            result.push(acc.clone())
        }

        for s in series {
            let name = s.name().to_string();
            if let Some(a) = f(acc.clone(), s)? {
                acc = a;
                acc.rename(&name);
                result.push(acc.clone());
            }
        }

        StructChunked::new(acc.name(), &result).map(|ca| Some(ca.into_series()))
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: exprs,
        function,
        output_type: cumfold_dtype(),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            input_wildcard_expansion: true,
            auto_explode: true,
            fmt_str: "cumfold",
            ..Default::default()
        },
    }
}

/// Create a new column with the the sum of the values in each row.
///
/// The name of the resulting column will be `"sum"`; use [`alias`](Expr::alias) to choose a different name.
pub fn sum_exprs<E: AsRef<[Expr]>>(exprs: E) -> Expr {
    let mut exprs = exprs.as_ref().to_vec();
    let func = |s1, s2| Ok(Some(&s1 + &s2));
    let init = match exprs.pop() {
        Some(e) => e,
        // use u32 as that is not cast to float as eagerly
        _ => lit(0u32),
    };
    fold_exprs(init, func, exprs).alias("sum")
}

/// Create a new column with the the maximum value per row.
///
/// The name of the resulting column will be `"max"`; use [`alias`](Expr::alias) to choose a different name.
pub fn max_exprs<E: AsRef<[Expr]>>(exprs: E) -> Expr {
    let exprs = exprs.as_ref().to_vec();
    if exprs.is_empty() {
        return Expr::Columns(Vec::new());
    }
    let func = |s1, s2| {
        let df = DataFrame::new_no_checks(vec![s1, s2]);
        df.hmax()
    };
    reduce_exprs(func, exprs).alias("max")
}

/// Create a new column with the the minimum value per row.
///
/// The name of the resulting column will be `"min"`; use [`alias`](Expr::alias) to choose a different name.
pub fn min_exprs<E: AsRef<[Expr]>>(exprs: E) -> Expr {
    let exprs = exprs.as_ref().to_vec();
    if exprs.is_empty() {
        return Expr::Columns(Vec::new());
    }
    let func = |s1, s2| {
        let df = DataFrame::new_no_checks(vec![s1, s2]);
        df.hmin()
    };
    reduce_exprs(func, exprs).alias("min")
}

/// Create a new column with the the bitwise-or of the elements in each row.
///
/// The name of the resulting column is arbitrary; use [`alias`](Expr::alias) to choose a different name.
pub fn any_exprs<E: AsRef<[Expr]>>(exprs: E) -> Expr {
    let exprs = exprs.as_ref().to_vec();
    let func = |s1: Series, s2: Series| Ok(Some(s1.bool()?.bitor(s2.bool()?).into_series()));
    fold_exprs(lit(false), func, exprs)
}

/// Create a new column with the the bitwise-and of the elements in each row.
///
/// The name of the resulting column is arbitrary; use [`alias`](Expr::alias) to choose a different name.
pub fn all_exprs<E: AsRef<[Expr]>>(exprs: E) -> Expr {
    let exprs = exprs.as_ref().to_vec();
    let func = |s1: Series, s2: Series| Ok(Some(s1.bool()?.bitand(s2.bool()?).into_series()));
    fold_exprs(lit(true), func, exprs)
}

/// Negates a boolean column.
pub fn not(expr: Expr) -> Expr {
    expr.not()
}

/// A column which is `true` wherever `expr` is null, `false` elsewhere.
pub fn is_null(expr: Expr) -> Expr {
    expr.is_null()
}

/// A column which is `false` wherever `expr` is null, `true` elsewhere.
pub fn is_not_null(expr: Expr) -> Expr {
    expr.is_not_null()
}

/// Casts the column given by `Expr` to a different type.
///
/// Follows the rules of Rust casting, with the exception that integers and floats can be cast to `DataType::Date` and
/// `DataType::DateTime(_, _)`. A column consisting entirely of of `Null` can be cast to any type, regardless of the
/// nominal type of the column.
pub fn cast(expr: Expr, data_type: DataType) -> Expr {
    Expr::Cast {
        expr: Box::new(expr),
        data_type,
        strict: false,
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

/// Take several expressions and collect them into a [`StructChunked`].
#[cfg(feature = "dtype-struct")]
pub fn as_struct(exprs: &[Expr]) -> Expr {
    map_multiple(
        |s| StructChunked::new(s[0].name(), s).map(|ca| Some(ca.into_series())),
        exprs,
        GetOutput::map_fields(|fld| Field::new(fld[0].name(), DataType::Struct(fld.to_vec()))),
    )
    .with_function_options(|mut options| {
        options.input_wildcard_expansion = true;
        options.fmt_str = "as_struct";
        options.pass_name_to_apply = true;
        options
    })
}

/// Create a column of length `n` containing `n` copies of the literal `value`. Generally you won't need this function,
/// as `lit(value)` already represents a column containing only `value` whose length is automatically set to the correct
/// number of rows.
pub fn repeat<L: Literal>(value: L, n: Expr) -> Expr {
    let function = |s: Series, n: Series| {
        let n =
            n.get(0).unwrap().extract::<usize>().ok_or_else(
                || polars_err!(ComputeError: "could not extract a size from {:?}", n),
            )?;
        Ok(Some(s.new_from_index(0, n)))
    };
    apply_binary(lit(value), n, function, GetOutput::same_type()).alias("repeat")
}

#[cfg(feature = "arg_where")]
/// Get the indices where `condition` evaluates `true`.
pub fn arg_where<E: Into<Expr>>(condition: E) -> Expr {
    let condition = condition.into();
    Expr::Function {
        input: vec![condition],
        function: FunctionExpr::ArgWhere,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            ..Default::default()
        },
    }
}

/// Folds the expressions from left to right keeping the first non-null values.
///
/// It is an error to provide an empty `exprs`.
pub fn coalesce(exprs: &[Expr]) -> Expr {
    let input = exprs.to_vec();
    Expr::Function {
        input,
        function: FunctionExpr::Coalesce,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            cast_to_supertypes: true,
            input_wildcard_expansion: true,
            ..Default::default()
        },
    }
}

/// Create a date range from a `start` and `stop` expression.
#[cfg(feature = "temporal")]
pub fn date_range(
    start: Expr,
    end: Expr,
    every: Duration,
    closed: ClosedWindow,
    tz: Option<TimeZone>,
) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::TemporalExpr(TemporalFunction::DateRange { every, closed, tz }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            cast_to_supertypes: true,
            ..Default::default()
        },
    }
}

/// Create a time range, named `name`, from a `start` and `stop` expression.
#[cfg(feature = "temporal")]
pub fn time_range(start: Expr, end: Expr, every: Duration, closed: ClosedWindow) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::TemporalExpr(TemporalFunction::TimeRange { every, closed }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            cast_to_supertypes: false,
            ..Default::default()
        },
    }
}

#[cfg(feature = "rolling_window")]
pub fn rolling_corr(x: Expr, y: Expr, options: RollingCovOptions) -> Expr {
    let x = x.cache();
    let y = y.cache();
    // see: https://github.com/pandas-dev/pandas/blob/v1.5.1/pandas/core/window/rolling.py#L1780-L1804
    let rolling_options = RollingOptions {
        window_size: Duration::new(options.window_size as i64),
        min_periods: options.min_periods as usize,
        ..Default::default()
    };

    let mean_x_y = (x.clone() * y.clone()).rolling_mean(rolling_options.clone());
    let mean_x = x.clone().rolling_mean(rolling_options.clone());
    let mean_y = y.clone().rolling_mean(rolling_options.clone());
    let var_x = x.clone().rolling_var(rolling_options.clone());
    let var_y = y.clone().rolling_var(rolling_options);

    let rolling_options_count = RollingOptions {
        window_size: Duration::new(options.window_size as i64),
        min_periods: 0,
        ..Default::default()
    };
    let ddof = options.ddof as f64;
    let count_x_y = (x + y)
        .is_not_null()
        .cast(DataType::Float64)
        .rolling_sum(rolling_options_count)
        .cache();
    let numerator = (mean_x_y - mean_x * mean_y) * (count_x_y.clone() / (count_x_y - lit(ddof)));
    let denominator = (var_x * var_y).pow(lit(0.5));

    numerator / denominator
}

#[cfg(feature = "rolling_window")]
pub fn rolling_cov(x: Expr, y: Expr, options: RollingCovOptions) -> Expr {
    let x = x.cache();
    let y = y.cache();
    // see: https://github.com/pandas-dev/pandas/blob/91111fd99898d9dcaa6bf6bedb662db4108da6e6/pandas/core/window/rolling.py#L1700
    let rolling_options = RollingOptions {
        window_size: Duration::new(options.window_size as i64),
        min_periods: options.min_periods as usize,
        ..Default::default()
    };

    let mean_x_y = (x.clone() * y.clone()).rolling_mean(rolling_options.clone());
    let mean_x = x.clone().rolling_mean(rolling_options.clone());
    let mean_y = y.clone().rolling_mean(rolling_options);
    let rolling_options_count = RollingOptions {
        window_size: Duration::new(options.window_size as i64),
        min_periods: 0,
        ..Default::default()
    };
    let count_x_y = (x + y)
        .is_not_null()
        .cast(DataType::Float64)
        .rolling_sum(rolling_options_count)
        .cache();

    let ddof = options.ddof as f64;

    (mean_x_y - mean_x * mean_y) * (count_x_y.clone() / (count_x_y - lit(ddof)))
}
