//! # Functions
//!
//! Functions on expressions that might be useful.
//!
use std::ops::{BitAnd, BitOr};

use polars_core::export::arrow::temporal_conversions::NANOSECONDS;
use polars_core::utils::arrow::temporal_conversions::SECONDS_IN_DAY;
#[cfg(feature = "rank")]
use polars_core::utils::coalesce_nulls_series;
#[cfg(feature = "dtype-struct")]
use polars_core::utils::get_supertype;

#[cfg(feature = "arg_where")]
use crate::dsl::function_expr::FunctionExpr;
use crate::dsl::function_expr::ListFunction;
#[cfg(feature = "strings")]
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

        let a_idx = a.rank(RankOptions {
            method: RankMethod::Min,
            ..Default::default()
        });
        let b_idx = b.rank(RankOptions {
            method: RankMethod::Min,
            ..Default::default()
        });
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
pub fn arg_sort_by<E: AsRef<[Expr]>>(by: E, reverse: &[bool]) -> Expr {
    let reverse = reverse.to_vec();
    let function = SpecialEq::new(Arc::new(move |by: &mut [Series]| {
        polars_core::functions::arg_sort_by(by, &reverse).map(|ca| Some(ca.into_series()))
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: by.as_ref().to_vec(),
        function,
        output_type: GetOutput::from_type(IDX_DTYPE),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            input_wildcard_expansion: true,
            fmt_str: "arg_sort_by",
            ..Default::default()
        },
    }
}

#[cfg(all(feature = "concat_str", feature = "strings"))]
/// Horizontally concat string columns in linear time
pub fn concat_str<E: AsRef<[Expr]>>(s: E, sep: &str) -> Expr {
    let input = s.as_ref().to_vec();
    let sep = sep.to_string();

    Expr::Function {
        input,
        function: StringFunction::ConcatHorizontal(sep).into(),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
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

    if segments.len() - 1 != args.len() {
        return Err(PolarsError::ShapeMisMatch(
            "number of placeholders should equal the number of arguments".into(),
        ));
    }

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
pub fn concat_lst<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(s: E) -> Expr {
    let s = s.as_ref().iter().map(|e| e.clone().into()).collect();

    Expr::Function {
        input: s,
        function: FunctionExpr::ListExpr(ListFunction::Concat),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            input_wildcard_expansion: true,
            fmt_str: "concat_list",
            ..Default::default()
        },
    }
}

/// Create list entries that are range arrays
/// - if `low` and `high` are a column, every element will expand into an array in a list column.
/// - if `low` and `high` are literals the output will be of `Int64`.
#[cfg(feature = "arange")]
pub fn arange(low: Expr, high: Expr, step: usize) -> Expr {
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

    let any_column_no_agg = has_col_without_agg(&low) || has_col_without_agg(&high);
    let literal_low = has_lit(&low);
    let literal_high = has_lit(&high);

    if (literal_low || literal_high) && !any_column_no_agg {
        let f = move |sa: Series, sb: Series| {
            let sa = sa.cast(&DataType::Int64)?;
            let sb = sb.cast(&DataType::Int64)?;
            let low = sa
                .i64()?
                .get(0)
                .ok_or_else(|| PolarsError::NoData("no data in `low` evaluation".into()))?;
            let high = sb
                .i64()?
                .get(0)
                .ok_or_else(|| PolarsError::NoData("no data in `high` evaluation".into()))?;

            let mut ca = if step > 1 {
                Int64Chunked::from_iter_values("arange", (low..high).step_by(step))
            } else {
                Int64Chunked::from_iter_values("arange", low..high)
            };
            let is_sorted = if high < low {
                IsSorted::Descending
            } else {
                IsSorted::Ascending
            };
            ca.set_sorted_flag(is_sorted);
            Ok(Some(ca.into_series()))
        };
        apply_binary(
            low,
            high,
            f,
            GetOutput::map_field(|_| Field::new("arange", DataType::Int64)),
        )
    } else {
        let f = move |sa: Series, sb: Series| {
            let mut sa = sa.cast(&DataType::Int64)?;
            let mut sb = sb.cast(&DataType::Int64)?;

            if sa.len() != sb.len() {
                if sa.len() == 1 {
                    sa = sa.new_from_index(0, sb.len())
                } else if sb.len() == 1 {
                    sb = sb.new_from_index(0, sa.len())
                } else {
                    let msg = format!("The length of the 'low' and 'high' arguments cannot be matched in the 'arange' expression.. \
                    Length of 'low': {}, length of 'high': {}", sa.len(), sb.len());
                    return Err(PolarsError::ComputeError(msg.into()));
                }
            }

            let low = sa.i64()?;
            let high = sb.i64()?;
            let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
                "arange",
                low.len(),
                low.len() * 3,
                DataType::Int64,
            );

            low.into_iter()
                .zip(high.into_iter())
                .for_each(|(opt_l, opt_h)| match (opt_l, opt_h) {
                    (Some(l), Some(r)) => {
                        if step > 1 {
                            builder.append_iter_values((l..r).step_by(step));
                        } else {
                            builder.append_iter_values(l..r);
                        }
                    }
                    _ => builder.append_null(),
                });

            Ok(Some(builder.finish().into_series()))
        };
        apply_binary(
            low,
            high,
            f,
            GetOutput::map_field(|_| Field::new("arange", DataType::List(DataType::Int64.into()))),
        )
    }
}

#[derive(Default)]
pub struct DatetimeArgs {
    pub year: Expr,
    pub month: Expr,
    pub day: Expr,
    pub hour: Option<Expr>,
    pub minute: Option<Expr>,
    pub second: Option<Expr>,
    pub microsecond: Option<Expr>,
}

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
        input: vec![
            year,
            month,
            day,
            hour.unwrap_or_else(|| lit(0)),
            minute.unwrap_or_else(|| lit(0)),
            second.unwrap_or_else(|| lit(0)),
            microsecond.unwrap_or_else(|| lit(0)),
        ],
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

#[derive(Default)]
pub struct DurationArgs {
    pub days: Option<Expr>,
    pub seconds: Option<Expr>,
    pub nanoseconds: Option<Expr>,
    pub microseconds: Option<Expr>,
    pub milliseconds: Option<Expr>,
    pub minutes: Option<Expr>,
    pub hours: Option<Expr>,
    pub weeks: Option<Expr>,
}

#[cfg(feature = "temporal")]
pub fn duration(args: DurationArgs) -> Expr {
    let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
        assert_eq!(s.len(), 8);
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
            args.days.unwrap_or_else(|| lit(0i64)),
            args.seconds.unwrap_or_else(|| lit(0i64)),
            args.nanoseconds.unwrap_or_else(|| lit(0i64)),
            args.microseconds.unwrap_or_else(|| lit(0i64)),
            args.milliseconds.unwrap_or_else(|| lit(0i64)),
            args.minutes.unwrap_or_else(|| lit(0i64)),
            args.hours.unwrap_or_else(|| lit(0i64)),
            args.weeks.unwrap_or_else(|| lit(0i64)),
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
/// * `name` - A string slice that holds the name of the column
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

/// Selects all columns
pub fn all() -> Expr {
    Expr::Wildcard
}

/// Select multiple columns by name
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

/// Sum all the values in this Expression.
pub fn sum(name: &str) -> Expr {
    col(name).sum()
}

/// Find the minimum of all the values in this Expression.
pub fn min(name: &str) -> Expr {
    col(name).min()
}

/// Find the maximum of all the values in this Expression.
pub fn max(name: &str) -> Expr {
    col(name).max()
}

/// Find the mean of all the values in this Expression.
pub fn mean(name: &str) -> Expr {
    col(name).mean()
}

/// Find the mean of all the values in this Expression.
pub fn avg(name: &str) -> Expr {
    col(name).mean()
}

/// Find the median of all the values in this Expression.
pub fn median(name: &str) -> Expr {
    col(name).median()
}

/// Find a specific quantile of all the values in this Expression.
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
pub fn map_binary<F: 'static>(a: Expr, b: Expr, f: F, output_type: GetOutput) -> Expr
where
    F: Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync,
{
    let function = prepare_binary_function!(f);
    a.map_many(function, &[b], output_type)
}

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
            None => Err(PolarsError::ComputeError(
                "Reduce did not have any expressions to fold".into(),
            )),
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
            None => Err(PolarsError::ComputeError(
                "Reduce did not have any expressions to fold".into(),
            )),
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

/// Get the the sum of the values per row
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

/// Get the the maximum value per row
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

/// Evaluate all the expressions with a bitwise or
pub fn any_exprs<E: AsRef<[Expr]>>(exprs: E) -> Expr {
    let exprs = exprs.as_ref().to_vec();
    let func = |s1: Series, s2: Series| Ok(Some(s1.bool()?.bitor(s2.bool()?).into_series()));
    fold_exprs(lit(false), func, exprs)
}

/// Evaluate all the expressions with a bitwise and
pub fn all_exprs<E: AsRef<[Expr]>>(exprs: E) -> Expr {
    let exprs = exprs.as_ref().to_vec();
    let func = |s1: Series, s2: Series| Ok(Some(s1.bool()?.bitand(s2.bool()?).into_series()));
    fold_exprs(lit(true), func, exprs)
}

/// [Not](Expr::Not) expression.
pub fn not(expr: Expr) -> Expr {
    expr.not()
}

/// [IsNull](Expr::IsNotNull) expression
pub fn is_null(expr: Expr) -> Expr {
    expr.is_null()
}

/// [IsNotNull](Expr::IsNotNull) expression.
pub fn is_not_null(expr: Expr) -> Expr {
    expr.is_not_null()
}

/// [Cast](Expr::Cast) expression.
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
        |s| StructChunked::new("", s).map(|ca| Some(ca.into_series())),
        exprs,
        GetOutput::map_fields(|fld| Field::new(fld[0].name(), DataType::Struct(fld.to_vec()))),
    )
    .with_function_options(|mut options| {
        options.input_wildcard_expansion = true;
        options.fmt_str = "as_struct";
        options
    })
}

/// Repeat a literal `value` `n` times.
pub fn repeat<L: Literal>(value: L, n_times: Expr) -> Expr {
    let function = |s: Series, n: Series| {
        let n = n.get(0).unwrap().extract::<usize>().ok_or_else(|| {
            PolarsError::ComputeError(format!("could not extract a size from {n:?}").into())
        })?;
        Ok(Some(s.new_from_index(0, n)))
    };
    apply_binary(lit(value), n_times, function, GetOutput::same_type())
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
            fmt_str: "arg_where",
            ..Default::default()
        },
    }
}

/// Folds the expressions from left to right keeping the first no null values.
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

///  Create a date range from a `start` and `stop` expression.
#[cfg(feature = "temporal")]
pub fn date_range(
    name: String,
    start: Expr,
    end: Expr,
    every: Duration,
    closed: ClosedWindow,
    tz: Option<TimeZone>,
) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::TemporalExpr(TemporalFunction::DateRange {
            name,
            every,
            closed,
            tz,
        }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            cast_to_supertypes: true,
            ..Default::default()
        },
    }
}
