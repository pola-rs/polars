#[cfg(feature = "timezones")]
use polars_core::datatypes::time_zone::parse_time_zone;
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_ops::prelude::*;
use polars_ops::series::SeriesMethods;

use crate::prelude::*;

pub trait PolarsUpsample {
    /// Upsample a [`DataFrame`] at a regular frequency.
    ///
    /// # Arguments
    /// * `by` - First group by these columns and then upsample for every group
    /// * `time_column` - Will be used to determine a date_range.
    ///   Note that this column has to be sorted for the output to make sense.
    /// * `every` - interval will start 'every' duration
    /// * `offset` - change the start of the date_range by this offset.
    ///
    /// The `every` and `offset` arguments are created with
    /// the following string language:
    /// - 1ns   (1 nanosecond)
    /// - 1us   (1 microsecond)
    /// - 1ms   (1 millisecond)
    /// - 1s    (1 second)
    /// - 1m    (1 minute)
    /// - 1h    (1 hour)
    /// - 1d    (1 calendar day)
    /// - 1w    (1 calendar week)
    /// - 1mo   (1 calendar month)
    /// - 1q    (1 calendar quarter)
    /// - 1y    (1 calendar year)
    /// - 1i    (1 index count)
    ///
    /// Or combine them:
    /// "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds
    ///
    /// By "calendar day", we mean the corresponding time on the next
    /// day (which may not be 24 hours, depending on daylight savings).
    /// Similarly for "calendar week", "calendar month", "calendar quarter",
    /// and "calendar year".
    fn upsample<I: IntoVec<PlSmallStr>>(
        &self,
        by: I,
        time_column: &str,
        every: Duration,
    ) -> PolarsResult<DataFrame>;

    /// Upsample a [`DataFrame`] at a regular frequency.
    ///
    /// Similar to [`upsample`][PolarsUpsample::upsample], but order of the
    /// DataFrame is maintained when `by` is specified.
    ///
    /// # Arguments
    /// * `by` - First group by these columns and then upsample for every group
    /// * `time_column` - Will be used to determine a date_range.
    ///   Note that this column has to be sorted for the output to make sense.
    /// * `every` - interval will start 'every' duration
    /// * `offset` - change the start of the date_range by this offset.
    ///
    /// The `every` and `offset` arguments are created with
    /// the following string language:
    /// - 1ns   (1 nanosecond)
    /// - 1us   (1 microsecond)
    /// - 1ms   (1 millisecond)
    /// - 1s    (1 second)
    /// - 1m    (1 minute)
    /// - 1h    (1 hour)
    /// - 1d    (1 calendar day)
    /// - 1w    (1 calendar week)
    /// - 1mo   (1 calendar month)
    /// - 1q    (1 calendar quarter)
    /// - 1y    (1 calendar year)
    /// - 1i    (1 index count)
    ///
    /// Or combine them:
    /// "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds
    ///
    /// By "calendar day", we mean the corresponding time on the next
    /// day (which may not be 24 hours, depending on daylight savings).
    /// Similarly for "calendar week", "calendar month", "calendar quarter",
    /// and "calendar year".
    fn upsample_stable<I: IntoVec<PlSmallStr>>(
        &self,
        by: I,
        time_column: &str,
        every: Duration,
    ) -> PolarsResult<DataFrame>;
}

impl PolarsUpsample for DataFrame {
    fn upsample<I: IntoVec<PlSmallStr>>(
        &self,
        by: I,
        time_column: &str,
        every: Duration,
    ) -> PolarsResult<DataFrame> {
        let by = by.into_vec();
        let time_type = self.column(time_column)?.dtype();
        ensure_duration_matches_dtype(every, time_type, "every")?;
        upsample_impl(self, by, time_column, every, false)
    }

    fn upsample_stable<I: IntoVec<PlSmallStr>>(
        &self,
        by: I,
        time_column: &str,
        every: Duration,
    ) -> PolarsResult<DataFrame> {
        let by = by.into_vec();
        let time_type = self.column(time_column)?.dtype();
        ensure_duration_matches_dtype(every, time_type, "every")?;
        upsample_impl(self, by, time_column, every, true)
    }
}

fn upsample_impl(
    source: &DataFrame,
    by: Vec<PlSmallStr>,
    index_column: &str,
    every: Duration,
    stable: bool,
) -> PolarsResult<DataFrame> {
    let s = source.column(index_column)?;
    let original_type = s.dtype();

    let needs_cast = matches!(
        original_type,
        DataType::Date | DataType::UInt32 | DataType::UInt64 | DataType::Int32 | DataType::Int64
    );

    let mut df = source.clone();

    if needs_cast {
        df.try_apply(index_column, |s| match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.cast(&DataType::Datetime(TimeUnit::Microseconds, None)),
            DataType::UInt32 | DataType::UInt64 | DataType::Int32 => s
                .cast(&DataType::Int64)?
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None)),
            DataType::Int64 => s.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None)),
            _ => Ok(s.clone()),
        })?;
    }

    let mut out = upsample_core(&df, by, index_column, every, stable)?;

    if needs_cast {
        out.try_apply(index_column, |s| s.cast(original_type))?;
    }

    Ok(out)
}

fn upsample_core(
    source: &DataFrame,
    by: Vec<PlSmallStr>,
    index_column: &str,
    every: Duration,
    stable: bool,
) -> PolarsResult<DataFrame> {
    if by.is_empty() {
        let index_column = source.column(index_column)?;
        return upsample_single_impl(source, index_column.as_materialized_series(), every);
    }

    let source_schema = source.schema();

    let group_keys_df = source.select(by)?;
    let group_keys_schema = group_keys_df.schema();

    let groups = if stable {
        group_keys_df.group_by_stable(group_keys_schema.iter_names_cloned())
    } else {
        group_keys_df.group_by(group_keys_schema.iter_names_cloned())
    }?
    .into_groups();

    let non_group_keys_df = unsafe {
        source.select_unchecked(
            source_schema
                .iter_names()
                .filter(|name| !group_keys_schema.contains(name.as_str())),
        )?
    };

    let upsample_index_col_idx: Option<usize> = non_group_keys_df.schema().index_of(index_column);

    // don't parallelize this, this may SO on large data.
    let dfs: Vec<DataFrame> = groups
        .iter()
        .map(|g| {
            let first_idx = g.first();

            let mut non_group_keys_df = unsafe { non_group_keys_df.gather_group_unchecked(&g) };

            if let Some(i) = upsample_index_col_idx {
                non_group_keys_df = upsample_single_impl(
                    &non_group_keys_df,
                    non_group_keys_df.columns()[i].as_materialized_series(),
                    every,
                )?
            }

            let mut out = non_group_keys_df;

            let group_keys_df = group_keys_df.new_from_index(first_idx as usize, out.height());

            let out_cols = unsafe { out.columns_mut() };

            out_cols.reserve(group_keys_df.width());
            out_cols.extend(group_keys_df.into_columns());

            Ok(out)
        })
        .collect::<PolarsResult<_>>()?;

    Ok(unsafe {
        accumulate_dataframes_vertical_unchecked(dfs)
            .select_unchecked(source_schema.iter_names())?
            .with_schema(source_schema.clone())
    })
}

fn upsample_single_impl(
    source: &DataFrame,
    index_column: &Series,
    every: Duration,
) -> PolarsResult<DataFrame> {
    index_column.ensure_sorted_arg("upsample")?;
    let index_col_name = index_column.name();

    use DataType::*;
    match index_column.dtype() {
        #[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
        Datetime(tu, tz) => {
            let s = index_column.cast(&Int64).unwrap();
            let ca = s.i64().unwrap();
            let first = ca.iter().flatten().next();
            let last = ca.iter().flatten().next_back();
            match (first, last) {
                (Some(first), Some(last)) => {
                    let tz = match tz {
                        #[cfg(feature = "timezones")]
                        Some(tz) => Some(parse_time_zone(tz)?),
                        _ => None,
                    };
                    let range = datetime_range_impl(
                        index_col_name.clone(),
                        first,
                        last,
                        every,
                        ClosedWindow::Both,
                        *tu,
                        tz.as_ref(),
                    )?
                    .into_series()
                    .into_frame();
                    range.join(
                        source,
                        [index_col_name.clone()],
                        [index_col_name.clone()],
                        JoinArgs::new(JoinType::Left),
                        None,
                    )
                },
                _ => polars_bail!(
                    ComputeError: "cannot determine upsample boundaries: all elements are null"
                ),
            }
        },
        dt => polars_bail!(
            ComputeError: "upsample not allowed for index column of dtype {}", dt,
        ),
    }
}
