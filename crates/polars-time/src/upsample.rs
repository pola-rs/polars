#[cfg(feature = "timezones")]
use polars_core::chunked_array::temporal::parse_time_zone;
use polars_core::prelude::*;
use polars_ops::prelude::*;
use polars_ops::series::SeriesMethods;

use crate::prelude::*;

pub trait PolarsUpsample {
    /// Upsample a [`DataFrame`] at a regular frequency.
    ///
    /// # Arguments
    /// * `by` - First group by these columns and then upsample for every group
    /// * `time_column` - Will be used to determine a date_range.
    ///                   Note that this column has to be sorted for the output to make sense.
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
    fn upsample<I: IntoVec<String>>(
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
    ///                   Note that this column has to be sorted for the output to make sense.
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
    fn upsample_stable<I: IntoVec<String>>(
        &self,
        by: I,
        time_column: &str,
        every: Duration,
    ) -> PolarsResult<DataFrame>;
}

impl PolarsUpsample for DataFrame {
    fn upsample<I: IntoVec<String>>(
        &self,
        by: I,
        time_column: &str,
        every: Duration,
    ) -> PolarsResult<DataFrame> {
        let by = by.into_vec();
        let time_type = self.column(time_column)?.dtype();
        ensure_duration_matches_data_type(every, time_type, "every")?;
        upsample_impl(self, by, time_column, every, false)
    }

    fn upsample_stable<I: IntoVec<String>>(
        &self,
        by: I,
        time_column: &str,
        every: Duration,
    ) -> PolarsResult<DataFrame> {
        let by = by.into_vec();
        let time_type = self.column(time_column)?.dtype();
        ensure_duration_matches_data_type(every, time_type, "every")?;
        upsample_impl(self, by, time_column, every, true)
    }
}

fn upsample_impl(
    source: &DataFrame,
    by: Vec<String>,
    index_column: &str,
    every: Duration,
    stable: bool,
) -> PolarsResult<DataFrame> {
    let s = source.column(index_column)?;
    let time_type = s.dtype();
    if matches!(time_type, DataType::Date) {
        let mut df = source.clone();
        df.apply(index_column, |s| {
            s.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap()
        })
        .unwrap();
        let mut out = upsample_impl(&df, by, index_column, every, stable)?;
        out.apply(index_column, |s| s.cast(time_type).unwrap())
            .unwrap();
        Ok(out)
    } else if matches!(
        time_type,
        DataType::UInt32 | DataType::UInt64 | DataType::Int32
    ) {
        let mut df = source.clone();

        df.apply(index_column, |s| {
            s.cast(&DataType::Int64)
                .unwrap()
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                .unwrap()
        })
        .unwrap();
        let mut out = upsample_impl(&df, by, index_column, every, stable)?;
        out.apply(index_column, |s| s.cast(time_type).unwrap())
            .unwrap();
        Ok(out)
    } else if matches!(time_type, DataType::Int64) {
        let mut df = source.clone();
        df.apply(index_column, |s| {
            s.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                .unwrap()
        })
        .unwrap();
        let mut out = upsample_impl(&df, by, index_column, every, stable)?;
        out.apply(index_column, |s| s.cast(time_type).unwrap())
            .unwrap();
        Ok(out)
    } else if by.is_empty() {
        let index_column = source.column(index_column)?;
        upsample_single_impl(source, index_column, every)
    } else {
        let gb = if stable {
            source.group_by_stable(by)
        } else {
            source.group_by(by)
        };
        // don't parallelize this, this may SO on large data.
        gb?.apply(|df| {
            let index_column = df.column(index_column)?;
            upsample_single_impl(&df, index_column, every)
        })
    }
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
                        index_col_name,
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
                        &[index_col_name],
                        &[index_col_name],
                        JoinArgs::new(JoinType::Left),
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
