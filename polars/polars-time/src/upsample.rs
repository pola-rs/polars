use polars_core::prelude::*;
use polars_ops::prelude::*;

use crate::prelude::*;

pub trait PolarsUpsample {
    /// Upsample a DataFrame at a regular frequency.
    ///
    /// # Arguments
    /// * `by` - First group by these columns and then upsample for every group
    /// * `time_column` - Will be used to determine a date_range.
    ///                   Note that this column has to be sorted for the output to make sense.
    /// * `every` - interval will start 'every' duration
    /// * `offset` - change the start of the date_range by this offset.
    ///
    /// The `period` and `offset` arguments are created with
    /// the following string language:
    /// - 1ns   (1 nanosecond)
    /// - 1us   (1 microsecond)
    /// - 1ms   (1 millisecond)
    /// - 1s    (1 second)
    /// - 1m    (1 minute)
    /// - 1h    (1 hour)
    /// - 1d    (1 day)
    /// - 1w    (1 week)
    /// - 1mo   (1 calendar month)
    /// - 1y    (1 calendar year)
    /// - 1i    (1 index count)
    /// Or combine them:
    /// "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds
    fn upsample<I: IntoVec<String>>(
        &self,
        by: I,
        time_column: &str,
        every: Duration,
        offset: Duration,
    ) -> PolarsResult<DataFrame>;

    /// Upsample a DataFrame at a regular frequency.
    ///
    /// # Arguments
    /// * `by` - First group by these columns and then upsample for every group
    /// * `time_column` - Will be used to determine a date_range.
    ///                   Note that this column has to be sorted for the output to make sense.
    /// * `every` - interval will start 'every' duration
    /// * `offset` - change the start of the date_range by this offset.
    ///
    /// The `period` and `offset` arguments are created with
    /// the following string language:
    /// - 1ns   (1 nanosecond)
    /// - 1us   (1 microsecond)
    /// - 1ms   (1 millisecond)
    /// - 1s    (1 second)
    /// - 1m    (1 minute)
    /// - 1h    (1 hour)
    /// - 1d    (1 day)
    /// - 1w    (1 week)
    /// - 1mo   (1 calendar month)
    /// - 1y    (1 calendar year)
    /// - 1i    (1 index count)
    /// Or combine them:
    /// "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds
    fn upsample_stable<I: IntoVec<String>>(
        &self,
        by: I,
        time_column: &str,
        every: Duration,
        offset: Duration,
    ) -> PolarsResult<DataFrame>;
}

impl PolarsUpsample for DataFrame {
    fn upsample<I: IntoVec<String>>(
        &self,
        by: I,
        time_column: &str,
        every: Duration,
        offset: Duration,
    ) -> PolarsResult<DataFrame> {
        let by = by.into_vec();
        upsample_impl(self, by, time_column, every, offset, false)
    }

    fn upsample_stable<I: IntoVec<String>>(
        &self,
        by: I,
        time_column: &str,
        every: Duration,
        offset: Duration,
    ) -> PolarsResult<DataFrame> {
        let by = by.into_vec();
        upsample_impl(self, by, time_column, every, offset, true)
    }
}

fn upsample_impl(
    source: &DataFrame,
    by: Vec<String>,
    index_column: &str,
    every: Duration,
    offset: Duration,
    stable: bool,
) -> PolarsResult<DataFrame> {
    let s = source.column(index_column)?;
    if matches!(s.dtype(), DataType::Date) {
        let mut df = source.clone();
        df.try_apply(index_column, |s| {
            s.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
        })
        .unwrap();
        let mut out = upsample_impl(&df, by, index_column, every, offset, stable).unwrap();
        out.try_apply(index_column, |s| s.cast(&DataType::Date))
            .unwrap();
        Ok(out)
    } else if by.is_empty() {
        let index_column = source.column(index_column)?;
        upsample_single_impl(source, index_column, every, offset)
    } else {
        let gb = if stable {
            source.groupby_stable(by)
        } else {
            source.groupby(by)
        };
        // don't parallelize this, this may SO on large data.
        gb?.apply(|df| {
            let index_column = df.column(index_column)?;
            upsample_single_impl(&df, index_column, every, offset)
        })
    }
}

fn upsample_single_impl(
    source: &DataFrame,
    index_column: &Series,
    every: Duration,
    offset: Duration,
) -> PolarsResult<DataFrame> {
    let index_col_name = index_column.name();

    use DataType::*;
    match index_column.dtype() {
        Datetime(tu, tz) => {
            let s = index_column.cast(&DataType::Int64).unwrap();
            let ca = s.i64().unwrap();
            let first = ca.into_iter().flatten().next();
            let last = ca.into_iter().flatten().next_back();
            match (first, last) {
                (Some(first), Some(last)) => {
                    let first = match tu {
                        TimeUnit::Nanoseconds => offset.add_ns(first),
                        TimeUnit::Microseconds => offset.add_us(first),
                        TimeUnit::Milliseconds => offset.add_ms(first),
                    };
                    let range = match tz {
                        #[cfg(feature = "timezones")]
                        Some(tz) => date_range_impl(
                            index_col_name,
                            first,
                            last,
                            every,
                            ClosedWindow::Both,
                            *tu,
                            Some(&"UTC".to_string()),
                        )?
                        .convert_time_zone(tz.clone())?
                        .into_series()
                        .into_frame(),
                        _ => date_range_impl(
                            index_col_name,
                            first,
                            last,
                            every,
                            ClosedWindow::Both,
                            *tu,
                            None,
                        )?
                        .into_series()
                        .into_frame(),
                    };
                    range.join(
                        source,
                        &[index_col_name],
                        &[index_col_name],
                        JoinType::Left,
                        None,
                    )
                }
                _ => Err(PolarsError::ComputeError(
                    "Cannot determine upsample boundaries. All elements are null.".into(),
                )),
            }
        }
        dt => Err(PolarsError::ComputeError(
            format!("upsample not allowed for index_column of dtype {dt:?}").into(),
        )),
    }
}
