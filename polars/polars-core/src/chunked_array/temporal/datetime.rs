use super::*;
use crate::prelude::*;
use arrow::temporal_conversions::timestamp_ns_to_datetime;

impl DatetimeChunked {
    pub fn as_datetime_iter(
        &self,
    ) -> impl Iterator<Item = Option<NaiveDateTime>> + TrustedLen + '_ {
        self.downcast_iter()
            .map(|iter| {
                iter.into_iter()
                    .map(|opt_v| opt_v.copied().map(timestamp_ns_to_datetime))
            })
            .flatten()
            .trust_my_length(self.len())
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    pub fn year(&self) -> Int32Chunked {
        self.apply_kernel_cast::<_, Int32Type>(datetime_to_year)
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    pub fn month(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(datetime_to_month)
    }

    /// Extract weekday from underlying NaiveDateTime representation.
    /// Returns the weekday number where monday = 0 and sunday = 6
    pub fn weekday(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(datetime_to_weekday)
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    pub fn week(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(datetime_to_week)
    }

    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    pub fn day(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(datetime_to_day)
    }

    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    pub fn hour(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(datetime_to_hour)
    }

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    pub fn minute(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(datetime_to_minute)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    pub fn second(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(datetime_to_second)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    pub fn nanosecond(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(datetime_to_nanosecond)
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    pub fn ordinal(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(datetime_to_ordinal)
    }

    /// Format Datetimewith a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn strftime(&self, fmt: &str) -> Utf8Chunked {
        let mut ca: Utf8Chunked = self.apply_kernel_cast(|arr| {
            let arr: Utf8Array<i64> = arr
                .into_iter()
                .map(|opt| opt.map(|v| format!("{}", timestamp_ns_to_datetime(*v).format(fmt))))
                .collect();
            Arc::new(arr)
        });
        ca.rename(self.name());
        ca
    }

    pub fn new_from_naive_datetime(name: &str, v: &[NaiveDateTime]) -> Self {
        let vals = v
            .iter()
            .map(naive_datetime_to_datetime)
            .collect_trusted::<AlignedVec<_>>();
        Int64Chunked::new_from_aligned_vec(name, vals).into()
    }

    pub fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self {
        Int64Chunked::new_from_opt_iter(
            name,
            v.iter().map(|s| {
                NaiveDateTime::parse_from_str(s, fmt)
                    .ok()
                    .as_ref()
                    .map(naive_datetime_to_datetime)
            }),
        )
        .into()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use polars_time::export::chrono::NaiveDateTime;

    #[test]
    fn from_datetime() {
        let datetimes: Vec<_> = [
            "1988-08-25 00:00:16",
            "2015-09-05 23:56:04",
            "2012-12-21 00:00:00",
        ]
        .iter()
        .map(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").unwrap())
        .collect();

        // NOTE: the values are checked and correct.
        let dt = DatetimeChunked::new_from_naive_datetime("name", &datetimes);
        assert_eq!(
            [
                588470416000_000_000,
                1441497364000_000_000,
                1356048000000_000_000
            ],
            dt.cont_slice().unwrap()
        );
    }
}
