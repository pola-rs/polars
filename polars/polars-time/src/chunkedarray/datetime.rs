use arrow::array::{Array, PrimitiveArray};
use arrow::compute::cast::{cast, CastOptions};
use arrow::compute::temporal;
use arrow::error::Result as ArrowResult;
use polars_arrow::export::arrow;
use polars_core::prelude::*;

use super::*;

fn cast_and_apply<
    F: Fn(&dyn Array) -> ArrowResult<PrimitiveArray<T::Native>>,
    T: PolarsNumericType,
>(
    ca: &DatetimeChunked,
    func: F,
) -> ChunkedArray<T> {
    let dtype = ca.dtype().to_arrow();
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let arr = cast(
                arr,
                &dtype,
                CastOptions {
                    wrapped: true,
                    partial: false,
                },
            )
            .unwrap();
            Box::from(func(&*arr).unwrap()) as ArrayRef
        })
        .collect();

    unsafe { ChunkedArray::from_chunks(ca.name(), chunks) }
}

pub trait DatetimeMethods: AsDatetime {
    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> Int32Chunked {
        cast_and_apply(self.as_datetime(), temporal::year)
    }

    fn iso_year(&self) -> Int32Chunked {
        let ca = self.as_datetime();
        let f = match ca.time_unit() {
            TimeUnit::Nanoseconds => datetime_to_iso_year_ns,
            TimeUnit::Microseconds => datetime_to_iso_year_us,
            TimeUnit::Milliseconds => datetime_to_iso_year_ms,
        };
        ca.apply_kernel_cast::<Int32Type>(&f)
    }

    /// Extract quarter from underlying NaiveDateTime representation.
    /// Quarters range from 1 to 4.
    fn quarter(&self) -> UInt32Chunked {
        let months = self.month();
        months_to_quarters(months)
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    fn month(&self) -> UInt32Chunked {
        cast_and_apply(self.as_datetime(), temporal::month)
    }

    /// Extract ISO weekday from underlying NaiveDateTime representation.
    /// Returns the weekday number where monday = 1 and sunday = 7
    fn weekday(&self) -> UInt32Chunked {
        cast_and_apply(self.as_datetime(), temporal::weekday)
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    fn week(&self) -> UInt32Chunked {
        cast_and_apply(self.as_datetime(), temporal::iso_week)
    }

    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> UInt32Chunked {
        cast_and_apply(self.as_datetime(), temporal::day)
    }

    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> UInt32Chunked {
        cast_and_apply(self.as_datetime(), temporal::hour)
    }

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> UInt32Chunked {
        cast_and_apply(self.as_datetime(), temporal::minute)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> UInt32Chunked {
        cast_and_apply(self.as_datetime(), temporal::second)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> UInt32Chunked {
        cast_and_apply(self.as_datetime(), temporal::nanosecond)
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal(&self) -> UInt32Chunked {
        let ca = self.as_datetime();
        let f = match ca.time_unit() {
            TimeUnit::Nanoseconds => datetime_to_ordinal_ns,
            TimeUnit::Microseconds => datetime_to_ordinal_us,
            TimeUnit::Milliseconds => datetime_to_ordinal_ms,
        };
        ca.apply_kernel_cast::<UInt32Type>(&f)
    }

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str, tu: TimeUnit) -> DatetimeChunked {
        let func = match tu {
            TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
            TimeUnit::Microseconds => datetime_to_timestamp_us,
            TimeUnit::Milliseconds => datetime_to_timestamp_ms,
        };

        Int64Chunked::from_iter_options(
            name,
            v.iter()
                .map(|s| NaiveDateTime::parse_from_str(s, fmt).ok().map(func)),
        )
        .into_datetime(tu, None)
    }
}

pub trait AsDatetime {
    fn as_datetime(&self) -> &DatetimeChunked;
}

impl AsDatetime for DatetimeChunked {
    fn as_datetime(&self) -> &DatetimeChunked {
        self
    }
}

impl DatetimeMethods for DatetimeChunked {}

#[cfg(test)]
mod test {
    use chrono::NaiveDateTime;

    use super::*;

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
        let dt = DatetimeChunked::from_naive_datetime(
            "name",
            datetimes.iter().copied(),
            TimeUnit::Nanoseconds,
        );
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
