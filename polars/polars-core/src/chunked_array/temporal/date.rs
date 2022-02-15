use super::*;
use crate::prelude::*;
use arrow::temporal_conversions::date32_to_date;
use std::fmt::Write;

pub(crate) fn naive_date_to_date(nd: NaiveDate) -> i32 {
    let nt = NaiveTime::from_hms(0, 0, 0);
    let ndt = NaiveDateTime::new(nd, nt);
    naive_datetime_to_date(ndt)
}

impl DateChunked {
    pub fn as_date_iter(&self) -> impl Iterator<Item = Option<NaiveDate>> + TrustedLen + '_ {
        // safety: we know the iterators len
        unsafe {
            self.downcast_iter()
                .flat_map(|iter| {
                    iter.into_iter()
                        .map(|opt_v| opt_v.copied().map(date32_to_date))
                })
                .trust_my_length(self.len())
        }
    }

    /// Extract month from underlying NaiveDate representation.
    /// Returns the year number in the calendar date.
    pub fn year(&self) -> Int32Chunked {
        self.apply_kernel_cast::<Int32Type>(&date_to_year)
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    pub fn month(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&date_to_month)
    }

    /// Extract weekday from underlying NaiveDate representation.
    /// Returns the weekday number where monday = 0 and sunday = 6
    pub fn weekday(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&date_to_weekday)
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    pub fn week(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&date_to_week)
    }

    /// Extract day from underlying NaiveDate representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    pub fn day(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&date_to_day)
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    pub fn ordinal(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&date_to_ordinal)
    }

    /// Format Date with a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn strftime(&self, fmt: &str) -> Utf8Chunked {
        let date = NaiveDate::from_ymd(2001, 1, 1);
        let fmted = format!("{}", date.format(fmt));

        let mut ca: Utf8Chunked = self.apply_kernel_cast(&|arr| {
            let mut buf = String::new();
            let mut mutarr =
                MutableUtf8Array::with_capacities(arr.len(), arr.len() * fmted.len() + 1);

            for opt in arr.into_iter() {
                match opt {
                    None => mutarr.push_null(),
                    Some(v) => {
                        buf.clear();
                        let datefmt = date32_to_date(*v).format(fmt);
                        write!(buf, "{}", datefmt).unwrap();
                        mutarr.push(Some(&buf))
                    }
                }
            }

            let arr: Utf8Array<i64> = mutarr.into();
            Arc::new(arr)
        });
        ca.rename(self.name());
        ca
    }

    /// Construct a new [`DateChunked`] from an iterator over [`NaiveDate`].
    pub fn from_naive_date<I: IntoIterator<Item = NaiveDate>>(name: &str, v: I) -> Self {
        let unit = v.into_iter().map(naive_date_to_date).collect::<Vec<_>>();
        Int32Chunked::from_vec(name, unit).into()
    }

    /// Construct a new [`DateChunked`] from an iterator over optional [`NaiveDate`].
    pub fn from_naive_date_options<I: IntoIterator<Item = Option<NaiveDate>>>(
        name: &str,
        v: I,
    ) -> Self {
        let unit = v.into_iter().map(|opt| opt.map(naive_date_to_date));
        Int32Chunked::from_iter_options(name, unit).into()
    }

    pub fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self {
        Int32Chunked::from_iter_options(
            name,
            v.iter().map(|s| {
                NaiveDate::parse_from_str(s, fmt)
                    .ok()
                    .as_ref()
                    .map(|v| naive_date_to_date(*v))
            }),
        )
        .into()
    }
}
