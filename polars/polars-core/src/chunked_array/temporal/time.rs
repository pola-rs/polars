use super::*;
use crate::chunked_array::kernels::temporal::{
    time_to_hour, time_to_minute, time_to_nanosecond, time_to_second,
};
use crate::prelude::*;
use arrow::temporal_conversions::{time64ns_to_time, NANOSECONDS};
use chrono::Timelike;
use std::fmt::Write;

pub(crate) fn time_to_time64ns(time: &NaiveTime) -> i64 {
    time.second() as i64 * NANOSECONDS + time.nanosecond() as i64
}

impl TimeChunked {
    pub fn as_time_iter(&self) -> impl Iterator<Item = Option<NaiveTime>> + TrustedLen + '_ {
        // we know the iterators len
        unsafe {
            self.downcast_iter()
                .flat_map(|iter| {
                    iter.into_iter()
                        .map(|opt_v| opt_v.copied().map(time64ns_to_time))
                })
                .trust_my_length(self.len())
        }
    }

    /// Format Date with a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn strftime(&self, fmt: &str) -> Utf8Chunked {
        let time = NaiveTime::from_hms(0, 0, 0);
        let fmted = format!("{}", time.format(fmt));

        let mut ca: Utf8Chunked = self.apply_kernel_cast(&|arr| {
            let mut buf = String::new();
            let mut mutarr =
                MutableUtf8Array::with_capacities(arr.len(), arr.len() * fmted.len() + 1);

            for opt in arr.into_iter() {
                match opt {
                    None => mutarr.push_null(),
                    Some(v) => {
                        buf.clear();
                        let timefmt = time64ns_to_time(*v).format(fmt);
                        write!(buf, "{}", timefmt).unwrap();
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

    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    pub fn hour(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&time_to_hour)
    }

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    pub fn minute(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&time_to_minute)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    pub fn second(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&time_to_second)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    pub fn nanosecond(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&time_to_nanosecond)
    }

    /// Construct a new [`TimeChunked`] from an iterator over [`NaiveTime`].
    pub fn from_naive_time<I: IntoIterator<Item = NaiveTime>>(name: &str, v: I) -> Self {
        let vals = v
            .into_iter()
            .map(|nt| time_to_time64ns(&nt))
            .collect::<Vec<_>>();
        Int64Chunked::from_vec(name, vals).into_time()
    }

    /// Construct a new [`TimeChunked`] from an iterator over optional [`NaiveTime`].
    pub fn from_naive_time_options<I: IntoIterator<Item = Option<NaiveTime>>>(
        name: &str,
        v: I,
    ) -> Self {
        let vals = v.into_iter().map(|opt| opt.map(|nt| time_to_time64ns(&nt)));
        Int64Chunked::from_iter_options(name, vals).into_time()
    }

    pub fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self {
        let mut ca: Int64Chunked = v
            .iter()
            .map(|s| {
                NaiveTime::parse_from_str(s, fmt)
                    .ok()
                    .as_ref()
                    .map(time_to_time64ns)
            })
            .collect_trusted();
        ca.rename(name);
        ca.into()
    }
}
