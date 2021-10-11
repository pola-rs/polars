use super::*;
use crate::chunked_array::kernels::temporal::{
    time_to_hour, time_to_minute, time_to_nanosecond, time_to_second,
};
use crate::prelude::*;
use crate::utils::chrono::Timelike;
use crate::utils::NoNull;
use arrow::temporal_conversions::{time64ns_to_time, NANOSECONDS};

pub(crate) fn time_to_time64ns(time: &NaiveTime) -> i64 {
    time.second() as i64 * NANOSECONDS + time.nanosecond() as i64
}

impl TimeChunked {
    pub fn as_time_iter(&self) -> impl Iterator<Item = Option<NaiveTime>> + TrustedLen + '_ {
        self.downcast_iter()
            .map(|iter| {
                iter.into_iter()
                    .map(|opt_v| opt_v.copied().map(time64ns_to_time))
            })
            .flatten()
            .trust_my_length(self.len())
    }

    /// Format Date with a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn strftime(&self, fmt: &str) -> Utf8Chunked {
        let mut ca: Utf8Chunked = self.apply_kernel_cast(|arr| {
            let arr: Utf8Array<i64> = arr
                .into_iter()
                .map(|opt| opt.map(|v| format!("{}", time64ns_to_time(*v).format(fmt))))
                .collect();
            Arc::new(arr)
        });
        ca.rename(self.name());
        ca
    }

    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    pub fn hour(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(time_to_hour)
    }

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    pub fn minute(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(time_to_minute)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    pub fn second(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(time_to_second)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    pub fn nanosecond(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(time_to_nanosecond)
    }

    pub fn new_from_naive_time(name: &str, v: &[NaiveTime]) -> Self {
        let ca: NoNull<Int64Chunked> = v.iter().map(time_to_time64ns).collect_trusted();
        let mut ca = ca.into_inner();
        ca.rename(name);
        ca.into()
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
