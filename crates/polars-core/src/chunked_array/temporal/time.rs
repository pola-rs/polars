use std::fmt::Write;

use arrow::temporal_conversions::{NANOSECONDS, time64ns_to_time};

use super::*;
use crate::prelude::*;

const SECONDS_IN_MINUTE: i64 = 60;
const SECONDS_IN_HOUR: i64 = 3_600;

pub fn time_to_time64ns(time: &NaiveTime) -> i64 {
    (time.hour() as i64 * SECONDS_IN_HOUR
        + time.minute() as i64 * SECONDS_IN_MINUTE
        + time.second() as i64)
        * NANOSECONDS
        + time.subsec_nanosecond() as i64
}

impl TimeChunked {
    /// Convert from Time into String with the given format.
    /// See [jiff strftime/strptime](https://docs.rs/jiff/latest/jiff/fmt/strtime/index.html).
    pub fn to_string(&self, format: &str) -> StringChunked {
        let mut ca: StringChunked = self.physical().apply_kernel_cast(&|arr| {
            let mut buf = String::new();
            // The "iso"/"iso:strict"/default format uses a fixed 3, 6, or 9
            // fractional digits per value (whichever is smallest and exact),
            // matching chrono's old `NaiveTime` Display - jiff's own `%.f`
            // directive instead always trims to the minimal representation,
            // so it can't be used as a single static format string here.
            let is_iso_default = matches!(format, "iso" | "iso:strict" | "%T%.f" | "%T%.9f");
            let mut mutarr = MutablePlString::with_capacity(arr.len());

            for opt in arr.into_iter() {
                match opt {
                    None => mutarr.push_null(),
                    Some(v) => {
                        buf.clear();
                        let time = time64ns_to_time(*v);
                        if is_iso_default {
                            match crate::fmt::chrono_compat_subsec_precision(
                                time.subsec_nanosecond(),
                            ) {
                                None => write!(buf, "{time}").unwrap(),
                                Some(prec) => write!(buf, "{time:.prec$}").unwrap(),
                            }
                        } else {
                            write!(buf, "{}", time.strftime(format)).unwrap();
                        }
                        mutarr.push_value(&buf)
                    },
                }
            }

            mutarr.freeze().boxed()
        });

        ca.rename(self.name().clone());
        ca
    }

    /// Convert from Time into String with the given format.
    /// See [jiff strftime/strptime](https://docs.rs/jiff/latest/jiff/fmt/strtime/index.html).
    ///
    /// Alias for `to_string`.
    pub fn strftime(&self, format: &str) -> StringChunked {
        self.to_string(format)
    }

    pub fn as_time_iter(&self) -> impl TrustedLen<Item = Option<NaiveTime>> + '_ {
        // we know the iterators len
        unsafe {
            self.physical()
                .downcast_iter()
                .flat_map(|iter| {
                    iter.into_iter()
                        .map(|opt_v| opt_v.copied().map(time64ns_to_time))
                })
                .trust_my_length(self.len())
        }
    }

    /// Construct a new [`TimeChunked`] from an iterator over [`NaiveTime`].
    pub fn from_naive_time<I: IntoIterator<Item = NaiveTime>>(name: PlSmallStr, v: I) -> Self {
        let vals = v
            .into_iter()
            .map(|nt| time_to_time64ns(&nt))
            .collect::<Vec<_>>();
        Int64Chunked::from_vec(name, vals).into_time()
    }

    /// Construct a new [`TimeChunked`] from an iterator over optional [`NaiveTime`].
    pub fn from_naive_time_options<I: IntoIterator<Item = Option<NaiveTime>>>(
        name: PlSmallStr,
        v: I,
    ) -> Self {
        let vals = v.into_iter().map(|opt| opt.map(|nt| time_to_time64ns(&nt)));
        Int64Chunked::from_iter_options(name, vals).into_time()
    }
}
