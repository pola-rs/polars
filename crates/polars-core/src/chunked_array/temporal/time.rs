use std::fmt::Write;

use arrow::temporal_conversions::{NANOSECONDS, time64ns_to_time};
use chrono::Timelike;
use polars_array::PlUtf8ViewArrayBuilder;
use polars_array::builder::StaticArrayBuilder;

use super::*;
use crate::prelude::*;

const SECONDS_IN_MINUTE: i64 = 60;
const SECONDS_IN_HOUR: i64 = 3_600;

pub fn time_to_time64ns(time: &NaiveTime) -> i64 {
    (time.hour() as i64 * SECONDS_IN_HOUR
        + time.minute() as i64 * SECONDS_IN_MINUTE
        + time.second() as i64)
        * NANOSECONDS
        + time.nanosecond() as i64
}

impl TimeChunked {
    /// Convert from Time into String with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn to_string(&self, format: &str) -> StringChunked {
        let format = if format == "iso" || format == "iso:strict" {
            "%T%.9f"
        } else {
            format
        };

        // One buffer is formatted into and appended per element, rather than one `String` being
        // allocated per element and thrown away.
        let mut buf = String::new();
        let chunks = self.physical().downcast_iter().map(|arr| {
            let mut builder = PlUtf8ViewArrayBuilder::with_capacity(arr.len());
            for opt in arr.iter() {
                match opt {
                    None => builder.push_null(),
                    Some(v) => {
                        buf.clear();
                        let timefmt = time64ns_to_time(v).format(format);
                        write!(buf, "{timefmt}").unwrap();
                        builder.push_value(&buf)
                    },
                }
            }
            builder.freeze()
        });
        let mut ca = StringChunked::from_chunk_iter(PlSmallStr::EMPTY, chunks);

        ca.rename(self.name().clone());
        ca
    }

    /// Convert from Time into String with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
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
                .flat_map(|iter| iter.into_iter().map(|opt_v| opt_v.map(time64ns_to_time)))
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
