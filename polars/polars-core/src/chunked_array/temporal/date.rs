use std::fmt::Write;

use arrow::temporal_conversions::date32_to_date;

use super::*;
use crate::prelude::*;

pub(crate) fn naive_date_to_date(nd: NaiveDate) -> i32 {
    let nt = NaiveTime::from_hms_opt(0, 0, 0).unwrap();
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

    /// Construct a new [`DateChunked`] from an iterator over [`NaiveDate`].
    pub fn from_naive_date<I: IntoIterator<Item = NaiveDate>>(name: &str, v: I) -> Self {
        let unit = v.into_iter().map(naive_date_to_date).collect::<Vec<_>>();
        Int32Chunked::from_vec(name, unit).into()
    }

    /// Format Date with a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn strftime(&self, fmt: &str) -> Utf8Chunked {
        let date = NaiveDate::from_ymd_opt(2001, 1, 1).unwrap();
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
                        write!(buf, "{datefmt}").unwrap();
                        mutarr.push(Some(&buf))
                    }
                }
            }

            let arr: Utf8Array<i64> = mutarr.into();
            Box::new(arr)
        });
        ca.rename(self.name());
        ca
    }

    /// Construct a new [`DateChunked`] from an iterator over optional [`NaiveDate`].
    pub fn from_naive_date_options<I: IntoIterator<Item = Option<NaiveDate>>>(
        name: &str,
        v: I,
    ) -> Self {
        let unit = v.into_iter().map(|opt| opt.map(naive_date_to_date));
        Int32Chunked::from_iter_options(name, unit).into()
    }
}
