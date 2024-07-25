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
    pub fn as_date_iter(&self) -> impl TrustedLen<Item = Option<NaiveDate>> + '_ {
        // SAFETY: we know the iterators len
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

    /// Convert from Date into String with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn to_string(&self, format: &str) -> PolarsResult<StringChunked> {
        let datefmt_f = |ndt: NaiveDate| ndt.format(format);
        self.try_apply_into_string_amortized(|val, buf| {
            let ndt = date32_to_date(val);
            write!(buf, "{}", datefmt_f(ndt))
        })
        .map_err(|_| polars_err!(ComputeError: "cannot format Date with format '{}'", format))
    }

    /// Convert from Date into String with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    ///
    /// Alias for `to_string`.
    pub fn strftime(&self, format: &str) -> PolarsResult<StringChunked> {
        self.to_string(format)
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
