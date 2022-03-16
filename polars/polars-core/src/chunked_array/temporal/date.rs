use super::*;
use crate::prelude::*;
use arrow::temporal_conversions::date32_to_date;

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
}
