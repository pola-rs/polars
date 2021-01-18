//! Traits and utilities for temporal data.
pub mod conversion;
pub(crate) mod conversions_utils;
pub use self::conversion::{
    AsDuration, AsNaiveDate, AsNaiveDateTime, AsNaiveTime, FromNaiveDate, FromNaiveDateTime,
    FromNaiveTime,
};
pub(crate) use self::conversions_utils::*;
use chrono::NaiveDateTime;

pub fn unix_time() -> NaiveDateTime {
    NaiveDateTime::from_timestamp(0, 0)
}

#[cfg(all(test, feature = "temporal"))]
mod test {
    use crate::prelude::*;
    use chrono::{NaiveDateTime, NaiveTime};

    #[test]
    fn from_time() {
        let times: Vec<_> = ["23:56:04", "00:00:00"]
            .iter()
            .map(|s| NaiveTime::parse_from_str(s, "%H:%M:%S").unwrap())
            .collect();
        let t = Time64NanosecondChunked::new_from_naive_time("times", &times);
        // NOTE: the values are checked and correct.
        assert_eq!([86164000000000, 0], t.cont_slice().unwrap());
    }

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
        let dt = Date64Chunked::new_from_naive_datetime("name", &datetimes);
        assert_eq!(
            [588470416000, 1441497364000, 1356048000000],
            dt.cont_slice().unwrap()
        );
    }

    #[test]
    fn from_date() {
        let dates = &[
            "2020-08-21",
            "2020-08-21",
            "2020-08-22",
            "2020-08-23",
            "2020-08-22",
        ];
        let fmt = "%Y-%m-%d";
        let ca = Date32Chunked::parse_from_str_slice("dates", dates, fmt);
        assert_eq!(
            [18495, 18495, 18496, 18497, 18496],
            ca.cont_slice().unwrap()
        );
    }
}
