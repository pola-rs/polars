use super::*;
#[cfg(feature = "dtype-time")]
use crate::chunked_array::temporal::time::time_to_time64ns;
use crate::prelude::*;
use polars_time::export::chrono;

#[cfg(feature = "dtype-time")]
fn time_pattern<F, K>(val: &str, convert: F) -> Option<&'static str>
// (string, fmt) -> result
where
    F: Fn(&str, &str) -> chrono::ParseResult<K>,
{
    for fmt in ["%T", "%T%.3f", "%T%.6f", "%T%.9f"] {
        if convert(val, fmt).is_ok() {
            return Some(fmt);
        }
    }
    None
}

fn date_pattern<F, K>(val: &str, convert: F) -> Option<&'static str>
// (string, fmt) -> result
where
    F: Fn(&str, &str) -> chrono::ParseResult<K>,
{
    for fmt in [
        // 2021-12-31
        "%Y-%m-%d",
        // 31-12-2021
        "%d-%m-%Y",
        // 21/12/31 12:54:98
        "%y/%m/%d %H:%M:%S",
        // 2021-12-31 24:58:01
        "%y-%m-%d %H:%M:%S",
        // 21/12/31 24:58:01
        "%y/%m/%d %H:%M:%S",
        //210319 23:58:50
        "%y%m%d %H:%M:%S",
        // 2019-04-18T02:45:55
        // 2021/12/31 12:54:98
        "%Y/%m/%d %H:%M:%S",
        // 2021-12-31 24:58:01
        "%Y-%m-%d %H:%M:%S",
        // 2021/12/31 24:58:01
        "%Y/%m/%d %H:%M:%S",
        // 20210319 23:58:50
        "%Y%m%d %H:%M:%S",
        // 2021319 (2021-03-19)
        "%Y%m%d",
        // 2019-04-18T02:45:55
        "%FT%H:%M:%S",
        // 2019-04-18T02:45:55.555000000
        // microseconds
        "%FT%H:%M:%S.%6f",
        // nanoseconds
        "%FT%H:%M:%S.%9f",
    ] {
        if convert(val, fmt).is_ok() {
            return Some(fmt);
        }
    }
    None
}

impl Utf8Chunked {
    fn get_first_val(&self) -> Result<&str> {
        let idx = match self.first_non_null() {
            Some(idx) => idx,
            None => {
                return Err(PolarsError::HasNullValues(
                    "Cannot determine date parsing format, all values are null".into(),
                ))
            }
        };
        let val = self.get(idx).expect("should not be null");
        Ok(val)
    }

    #[cfg(feature = "dtype-datetime")]
    fn sniff_fmt_datetime(&self) -> Result<&'static str> {
        let val = self.get_first_val()?;
        if let Some(pattern) = date_pattern(val, NaiveDateTime::parse_from_str) {
            return Ok(pattern);
        }
        Err(PolarsError::ComputeError(
            "Could not find an appropriate format to parse dates, please define a fmt".into(),
        ))
    }

    #[cfg(feature = "dtype-date")]
    fn sniff_fmt_date(&self) -> Result<&'static str> {
        let val = self.get_first_val()?;
        if let Some(pattern) = date_pattern(val, NaiveDate::parse_from_str) {
            return Ok(pattern);
        }
        Err(PolarsError::ComputeError(
            "Could not find an appropriate format to parse dates, please define a fmt".into(),
        ))
    }

    #[cfg(feature = "dtype-time")]
    fn sniff_fmt_time(&self) -> Result<&'static str> {
        let val = self.get_first_val()?;
        if let Some(pattern) = time_pattern(val, NaiveTime::parse_from_str) {
            return Ok(pattern);
        }
        Err(PolarsError::ComputeError(
            "Could not find an appropriate format to parse times, please define a fmt".into(),
        ))
    }

    #[cfg(feature = "dtype-time")]
    pub fn as_time(&self, fmt: Option<&str>) -> Result<TimeChunked> {
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => self.sniff_fmt_time()?,
        };

        let mut ca: Int64Chunked = match self.has_validity() {
            false => self
                .into_no_null_iter()
                .map(|s| {
                    NaiveTime::parse_from_str(s, fmt)
                        .ok()
                        .as_ref()
                        .map(time_to_time64ns)
                })
                .collect_trusted(),
            _ => self
                .into_iter()
                .map(|opt_s| {
                    let opt_nd = opt_s.map(|s| {
                        NaiveTime::parse_from_str(s, fmt)
                            .ok()
                            .as_ref()
                            .map(time_to_time64ns)
                    });
                    match opt_nd {
                        None => None,
                        Some(None) => None,
                        Some(Some(nd)) => Some(nd),
                    }
                })
                .collect_trusted(),
        };
        ca.rename(self.name());
        Ok(ca.into())
    }

    #[cfg(feature = "dtype-date")]
    pub fn as_date(&self, fmt: Option<&str>) -> Result<DateChunked> {
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => self.sniff_fmt_date()?,
        };

        let mut ca: Int32Chunked = match self.has_validity() {
            false => self
                .into_no_null_iter()
                .map(|s| {
                    NaiveDate::parse_from_str(s, fmt)
                        .ok()
                        .map(naive_date_to_date)
                })
                .collect_trusted(),
            _ => self
                .into_iter()
                .map(|opt_s| {
                    let opt_nd = opt_s.map(|s| {
                        NaiveDate::parse_from_str(s, fmt)
                            .ok()
                            .map(naive_date_to_date)
                    });
                    match opt_nd {
                        None => None,
                        Some(None) => None,
                        Some(Some(nd)) => Some(nd),
                    }
                })
                .collect_trusted(),
        };
        ca.rename(self.name());
        Ok(ca.into())
    }

    #[cfg(feature = "dtype-datetime")]
    pub fn as_datetime(&self, fmt: Option<&str>) -> Result<DatetimeChunked> {
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => self.sniff_fmt_datetime()?,
        };

        let mut ca: Int64Chunked = match self.has_validity() {
            false => self
                .into_no_null_iter()
                .map(|s| {
                    NaiveDateTime::parse_from_str(s, fmt)
                        .ok()
                        .map(|dt| naive_datetime_to_datetime(&dt))
                })
                .collect_trusted(),
            _ => self
                .into_iter()
                .map(|opt_s| {
                    let opt_nd = opt_s.map(|s| {
                        NaiveDateTime::parse_from_str(s, fmt)
                            .ok()
                            .map(|dt| naive_datetime_to_datetime(&dt))
                    });
                    match opt_nd {
                        None => None,
                        Some(None) => None,
                        Some(Some(nd)) => Some(nd),
                    }
                })
                .collect_trusted(),
        };
        ca.rename(self.name());
        Ok(ca.into())
    }
}
