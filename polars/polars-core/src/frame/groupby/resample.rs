use crate::frame::groupby::GroupBy;
use crate::prelude::*;
use crate::utils::chrono::{Datelike, NaiveDate};

pub enum SampleRule {
    Month(u32),
    Week(u32),
    Day(u32),
    Hour(u32),
    Minute(u32),
    Second(u32),
}

impl Date32Chunked {
    pub fn round(&self, rule: SampleRule) -> Date32Chunked {
        use SampleRule::*;
        let mut out = match rule {
            Month(n) => {
                let year = self.year();
                let month = &self.month() / n;
                year.into_iter()
                    .zip(month.into_iter())
                    .map(|(yr, month)| match (yr, month) {
                        (Some(yr), Some(month)) => NaiveDate::from_ymd_opt(yr, month, 1)
                            .map(|nd| (nd.and_hms(0, 0, 0).timestamp() / SECONDS_IN_DAY) as i32),
                        _ => None,
                    })
                    .collect()
            }
            Week(n) => {
                let year = self.year();
                // We floor divide to create a bucket.
                let week = self.week() / n;
                year.into_iter()
                    // convert to ordinal days by multiplying the week no. by 7
                    // the week number starts with 1 so we translate the week numbers by 1
                    .zip((&(&week - 1) * 7).into_iter())
                    .map(|(yr, od)| match (yr, od) {
                        (Some(yr), Some(od)) => {
                            // the calendar week doesn't start on a monday, so we must offset
                            let offset = 8 - NaiveDate::from_ymd(yr, 1, 1)
                                .weekday()
                                .num_days_from_monday();

                            NaiveDate::from_yo_opt(yr, od + offset)
                                .map(|nd| (nd.and_hms(0, 0, 0).timestamp() / SECONDS_IN_DAY) as i32)
                        }
                        _ => None,
                    })
                    .collect()
            }
            Day(n) => {
                // just floor divide to create a bucket
                self / n * n
            }
            Hour(_) => {
                // date32 does not have hours
                self.clone()
            }
            Minute(_) => {
                // date32 does not have minutes
                self.clone()
            }
            Second(_) => {
                // date32 does not have minutes
                self.clone()
            }
        };
        out.rename(self.name());
        out
    }
}

impl Date64Chunked {
    pub fn round(&self, rule: SampleRule) -> Date64Chunked {
        use SampleRule::*;
        let mut out = match rule {
            Month(n) => {
                let year = self.year();
                let month = &self.month() / n;
                year.into_iter()
                    .zip(month.into_iter())
                    .map(|(yr, month)| match (yr, month) {
                        (Some(yr), Some(month)) => NaiveDate::from_ymd_opt(yr, month, 1)
                            .map(|nd| nd.and_hms(0, 0, 0).timestamp_millis()),
                        _ => None,
                    })
                    .collect()
            }
            Week(n) => {
                let year = self.year();
                // We floor divide to create a bucket.
                let week = self.week() / n;
                year.into_iter()
                    // convert to ordinal days by multiplying the week no. by 7
                    // the week number starts with 1 so we translate the week numbers by 1
                    .zip((&(&week - 1) * 7).into_iter())
                    .map(|(yr, od)| match (yr, od) {
                        (Some(yr), Some(od)) => {
                            // the calendar week doesn't start on a monday, so we must offset
                            let offset = 8 - NaiveDate::from_ymd(yr, 1, 1)
                                .weekday()
                                .num_days_from_monday();

                            NaiveDate::from_yo_opt(yr, od + offset)
                                .map(|nd| nd.and_hms(0, 0, 0).timestamp_millis())
                        }
                        _ => None,
                    })
                    .collect()
            }
            Day(n) => {
                // just floor divide to create a bucket
                let fact = 1000 * 3600 * 24 * n;
                self / fact * fact
            }
            Hour(n) => {
                let fact = 1000 * 3600 * n;
                self / fact * fact
            }
            Minute(n) => {
                let fact = 1000 * 60 * n;
                self / fact * fact
            }
            Second(n) => {
                let fact = 1000 * n;
                self / fact * fact
            }
        };
        out.rename(self.name());
        out
    }
}

impl DataFrame {
    /// Downsample a temporal column by some frequency/ rule
    ///
    /// # Examples
    ///
    /// Consider the following input DataFrame:
    ///
    /// ```text
    /// ╭─────────────────────┬─────╮
    /// │ ms                  ┆ i   │
    /// │ ---                 ┆ --- │
    /// │ date64(ms)          ┆ u8  │
    /// ╞═════════════════════╪═════╡
    /// │ 2000-01-01 00:00:00 ┆ 0   │
    /// ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    /// │ 2000-01-01 00:01:00 ┆ 1   │
    /// ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    /// │ 2000-01-01 00:02:00 ┆ 2   │
    /// ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    /// │ 2000-01-01 00:03:00 ┆ 3   │
    /// ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    /// │ ...                 ┆ ... │
    /// ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    /// │ 2000-01-01 00:15:00 ┆ 15  │
    /// ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    /// │ 2000-01-01 00:16:00 ┆ 16  │
    /// ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    /// │ 2000-01-01 00:17:00 ┆ 17  │
    /// ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    /// │ 2000-01-01 00:18:00 ┆ 18  │
    /// ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    /// │ 2000-01-01 00:19:00 ┆ 19  │
    /// ╰─────────────────────┴─────╯
    /// ```
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_core::frame::groupby::resample::SampleRule;
    ///
    /// fn example(df: &DataFrame) -> Result<DataFrame> {
    ///     df.downsample("datetime", SampleRule::Minute(5))?
    ///         .first()?
    ///         .sort("datetime", false)
    /// }
    /// ```
    /// outputs:
    /// ```text
    ///  ╭─────────────────────┬─────────╮
    ///  │ ms                  ┆ i_first │
    ///  │ ---                 ┆ ---     │
    ///  │ date64(ms)          ┆ u8      │
    ///  ╞═════════════════════╪═════════╡
    ///  │ 2000-01-01 00:00:00 ┆ 0       │
    ///  ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
    ///  │ 2000-01-01 00:05:00 ┆ 5       │
    ///  ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
    ///  │ 2000-01-01 00:10:00 ┆ 10      │
    ///  ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
    ///  │ 2000-01-01 00:15:00 ┆ 15      │
    ///  ╰─────────────────────┴─────────╯
    /// ```
    #[cfg_attr(docsrs, doc(cfg(all(feature = "downsample", feature = "temporal"))))]
    #[cfg(all(feature = "downsample", feature = "temporal"))]
    pub fn downsample(&self, key: &str, rule: SampleRule) -> Result<GroupBy> {
        let s = self.column(key)?;
        self.downsample_with_series(s, rule)
    }

    /// See [downsample](crate::frame::DataFrame::downsample).
    #[cfg_attr(docsrs, doc(cfg(all(feature = "downsample", feature = "temporal"))))]
    #[cfg(all(feature = "downsample", feature = "temporal"))]
    pub fn downsample_with_series(&self, key: &Series, rule: SampleRule) -> Result<GroupBy> {
        use SampleRule::*;

        let year_c = "__POLARS_TEMP_YEAR";
        let day_c = "__POLARS_TEMP_DAY";
        let hour_c = "__POLARS_TEMP_HOUR";
        let minute_c = "__POLARS_TEMP_MINUTE";
        let second_c = "__POLARS_TEMP_SECOND";
        let temp_key = "__POLAR_TEMP_NAME";

        let key_dtype = key.dtype();
        let mut key_phys = key.to_physical_repr();
        let mut key = key.clone();
        let key_name = key.name().to_string();
        let wrong_key_dtype = || Err(PolarsError::Other("key should be date32 || date64".into()));
        let wrong_key_dtype_date64 = || Err(PolarsError::Other("key should be date64".into()));

        // We add columns to group on. We need to make sure that we do not groupby seconds
        // that belong to another minute, or another day, year, etc. That's why we add all
        // those columns to make sure that te group is unique in cyclic events.

        // year is needed in all branches
        let mut year = key.year()?.into_series();
        year.rename(year_c);

        let mut df = self.clone();
        df.drop(key.name())?;

        let selection = self
            .get_columns()
            .iter()
            .filter_map(|c| {
                let name = c.name();
                if name == key.name() {
                    None
                } else {
                    Some(name)
                }
            })
            .collect::<Vec<_>>();

        let gb = match rule {
            Month(n) => {
                let month = &key.month()? / n;

                key = year
                    .i32()?
                    .into_iter()
                    .zip(month.into_iter())
                    .map(|(yr, month)| match (yr, month) {
                        (Some(yr), Some(month)) => NaiveDate::from_ymd_opt(yr, month, 1)
                            .map(|nd| nd.and_hms(0, 0, 0).timestamp_millis()),
                        _ => None,
                    })
                    .collect::<Date64Chunked>()
                    .into_series();

                key.rename(&key_name);

                let mut tempkey = key.clone();
                tempkey.rename(temp_key);

                df.hstack_mut(&[tempkey])?;
                df.groupby_stable(&[temp_key])?
            }

            Week(n) => {
                // We floor divide to create a bucket.
                let week = &key.week()? / n;

                key = year
                    .i32()?
                    .into_iter()
                    // convert to ordinal days by multiplying the week no. by 7
                    // the week number starts with 1 so we translate the week numbers by 1
                    .zip((&(&week - 1) * 7).into_iter())
                    .map(|(yr, od)| match (yr, od) {
                        (Some(yr), Some(od)) => {
                            // the calendar week doesn't start on a monday, so we must offset
                            let offset = 8 - NaiveDate::from_ymd(yr, 1, 1)
                                .weekday()
                                .num_days_from_monday();

                            NaiveDate::from_yo_opt(yr, od + offset)
                                .map(|nd| nd.and_hms(0, 0, 0).timestamp_millis())
                        }
                        _ => None,
                    })
                    .collect::<Date64Chunked>()
                    .into_series();

                key.rename(&key_name);

                let mut tempkey = key.clone();
                tempkey.rename(temp_key);

                df.hstack_mut(&[tempkey])?;
                df.groupby_stable(&[temp_key])?
            }
            Day(n) => {
                // We floor divide to create a bucket.
                let mut day = (&key.ordinal_day()? / n).into_series();
                day.rename(day_c);

                df.hstack_mut(&[year, day])?;

                match key_dtype {
                    DataType::Date32 => {
                        // round to lower bucket
                        key_phys = key_phys / n;
                        key_phys = key_phys * n;
                    }
                    DataType::Date64 => {
                        let fact = 1000 * 3600 * 24 * n;
                        // round to lower bucket
                        key_phys = key_phys / fact;
                        key_phys = key_phys * fact;
                    }
                    _ => return wrong_key_dtype(),
                }
                key = key_phys
                    .cast_with_dtype(key_dtype)
                    .expect("back to original type");

                df.groupby_stable(&[year_c, day_c])?
            }
            Hour(n) => {
                let mut day = key.ordinal_day()?.into_series();
                day.rename(day_c);

                // We floor divide to create a bucket.
                let mut hour = (&key.hour()? / n).into_series();
                hour.rename(hour_c);
                df.hstack_mut(&[year, day, hour])?;

                match key_dtype {
                    DataType::Date64 => {
                        let fact = 1000 * 3600 * n;
                        // round to lower bucket
                        key_phys = key_phys / fact;
                        key_phys = key_phys * fact;
                    }
                    _ => return wrong_key_dtype(),
                }
                key = key_phys
                    .cast_with_dtype(key_dtype)
                    .expect("back to original type");
                df.groupby_stable(&[year_c, day_c, hour_c])?
            }
            Minute(n) => {
                let mut day = key.ordinal_day()?.into_series();
                day.rename(day_c);
                let mut hour = key.hour()?.into_series();
                hour.rename(hour_c);

                // We floor divide to create a bucket.
                let mut minute = (&key.minute()? / n).into_series();
                minute.rename(minute_c);

                df.hstack_mut(&[year, day, hour, minute])?;

                match key_dtype {
                    DataType::Date64 => {
                        let fact = 1000 * 60 * n;
                        // round to lower bucket
                        key_phys = key_phys / fact;
                        key_phys = key_phys * fact;
                    }
                    _ => return wrong_key_dtype_date64(),
                }
                key = key_phys
                    .cast_with_dtype(key_dtype)
                    .expect("back to original type");

                df.groupby_stable(&[year_c, day_c, hour_c, minute_c])?
            }
            Second(n) => {
                let mut day = key.ordinal_day()?.into_series();
                day.rename(day_c);
                let mut hour = key.hour()?.into_series();
                hour.rename(hour_c);
                let mut minute = key.minute()?.into_series();
                minute.rename(minute_c);

                // We floor divide to create a bucket.
                let mut second = (&key.second()? / n).into_series();
                second.rename(second_c);

                df.hstack_mut(&[year, day, hour, minute, second])?;

                match key_dtype {
                    DataType::Date64 => {
                        let fact = 1000 * n;
                        // round to lower bucket
                        key_phys = key_phys / fact;
                        key_phys = key_phys * fact;
                    }
                    _ => return wrong_key_dtype_date64(),
                }
                key = key_phys
                    .cast_with_dtype(key_dtype)
                    .expect("back to original type");

                df.groupby_stable(&[day_c, hour_c, minute_c, second_c])?
            }
        };

        Ok(GroupBy::new(self, vec![key], gb.groups, Some(selection)))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_downsample() -> Result<()> {
        let ts = Date64Chunked::new_from_slice(
            "ms",
            &[
                946684800000,
                946684860000,
                946684920000,
                946684980000,
                946685040000,
                946685100000,
                946685160000,
                946685220000,
                946685280000,
                946685340000,
                946685400000,
                946685460000,
                946685520000,
                946685580000,
                946685640000,
                946685700000,
                946685760000,
                946685820000,
                946685880000,
                946685940000,
            ],
        )
        .into_series();
        let idx = UInt8Chunked::new_from_iter("i", 0..20).into_series();

        let df = DataFrame::new(vec![ts, idx])?;
        dbg!(&df);
        let out = df
            .downsample("ms", SampleRule::Minute(5))?
            .first()?
            .sort("ms", false)?;
        dbg!(&out);
        assert_eq!(
            Vec::from(out.column("i_first")?.u8()?),
            &[Some(0), Some(5), Some(10), Some(15)]
        );

        // check if we can run them without errors
        df.downsample("ms", SampleRule::Week(1))?;
        df.downsample("ms", SampleRule::Day(1))?;
        df.downsample("ms", SampleRule::Hour(1))?;
        df.downsample("ms", SampleRule::Minute(1))?;
        df.downsample("ms", SampleRule::Second(1))?;
        Ok(())
    }

    #[test]
    fn test_downsample_bucket_floors() -> Result<()> {
        // test if the floor divide make sense

        let data = "20210216 23:58:58
20210217 23:58:58
20210310 23:58:58
20210311 23:58:57
20210312 23:58:55
20210313 23:58:55
20210314 23:58:54
20210315 23:58:54
20210316 23:58:50
20210317 23:58:50
20210318 23:58:49
20210319 23:59:01";
        let data: Vec<_> = data.split('\n').collect();

        let date = Utf8Chunked::new_from_slice("date", &data);
        let date = date.as_date64(None)?.into_series();
        let values =
            UInt32Chunked::new_from_iter("values", (0..date.len()).map(|v| v as u32)).into_series();

        let df = DataFrame::new(vec![date.clone(), values.clone()]).unwrap();
        let out = df.downsample("date", SampleRule::Week(1))?.first()?;

        assert_eq!(
            Vec::from(&out.column("date")?.year()?),
            &[Some(2021), Some(2021), Some(2021)]
        );
        assert_eq!(
            Vec::from(&out.column("date")?.month()?),
            &[Some(2), Some(3), Some(3)]
        );
        // ordinal days match with 2021-02-15, 2021-03-08, 2021-03-15
        assert_eq!(
            Vec::from(&out.column("date")?.ordinal_day()?),
            &[Some(46), Some(67), Some(74)]
        );

        let df = DataFrame::new(vec![date, values]).unwrap();
        let out = df.downsample("date", SampleRule::Month(1))?.first()?;
        // ordinal days match with 2021-02-01, 2021-03-01
        assert_eq!(
            Vec::from(&out.column("date")?.ordinal_day()?),
            &[Some(32), Some(60)]
        );
        Ok(())
    }
}
