use crate::frame::groupby::GroupBy;
use crate::prelude::*;

pub enum SampleRule {
    Week(u32),
    Day(u32),
    Hour(u32),
    Minute(u32),
    Second(u32),
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
    ///     df.downsample("datetime", SampleRule::Minute(6))?
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
    #[cfg_attr(docsrs, doc(cfg(feature = "downsample", feature = "temporal")))]
    #[cfg(all(feature = "downsample", feature = "temporal"))]
    pub fn downsample(&self, key: &str, rule: SampleRule) -> Result<GroupBy> {
        let s = self.column(key)?;
        self.downsample_with_series(s, rule)
    }

    /// See [downsample](crate::frame::DataFrame::downsample).
    #[cfg_attr(docsrs, doc(cfg(feature = "downsample", feature = "temporal")))]
    #[cfg(all(feature = "downsample", feature = "temporal"))]
    pub fn downsample_with_series(&self, key: &Series, rule: SampleRule) -> Result<GroupBy> {
        use SampleRule::*;

        let year_c = "__POLARS_TEMP_YEAR";
        let week_c = "__POLARS_TEMP_WEEK";
        let day_c = "__POLARS_TEMP_DAY";
        let hour_c = "__POLARS_TEMP_HOUR";
        let minute_c = "__POLARS_TEMP_MINUTE";
        let second_c = "__POLARS_TEMP_SECOND";

        let mut key = key.clone();
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
            Week(n) => {
                // We floor divide to create a bucket.
                let mut week = (&key.week()? / n).into_series();
                week.rename(week_c);

                df.hstack_mut(&[year, week])?;

                match key.dtype() {
                    DataType::Date32 => key = key / (7 * n),
                    DataType::Date64 => key = key / (1000 * 3600 * 24 * 7 * n),
                    _ => return wrong_key_dtype(),
                }

                df.groupby_stable(&[year_c, week_c])?
            }
            Day(n) => {
                // We floor divide to create a bucket.
                let mut day = (&key.ordinal_day()? / n).into_series();
                day.rename(day_c);

                df.hstack_mut(&[year, day])?;

                match key.dtype() {
                    DataType::Date32 => key = key / n,
                    DataType::Date64 => key = key / (1000 * 3600 * 24 * n),
                    _ => return wrong_key_dtype(),
                }

                df.groupby_stable(&[year_c, day_c])?
            }
            Hour(n) => {
                let mut day = key.ordinal_day()?.into_series();
                day.rename(day_c);

                // We floor divide to create a bucket.
                let mut hour = (&key.hour()? / n).into_series();
                hour.rename(hour_c);
                df.hstack_mut(&[year, day, hour])?;

                match key.dtype() {
                    DataType::Date64 => key = key / (1000 * 3600 * n),
                    _ => return wrong_key_dtype(),
                }
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

                match key.dtype() {
                    DataType::Date64 => key = key / (1000 * 3600 * n),
                    _ => return wrong_key_dtype_date64(),
                }

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

                match key.dtype() {
                    DataType::Date64 => key = key / (1000 * n),
                    _ => return wrong_key_dtype_date64(),
                }

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
    fn test_downsample() {
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

        let df = DataFrame::new(vec![ts, idx]).unwrap();
        dbg!(&df);
        let out = df
            .downsample("ms", SampleRule::Minute(5))
            .unwrap()
            .first()
            .unwrap()
            .sort("ms", false)
            .unwrap();
        dbg!(&out);
        assert_eq!(
            Vec::from(out.column("i_first").unwrap().u8().unwrap()),
            &[Some(0), Some(5), Some(10), Some(15)]
        );

        // check if we can run them without errors
        df.downsample("ms", SampleRule::Week(1)).unwrap();
        df.downsample("ms", SampleRule::Day(1)).unwrap();
        df.downsample("ms", SampleRule::Hour(1)).unwrap();
        df.downsample("ms", SampleRule::Minute(1)).unwrap();
        df.downsample("ms", SampleRule::Second(1)).unwrap();
    }
}
