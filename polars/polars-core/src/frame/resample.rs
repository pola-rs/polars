use crate::frame::groupby::GroupBy;
use crate::prelude::*;

pub enum SampleRule {
    Second(u32),
    Minute(u32),
    Day(u32),
    Hour(u32),
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
    /// use polars_core::frame::resample::SampleRule;
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
    pub fn downsample(&self, key: &str, rule: SampleRule) -> Result<GroupBy> {
        let s = self.column(key)?;
        self.downsample_with_series(s, rule)
    }

    /// See [downsample](crate::frame::DataFrame::downsample).
    pub fn downsample_with_series(&self, key: &Series, rule: SampleRule) -> Result<GroupBy> {
        use SampleRule::*;
        // todo! implement logic for date32 if we don't want to pay the casting price
        let key = key.cast::<Date64Type>()?;

        // first we floor divide so that we get buckets that fit our frequency.
        let (gb, n, multiply) = match rule {
            Second(n) => {
                let gb = &key / 1000 / n;
                (gb, n, 1000)
            }
            Minute(n) => {
                let gb = &key / 60000 / n;
                (gb, n, 60000)
            }
            Day(n) => {
                let fact = 1000 * 3600 * 24;
                let gb = &key / fact / n;
                (gb, n, fact)
            }
            Hour(n) => {
                let fact = 1000 * 3600;
                let gb = &key / fact / n;
                (gb, n, fact)
            }
        };
        let mut gb = self.groupby_with_series(vec![gb], true)?;
        // we restore the original scale by multiplying with the earlier floor division value
        gb.selected_keys = gb
            .selected_keys
            .into_iter()
            .map(|s| &s * multiply * n)
            .collect();
        Ok(gb)
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
    }
}
