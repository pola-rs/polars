use super::GroupBy;
use crate::prelude::*;
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::Ordering;

use crate::frame::groupby::{GroupsIndicator, GroupsProxy};
use crate::utils::accumulate_dataframes_vertical;
use crate::POOL;
#[cfg(feature = "dtype-date")]
use arrow::temporal_conversions::date32_to_date;
#[cfg(feature = "dtype-datetime")]
use arrow::temporal_conversions::{timestamp_ms_to_datetime, timestamp_ns_to_datetime};

pub enum PivotAgg {
    First,
    Sum,
    Min,
    Max,
    Mean,
    Median,
    Count,
    Last,
}

impl DataFrame {
    pub fn pivot<I0, S0, I1, S1, I2, S2>(
        &self,
        values: I0,
        index: I1,
        columns: I2,
        agg_fn: PivotAgg,
    ) -> Result<DataFrame>
    where
        I0: IntoIterator<Item = S0>,
        S0: AsRef<str>,
        I1: IntoIterator<Item = S1>,
        S1: AsRef<str>,
        I2: IntoIterator<Item = S2>,
        S2: AsRef<str>,
    {
        let values = values
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>();
        let index = index
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>();
        let columns = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>();
        let groups = self.groupby(&index)?.groups;
        self.pivot_impl(&values, &index, &columns, &groups, agg_fn)
    }

    pub fn pivot_stable<I0, S0, I1, S1, I2, S2>(
        &self,
        values: I0,
        index: I1,
        columns: I2,
        agg_fn: PivotAgg,
    ) -> Result<DataFrame>
    where
        I0: IntoIterator<Item = S0>,
        S0: AsRef<str>,
        I1: IntoIterator<Item = S1>,
        S1: AsRef<str>,
        I2: IntoIterator<Item = S2>,
        S2: AsRef<str>,
    {
        let values = values
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>();
        let index = index
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>();
        let columns = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>();

        let groups = self.groupby_stable(&index)?.groups;

        self.pivot_impl(&values, &index, &columns, &groups, agg_fn)
    }

    fn pivot_impl(
        &self,
        // these columns will be aggregated in the nested groupby
        values: &[String],
        // keys of the first groupby operation
        index: &[String],
        // these columns will be used for a nested groupby
        // the rows of this nested groupby will be pivoted as header column values
        columns: &[String],
        // matching a groupby on index
        groups: &GroupsProxy,
        // aggregation function
        agg_fn: PivotAgg,
    ) -> Result<DataFrame> {
        // broadcast values argument
        let mut values = values.to_vec();
        if values.len() != columns.len() && values.len() == 1 {
            for _ in 0..columns.len() - 1 {
                values.push(values[0].clone())
            }
        }
        assert_eq!(
            values.len(),
            columns.len(),
            "given values should match the given columns in length"
        );

        let values_and_columns = (0..values.len())
            .map(|i| {
                // take only the columns we will use in a smaller dataframe
                // make sure that we take the physical types for the column
                let column = self
                    .column(columns[i].as_str())?
                    .to_physical_repr()
                    .into_owned();
                let values = self.column(values[i].as_str())?;

                Ok(DataFrame::new_no_checks(vec![values.clone(), column]))
            })
            .collect::<Result<Vec<_>>>()?;

        // make sure that we make smaller dataframes then the take operations are cheaper
        let index_df = self.select(index)?;

        let mut im_result = POOL.install(|| {
            groups
                .par_iter()
                .map(|indicator| {
                    // Here we do a nested group by.
                    // Everything we do here produces a single row in the final dataframe

                    // nested group by keys

                    // safety:
                    // group tuples are in bounds
                    // shape (1, len(keys)
                    let sub_index_df = match indicator {
                        GroupsIndicator::Idx(g) => unsafe {
                            index_df.take_unchecked_slice(&g.1[..1])
                        },
                        GroupsIndicator::Slice([first, len]) => {
                            index_df.slice(first as i64, len as usize)
                        }
                    };

                    // in `im_result` we store the intermediate results
                    // The first dataframe in the vec is the index dataframe (a single row)
                    // The rest of the dataframes in `im_result` are the aggregation results (they still have to be pivoted)
                    let mut im_result = Vec::with_capacity(columns.len());
                    im_result.push(sub_index_df);

                    // for every column we compute aggregates we do this branch
                    for (i, column) in columns.iter().enumerate() {
                        // Here we do another groupby where
                        // - `columns` are the keys
                        // - `values` are the aggregation results

                        // this yields:
                        // keys  | values
                        // key_1  | agg_result_1
                        // key_2  | agg_result_2
                        // key_n  | agg_result_n

                        // which later must be transposed to
                        //
                        // header: key_1, key_2, key_n
                        //        agg_1, agg_2, agg_3

                        // safety:
                        // group tuples are in bounds
                        let sub_vals_and_cols = match indicator {
                            GroupsIndicator::Idx(g) => unsafe {
                                values_and_columns[i].take_unchecked_slice(&g.1)
                            },
                            GroupsIndicator::Slice([first, len]) => {
                                values_and_columns[i].slice(first as i64, len as usize)
                            }
                        };

                        let s = sub_vals_and_cols.column(column).unwrap().clone();
                        let gb = sub_vals_and_cols
                            .groupby_with_series(vec![s], false)
                            .unwrap();

                        use PivotAgg::*;
                        let mut df_result = match agg_fn {
                            Sum => gb.sum().unwrap(),
                            Min => gb.min().unwrap(),
                            Max => gb.max().unwrap(),
                            Mean => gb.mean().unwrap(),
                            Median => gb.median().unwrap(),
                            First => gb.first().unwrap(),
                            Count => gb.count().unwrap(),
                            Last => gb.last().unwrap(),
                        };

                        // make sure we keep the original names
                        df_result.columns[1].rename(&values[i]);

                        // store the results and transpose them later
                        im_result.push(df_result);
                    }
                    im_result
                })
                .collect::<Vec<_>>()
        });
        // Now we have a lot of small DataFrames with aggregation results
        // we first join them together.
        // This will lead to a long dataframe that finally is transposed

        // for every column where the values are aggregated
        let mut all_values = (0..columns.len())
            .map(|i| {
                let to_join = im_result
                    .iter_mut()
                    .map(|v| std::mem::take(&mut v[i + 1]))
                    .collect::<Vec<_>>();
                let mut name_count = 0;

                let mut joined = to_join
                    .iter()
                    .map(Cow::Borrowed)
                    .reduce(|df_l, df_r| {
                        let mut out = df_l
                            .outer_join(&df_r, [columns[i].as_str()], [columns[i].as_str()])
                            .unwrap();
                        let last_idx = out.width() - 1;
                        out.columns[last_idx].rename(&format!("{}_{}", values[i], name_count));
                        name_count += 1;
                        Cow::Owned(out)
                    })
                    .unwrap()
                    .into_owned();
                let header = joined
                    .drop_in_place(&columns[i])
                    .unwrap()
                    .cast(&DataType::Utf8)
                    .unwrap();
                let header = header.utf8().unwrap();
                let mut values = joined.transpose().unwrap();

                for (opt_name, s) in header.into_iter().zip(values.columns.iter_mut()) {
                    match opt_name {
                        None => s.rename("null"),
                        Some(v) => s.rename(v),
                    };
                }
                values
            })
            .collect::<Vec<_>>();

        let indices = im_result.iter_mut().map(|v| std::mem::take(&mut v[0]));
        let mut out = accumulate_dataframes_vertical(indices).unwrap();

        // values is the dataframe to stack
        // columns is the original series that is pivoted
        for (values, columns) in all_values.iter_mut().zip(columns) {
            let mut cols = std::mem::take(&mut values.columns);
            sort_cols(&mut cols, 0);

            let df = DataFrame::new_no_checks(cols);
            let df = finish_logical_types(df, self.column(columns).unwrap()).unwrap();

            out = out.hstack(&df.columns)?
        }
        Ok(out)
    }
}

impl<'df> GroupBy<'df> {
    /// Pivot a column of the current `DataFrame` and perform one of the following aggregations:
    ///
    /// * first
    /// * last
    /// * sum
    /// * min
    /// * max
    /// * mean
    /// * median
    ///
    /// The pivot operation consists of a group by one, or multiple columns (these will be the new
    /// y-axis), column that will be pivoted (this will be the new x-axis) and an aggregation.
    ///
    /// # Panics
    /// If the values column is not a numerical type, the code will panic.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_core::df;
    ///
    /// fn example() -> Result<DataFrame> {
    ///     let df = df!["foo" => ["A", "A", "B", "B", "C"],
    ///         "N" => [1, 2, 2, 4, 2],
    ///         "bar" => ["k", "l", "m", "n", "0"]
    ///         ]?;
    ///
    ///     df.groupby(["foo"])?
    ///     .pivot(["bar"], ["N"])
    ///     .first()
    /// }
    /// ```
    /// Transforms:
    ///
    /// ```text
    /// +-----+-----+-----+
    /// | foo | N   | bar |
    /// | --- | --- | --- |
    /// | str | i32 | str |
    /// +=====+=====+=====+
    /// | "A" | 1   | "k" |
    /// +-----+-----+-----+
    /// | "A" | 2   | "l" |
    /// +-----+-----+-----+
    /// | "B" | 2   | "m" |
    /// +-----+-----+-----+
    /// | "B" | 4   | "n" |
    /// +-----+-----+-----+
    /// | "C" | 2   | "o" |
    /// +-----+-----+-----+
    /// ```
    ///
    /// Into:
    ///
    /// ```text
    /// +-----+------+------+------+------+------+
    /// | foo | o    | n    | m    | l    | k    |
    /// | --- | ---  | ---  | ---  | ---  | ---  |
    /// | str | i32  | i32  | i32  | i32  | i32  |
    /// +=====+======+======+======+======+======+
    /// | "A" | null | null | null | 2    | 1    |
    /// +-----+------+------+------+------+------+
    /// | "B" | null | 4    | 2    | null | null |
    /// +-----+------+------+------+------+------+
    /// | "C" | 2    | null | null | null | null |
    /// +-----+------+------+------+------+------+
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
    pub fn pivot(&mut self, columns: impl IntoVec<String>, values: impl IntoVec<String>) -> Pivot {
        // same as select method
        let columns = columns.into_vec();
        let values = values.into_vec();

        Pivot {
            gb: self,
            columns,
            values,
        }
    }
}

/// Intermediate structure when a `pivot` operation is applied.
/// See [the pivot method for more information.](../group_by/struct.GroupBy.html#method.pivot)
#[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
pub struct Pivot<'df> {
    gb: &'df GroupBy<'df>,
    columns: Vec<String>,
    values: Vec<String>,
}

pub(crate) trait ChunkPivot {
    fn pivot<'a>(
        &self,
        _pivot_series: &'a Series,
        _keys: Vec<Series>,
        _groups: &GroupsProxy,
        _agg_type: PivotAgg,
    ) -> Result<DataFrame> {
        Err(PolarsError::InvalidOperation(
            "Pivot operation not implemented for this type".into(),
        ))
    }

    fn pivot_count<'a>(
        &self,
        _pivot_series: &'a Series,
        _keys: Vec<Series>,
        _groups: &GroupsProxy,
    ) -> Result<DataFrame> {
        Err(PolarsError::InvalidOperation(
            "Pivot count operation not implemented for this type".into(),
        ))
    }
}

fn sort_cols(cols: &mut [Series], offset: usize) {
    (&mut cols[offset..]).sort_unstable_by(|s1, s2| {
        if s1.name() > s2.name() {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    });
}

fn finish_logical_types(mut out: DataFrame, columns: &Series) -> Result<DataFrame> {
    match columns.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical => {
            let piv = columns.categorical().unwrap();
            let rev_map = piv.categorical_map.as_ref().unwrap().clone();
            for s in out.columns.iter_mut() {
                let category = s.name().parse::<u32>().unwrap();
                let name = rev_map.get(category);
                s.rename(name);
            }
        }
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(tu, _) => {
            let fun = match tu {
                TimeUnit::Nanoseconds => timestamp_ns_to_datetime,
                TimeUnit::Milliseconds => timestamp_ms_to_datetime,
            };

            for s in out.columns.iter_mut() {
                let ts = s.name().parse::<i64>().unwrap();
                let nd = fun(ts);
                s.rename(&format!("{}", nd));
            }
        }
        #[cfg(feature = "dtype-date")]
        DataType::Date => {
            for s in out.columns.iter_mut() {
                let days = s.name().parse::<i32>().unwrap();
                let nd = date32_to_date(days);
                s.rename(&format!("{}", nd));
            }
        }
        _ => {}
    }
    Ok(out)
}

impl<'df> Pivot<'df> {
    fn execute(&self, agg: PivotAgg) -> Result<DataFrame> {
        let index = self
            .gb
            .selected_keys
            .iter()
            .map(|s| s.name().to_string())
            .collect::<Vec<_>>();
        self.gb
            .df
            .pivot_impl(&self.values, &index, &self.columns, &self.gb.groups, agg)
    }

    /// Aggregate the pivot results by taking the count values.
    pub fn count(&self) -> Result<DataFrame> {
        self.execute(PivotAgg::Count)
    }

    /// Aggregate the pivot results by taking the first occurring value.
    pub fn first(&self) -> Result<DataFrame> {
        self.execute(PivotAgg::First)
    }

    /// Aggregate the pivot results by taking the sum of all duplicates.
    pub fn sum(&self) -> Result<DataFrame> {
        self.execute(PivotAgg::Sum)
    }

    /// Aggregate the pivot results by taking the minimal value of all duplicates.
    pub fn min(&self) -> Result<DataFrame> {
        self.execute(PivotAgg::Min)
    }

    /// Aggregate the pivot results by taking the maximum value of all duplicates.
    pub fn max(&self) -> Result<DataFrame> {
        self.execute(PivotAgg::Max)
    }

    /// Aggregate the pivot results by taking the mean value of all duplicates.
    pub fn mean(&self) -> Result<DataFrame> {
        self.execute(PivotAgg::Mean)
    }
    /// Aggregate the pivot results by taking the median value of all duplicates.
    pub fn median(&self) -> Result<DataFrame> {
        self.execute(PivotAgg::Median)
    }

    /// Aggregate the pivot results by taking the last value of all duplicates.
    pub fn last(&self) -> Result<DataFrame> {
        self.execute(PivotAgg::Last)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_pivot() {
        let s0 = Series::new("foo", ["A", "A", "B", "B", "C"].as_ref());
        let s1 = Series::new("N", [1, 2, 2, 4, 2].as_ref());
        let s2 = Series::new("bar", ["k", "l", "m", "m", "l"].as_ref());
        let df = DataFrame::new(vec![s0, s1, s2]).unwrap();

        let pvt = df
            .groupby(["foo"])
            .unwrap()
            .pivot(["bar"], ["N"])
            .sum()
            .unwrap();
        assert_eq!(pvt.get_column_names(), &["foo", "k", "l", "m"]);
        assert_eq!(
            Vec::from(&pvt.column("m").unwrap().i32().unwrap().sort(false)),
            &[None, None, Some(6)]
        );
        let pvt = df
            .groupby(["foo"])
            .unwrap()
            .pivot(["bar"], ["N"])
            .min()
            .unwrap();
        assert_eq!(
            Vec::from(&pvt.column("m").unwrap().i32().unwrap().sort(false)),
            &[None, None, Some(2)]
        );
        let pvt = df
            .groupby(["foo"])
            .unwrap()
            .pivot(["bar"], ["N"])
            .max()
            .unwrap();
        assert_eq!(
            Vec::from(&pvt.column("m").unwrap().i32().unwrap().sort(false)),
            &[None, None, Some(4)]
        );
        let pvt = df
            .groupby(["foo"])
            .unwrap()
            .pivot(["bar"], ["N"])
            .mean()
            .unwrap();
        assert_eq!(
            Vec::from(&pvt.column("m").unwrap().f64().unwrap().sort(false)),
            &[None, None, Some(3.0)]
        );
        let pvt = df
            .groupby(["foo"])
            .unwrap()
            .pivot(["bar"], ["N"])
            .count()
            .unwrap();
        assert_eq!(
            Vec::from(&pvt.column("m").unwrap().u32().unwrap().sort(false)),
            &[None, None, Some(2)]
        );
    }

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_pivot_categorical() -> Result<()> {
        let mut df = df![
            "A" => [1, 1, 1, 1, 1, 1, 1, 1],
            "B" => [8, 2, 3, 6, 3, 6, 2, 2],
            "C" => ["a", "b", "c", "a", "b", "c", "a", "b"]
        ]?;
        df.try_apply("C", |s| s.cast(&DataType::Categorical))?;

        let out = df.groupby(["B"])?.pivot(["C"], ["A"]).count()?;
        assert_eq!(out.get_column_names(), &["B", "a", "b", "c"]);

        Ok(())
    }

    #[test]
    #[cfg(feature = "dtype-date")]
    fn test_pivot_date() -> Result<()> {
        let mut df = df![
            "A" => [1, 1, 1, 1, 1, 1, 1, 1],
            "B" => [8, 2, 3, 6, 3, 6, 2, 2],
            "C" => [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        ]?;
        df.try_apply("C", |s| s.cast(&DataType::Date))?;

        let out = df.groupby(["B"])?.pivot(["C"], ["A"]).count()?;
        assert_eq!(out.get_column_names(), &["B", "1972-09-27"]);

        Ok(())
    }

    #[test]
    fn test_pivot_new() -> Result<()> {
        let df = df!["A"=> ["foo", "foo", "foo", "foo", "foo",
            "bar", "bar", "bar", "bar"],
            "B"=> ["one", "one", "one", "two", "two",
            "one", "one", "two", "two"],
            "C"=> ["small", "large", "large", "small",
            "small", "large", "small", "small", "large"],
            "breaky"=> ["jam", "egg", "egg", "egg",
             "jam", "jam", "potato", "jam", "jam"],
            "D"=> [1, 2, 2, 3, 3, 4, 5, 6, 7],
            "E"=> [2, 4, 5, 5, 6, 6, 8, 9, 9]
        ]?;

        let out = (df.pivot_stable(["D"], ["A", "B"], ["C"], PivotAgg::Sum))?;
        let expected = df![
            "A" => ["foo", "foo", "bar", "bar"],
            "B" => ["one", "two", "one", "two"],
            "large" => [Some(4), None, Some(4), Some(7)],
            "small" => [1, 6, 5, 6],
        ]?;
        assert!(out.frame_equal_missing(&expected));

        let out = df.pivot_stable(["D"], ["A", "B"], ["C", "breaky"], PivotAgg::Sum)?;
        let expected = df![
            "A" => ["foo", "foo", "bar", "bar"],
            "B" => ["one", "two", "one", "two"],
            "large" => [Some(4), None, Some(4), Some(7)],
            "small" => [1, 6, 5, 6],
            "egg" => [Some(4), Some(3), None, None],
            "jam" => [1, 3, 4, 13],
            "potato" => [None, None, Some(5), None]
        ]?;
        assert!(out.frame_equal_missing(&expected));

        Ok(())
    }
}
