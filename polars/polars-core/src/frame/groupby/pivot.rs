use super::GroupBy;
use crate::prelude::*;
use rayon::prelude::*;

use crate::frame::groupby::hashing::HASHMAP_INIT_SIZE;
use crate::POOL;

#[derive(Copy, Clone)]
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
    /// Do a pivot operation based on the group key, a pivot column and an aggregation function on the values column.
    ///
    /// # Note
    /// Polars'/arrow memory is not ideal for transposing operations like pivots.
    /// If you have a relatively large table, consider using a groupby over a pivot.
    pub fn pivot<I0, S0, I1, S1, I2, S2>(
        &self,
        values: I0,
        index: I1,
        columns: I2,
        agg_fn: PivotAgg,
        sort_columns: bool,
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
        self.pivot_impl(&values, &index, &columns, agg_fn, sort_columns, false)
    }

    pub fn pivot_stable<I0, S0, I1, S1, I2, S2>(
        &self,
        values: I0,
        index: I1,
        columns: I2,
        agg_fn: PivotAgg,
        sort_columns: bool,
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

        self.pivot_impl(&values, &index, &columns, agg_fn, sort_columns, true)
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
        // aggregation function
        agg_fn: PivotAgg,
        sort_columns: bool,
        stable: bool,
    ) -> Result<DataFrame> {
        let keys = self.select_series(index)?;

        let mut final_cols = vec![];

        let mut count = 0;
        let out: Result<()> = POOL.install(|| {
            for column in columns {
                let mut groupby = index.to_vec();
                groupby.push(column.clone());

                let groups = self.groupby_stable(groupby)?.groups;

                let local_keys = keys
                    .par_iter()
                    .map(|k| k.agg_first(&groups))
                    .collect::<Vec<_>>();

                // this are the row locations
                let local_keys = DataFrame::new_no_checks(local_keys);
                let local_keys_gb = local_keys.groupby_stable(index)?;
                if !stable {
                    println!("unstable pivot not yet supported, using stable pivot");
                };
                let local_index_groups = &local_keys_gb.groups;

                let column_s = self.column(column)?;
                let column_agg = column_s.agg_first(&groups);
                let column_agg_physical = column_agg.to_physical_repr();

                let mut col_to_idx = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);

                let mut idx = 0 as IdxSize;
                let col_locations = column_agg_physical
                    .iter()
                    .map(|v| {
                        let idx = *col_to_idx.entry(v).or_insert_with(|| {
                            let old_idx = idx;
                            idx += 1;
                            old_idx
                        });
                        idx
                    })
                    .collect::<Vec<_>>();

                for value_col in values {
                    let value_col = self.column(value_col)?;

                    use PivotAgg::*;
                    let value_agg = match agg_fn {
                        Sum => value_col.agg_sum(&groups).unwrap(),
                        Min => value_col.agg_min(&groups).unwrap(),
                        Max => value_col.agg_max(&groups).unwrap(),
                        Last => value_col.agg_last(&groups),
                        First => value_col.agg_first(&groups),
                        Mean => value_col.agg_mean(&groups).unwrap(),
                        Median => value_col.agg_median(&groups).unwrap(),
                        Count => groups.group_count().into_series(),
                    };

                    let headers = column_agg.unique_stable()?.cast(&DataType::Utf8)?;
                    let headers = headers.utf8().unwrap();
                    let n_rows = local_index_groups.len();
                    let n_cols = headers.len();

                    let mut buf = vec![AnyValue::Null; n_rows * n_cols];

                    let mut col_idx_iter = col_locations.iter();
                    let value_agg_phys = value_agg.to_physical_repr();
                    let mut value_iter = value_agg_phys.iter();
                    for (row_idx, g) in local_index_groups.idx_ref().iter().enumerate() {
                        for _ in g.1 {
                            let val = value_iter.next().unwrap();
                            let col_idx = col_idx_iter.next().unwrap();

                            // Safety:
                            // in bounds
                            unsafe {
                                let idx = row_idx as usize + *col_idx as usize * n_rows;
                                debug_assert!(idx < buf.len());
                                *buf.get_unchecked_mut(idx) = val;
                            }
                        }
                    }
                    let headers_iter = headers.par_iter_indexed();

                    let mut cols = (0..n_cols)
                        .into_par_iter()
                        .zip(headers_iter)
                        .map(|(i, opt_name)| {
                            let offset = i * n_rows;
                            let avs = &buf[offset..offset + n_rows];
                            let name = opt_name.unwrap_or("null");
                            let mut out = Series::new(name, avs);
                            finish_logical_type(&mut out, value_agg.dtype());
                            out
                        })
                        .collect::<Vec<_>>();

                    if sort_columns {
                        cols.sort_unstable_by(|a, b| a.name().partial_cmp(b.name()).unwrap());
                    }

                    let cols = if count == 0 {
                        let mut final_cols = local_keys_gb.keys();
                        final_cols.extend(cols);
                        final_cols
                    } else {
                        cols
                    };
                    count += 1;
                    final_cols.extend_from_slice(&cols);
                }
            }
            Ok(())
        });
        let _ = out?;
        Ok(DataFrame::new_no_checks(final_cols))
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

// Takes a `DataFrame` that only consists of the column aggregates that are pivoted by
// the values in `columns`
fn finish_logical_type(column: &mut Series, dtype: &DataType) {
    *column = column.cast(dtype).unwrap();
}

impl<'df> Pivot<'df> {
    fn execute(&self, agg: PivotAgg) -> Result<DataFrame> {
        println!("This pivot syntax is deprecated. Consider using DataFrame::pivot");

        let index = self
            .gb
            .selected_keys
            .iter()
            .map(|s| s.name().to_string())
            .collect::<Vec<_>>();
        self.gb
            .df
            .pivot_impl(&self.values, &index, &self.columns, agg, true, false)
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
