use self::hashing::*;
use crate::prelude::*;
#[cfg(feature = "groupby_list")]
use crate::utils::Wrap;
use crate::utils::{
    accumulate_dataframes_vertical, copy_from_slice_unchecked, set_partition_size, split_offsets,
};
use crate::vector_hasher::{get_null_hash_value, AsU64, StrHash};
use crate::POOL;
use ahash::{CallHasher, RandomState};
use hashbrown::HashMap;
use num::NumCast;
use polars_arrow::prelude::QuantileInterpolOptions;
use rayon::prelude::*;
use std::fmt::Debug;
use std::hash::Hash;

pub mod aggregations;
pub(crate) mod hashing;
mod into_groups;
#[cfg(feature = "rows")]
pub(crate) mod pivot;
mod proxy;

#[cfg(feature = "rows")]
pub use pivot::PivotAgg;

pub use into_groups::*;
use polars_arrow::array::ValueSize;
pub use proxy::*;

impl DataFrame {
    pub fn groupby_with_series(
        &self,
        by: Vec<Series>,
        multithreaded: bool,
        sorted: bool,
    ) -> Result<GroupBy> {
        if by.is_empty() {
            return Err(PolarsError::ComputeError(
                "expected keys in groupby operation, got nothing".into(),
            ));
        }

        macro_rules! finish_packed_bit_path {
            ($ca0:expr, $ca1:expr, $pack_fn:expr) => {{
                let n_partitions = set_partition_size();

                // we split so that we can prepare the data over multiple threads.
                // pack the bit values together and add a final byte that will be 0
                // when there are no null values.
                // otherwise we use two bits of this byte to represent null values.
                let splits = split_offsets($ca0.len(), n_partitions);

                let keys = POOL.install(|| {
                    splits
                        .into_par_iter()
                        .map(|(offset, len)| {
                            let ca0 = $ca0.slice(offset as i64, len);
                            let ca1 = $ca1.slice(offset as i64, len);
                            ca0.into_iter()
                                .zip(ca1.into_iter())
                                .map(|(l, r)| $pack_fn(l, r))
                                .collect_trusted::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                });

                return Ok(GroupBy::new(
                    self,
                    by,
                    groupby_threaded_num(keys, 0, n_partitions as u64, sorted),
                    None,
                ));
            }};
        }

        if by.is_empty() || by[0].len() != self.height() {
            return Err(PolarsError::ShapeMisMatch(
                "the Series used as keys should have the same length as the DataFrame".into(),
            ));
        };

        use DataType::*;
        // make sure that categorical and small integers are used as uint32 in value type
        let keys_df = DataFrame::new_no_checks(
            by.iter()
                .map(|s| match s.dtype() {
                    Int8 | UInt8 | Int16 | UInt16 => s.cast(&DataType::UInt32).unwrap(),
                    #[cfg(feature = "dtype-categorical")]
                    Categorical(_) => s.cast(&DataType::UInt32).unwrap(),
                    Float32 => s.bit_repr_small().into_series(),
                    // otherwise we use the vec hash for float
                    Float64 => s.bit_repr_large().into_series(),
                    _ => {
                        // is date like
                        if !s.dtype().is_numeric() && s.is_numeric_physical() {
                            s.to_physical_repr().into_owned()
                        } else {
                            s.clone()
                        }
                    }
                })
                .collect(),
        );

        let n_partitions = set_partition_size();

        let groups = match by.len() {
            1 => {
                let series = &by[0];
                series.group_tuples(multithreaded, sorted)
            }
            2 => {
                // multiple keys is always multi-threaded
                // reduce code paths
                let s0 = &keys_df.get_columns()[0];
                let s1 = &keys_df.get_columns()[1];

                // fast path for numeric data
                // uses the bit values to tightly pack those into arrays.
                if s0.dtype().is_numeric() && s1.dtype().is_numeric() {
                    match (s0.bit_repr_is_large(), s1.bit_repr_is_large()) {
                        (false, false) => {
                            let ca0 = s0.bit_repr_small();
                            let ca1 = s1.bit_repr_small();
                            finish_packed_bit_path!(ca0, ca1, pack_u32_tuples)
                        }
                        (true, true) => {
                            let ca0 = s0.bit_repr_large();
                            let ca1 = s1.bit_repr_large();
                            finish_packed_bit_path!(ca0, ca1, pack_u64_tuples)
                        }
                        (true, false) => {
                            let ca0 = s0.bit_repr_large();
                            let ca1 = s1.bit_repr_small();
                            // small first
                            finish_packed_bit_path!(ca1, ca0, pack_u32_u64_tuples)
                        }
                        (false, true) => {
                            let ca0 = s0.bit_repr_small();
                            let ca1 = s1.bit_repr_large();
                            // small first
                            finish_packed_bit_path!(ca0, ca1, pack_u32_u64_tuples)
                        }
                    }
                } else if matches!((s0.dtype(), s1.dtype()), (DataType::Utf8, DataType::Utf8)) {
                    let lhs = s0.utf8().unwrap();
                    let rhs = s1.utf8().unwrap();

                    // arbitrarily chosen bound, if avg no of bytes to encode is larger than this
                    // value we fall back to default groupby
                    if (lhs.get_values_size() + rhs.get_values_size()) / (lhs.len() + 1) < 128 {
                        pack_utf8_columns(lhs, rhs, n_partitions, sorted)
                    } else {
                        groupby_threaded_multiple_keys_flat(keys_df, n_partitions, sorted)
                    }
                } else {
                    groupby_threaded_multiple_keys_flat(keys_df, n_partitions, sorted)
                }
            }
            _ => groupby_threaded_multiple_keys_flat(keys_df, n_partitions, sorted),
        };
        Ok(GroupBy::new(self, by, groups, None))
    }

    /// Group DataFrame using a Series column.
    ///
    /// # Example
    ///
    /// ```
    /// use polars_core::prelude::*;
    /// fn groupby_sum(df: &DataFrame) -> Result<DataFrame> {
    ///     df.groupby(["column_name"])?
    ///     .select(["agg_column_name"])
    ///     .sum()
    /// }
    /// ```
    pub fn groupby<I, S>(&self, by: I) -> Result<GroupBy>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let selected_keys = self.select_series(by)?;
        self.groupby_with_series(selected_keys, true, false)
    }

    /// Group DataFrame using a Series column.
    /// The groups are ordered by their smallest row index.
    pub fn groupby_stable<I, S>(&self, by: I) -> Result<GroupBy>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let selected_keys = self.select_series(by)?;
        self.groupby_with_series(selected_keys, true, true)
    }
}

/// Returned by a groupby operation on a DataFrame. This struct supports
/// several aggregations.
///
/// Until described otherwise, the examples in this struct are performed on the following DataFrame:
///
/// ```ignore
/// use polars_core::prelude::*;
///
/// let dates = &[
/// "2020-08-21",
/// "2020-08-21",
/// "2020-08-22",
/// "2020-08-23",
/// "2020-08-22",
/// ];
/// // date format
/// let fmt = "%Y-%m-%d";
/// // create date series
/// let s0 = DateChunked::parse_from_str_slice("date", dates, fmt)
///         .into_series();
/// // create temperature series
/// let s1 = Series::new("temp", [20, 10, 7, 9, 1]);
/// // create rain series
/// let s2 = Series::new("rain", [0.2, 0.1, 0.3, 0.1, 0.01]);
/// // create a new DataFrame
/// let df = DataFrame::new(vec![s0, s1, s2]).unwrap();
/// println!("{:?}", df);
/// ```
///
/// Outputs:
///
/// ```text
/// +------------+------+------+
/// | date       | temp | rain |
/// | ---        | ---  | ---  |
/// | Date     | i32  | f64  |
/// +============+======+======+
/// | 2020-08-21 | 20   | 0.2  |
/// +------------+------+------+
/// | 2020-08-21 | 10   | 0.1  |
/// +------------+------+------+
/// | 2020-08-22 | 7    | 0.3  |
/// +------------+------+------+
/// | 2020-08-23 | 9    | 0.1  |
/// +------------+------+------+
/// | 2020-08-22 | 1    | 0.01 |
/// +------------+------+------+
/// ```
///
#[derive(Debug, Clone)]
pub struct GroupBy<'df> {
    df: &'df DataFrame,
    pub(crate) selected_keys: Vec<Series>,
    // [first idx, [other idx]]
    pub(crate) groups: GroupsProxy,
    // columns selected for aggregation
    pub(crate) selected_agg: Option<Vec<String>>,
}

impl<'df> GroupBy<'df> {
    pub fn new(
        df: &'df DataFrame,
        by: Vec<Series>,
        groups: GroupsProxy,
        selected_agg: Option<Vec<String>>,
    ) -> Self {
        GroupBy {
            df,
            selected_keys: by,
            groups,
            selected_agg,
        }
    }

    /// Select the column(s) that should be aggregated.
    /// You can select a single column or a slice of columns.
    ///
    /// Note that making a selection with this method is not required. If you
    /// skip it all columns (except for the keys) will be selected for aggregation.
    #[must_use]
    pub fn select<I: IntoIterator<Item = S>, S: AsRef<str>>(mut self, selection: I) -> Self {
        self.selected_agg = Some(
            selection
                .into_iter()
                .map(|s| s.as_ref().to_string())
                .collect(),
        );
        self
    }

    /// Get the internal representation of the GroupBy operation.
    /// The Vec returned contains:
    ///     (first_idx, Vec<indexes>)
    ///     Where second value in the tuple is a vector with all matching indexes.
    pub fn get_groups(&self) -> &GroupsProxy {
        &self.groups
    }

    /// Get the internal representation of the GroupBy operation.
    /// The Vec returned contains:
    ///     (first_idx, Vec<indexes>)
    ///     Where second value in the tuple is a vector with all matching indexes.
    pub fn get_groups_mut(&mut self) -> &mut GroupsProxy {
        &mut self.groups
    }

    pub fn take_groups(self) -> GroupsProxy {
        self.groups
    }

    pub fn keys(&self) -> Vec<Series> {
        POOL.install(|| {
            self.selected_keys
                .par_iter()
                .map(|s| {
                    // Safety
                    // groupby indexes are in bound.
                    unsafe {
                        s.take_iter_unchecked(
                            &mut self.groups.idx_ref().iter().map(|(idx, _)| idx as usize),
                        )
                    }
                })
                .collect()
        })
    }

    fn prepare_agg(&self) -> Result<(Vec<Series>, Vec<Series>)> {
        let selection = match &self.selected_agg {
            Some(selection) => selection.clone(),
            None => {
                let by: Vec<_> = self.selected_keys.iter().map(|s| s.name()).collect();
                self.df
                    .get_column_names()
                    .into_iter()
                    .filter(|a| !by.contains(a))
                    .map(|s| s.to_string())
                    .collect()
            }
        };

        let keys = self.keys();
        let agg_col = self.df.select_series(selection)?;
        Ok((keys, agg_col))
    }

    /// Aggregate grouped series and compute the mean per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby(["date"])?.select(&["temp", "rain"]).mean()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+-----------+-----------+
    /// | date       | temp_mean | rain_mean |
    /// | ---        | ---       | ---       |
    /// | Date     | f64       | f64       |
    /// +============+===========+===========+
    /// | 2020-08-23 | 9         | 0.1       |
    /// +------------+-----------+-----------+
    /// | 2020-08-22 | 4         | 0.155     |
    /// +------------+-----------+-----------+
    /// | 2020-08-21 | 15        | 0.15      |
    /// +------------+-----------+-----------+
    /// ```
    pub fn mean(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;

        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::Mean);
            let opt_agg = agg_col.agg_mean(&self.groups);
            if let Some(mut agg) = opt_agg {
                agg.rename(&new_name);
                cols.push(agg);
            }
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped series and compute the sum per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby(["date"])?.select(["temp"]).sum()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+----------+
    /// | date       | temp_sum |
    /// | ---        | ---      |
    /// | Date     | i32      |
    /// +============+==========+
    /// | 2020-08-23 | 9        |
    /// +------------+----------+
    /// | 2020-08-22 | 8        |
    /// +------------+----------+
    /// | 2020-08-21 | 30       |
    /// +------------+----------+
    /// ```
    pub fn sum(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;

        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::Sum);
            let opt_agg = agg_col.agg_sum(&self.groups);
            if let Some(mut agg) = opt_agg {
                agg.rename(&new_name);
                cols.push(agg);
            }
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped series and compute the minimal value per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby(["date"])?.select(["temp"]).min()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+----------+
    /// | date       | temp_min |
    /// | ---        | ---      |
    /// | Date     | i32      |
    /// +============+==========+
    /// | 2020-08-23 | 9        |
    /// +------------+----------+
    /// | 2020-08-22 | 1        |
    /// +------------+----------+
    /// | 2020-08-21 | 10       |
    /// +------------+----------+
    /// ```
    pub fn min(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::Min);
            let opt_agg = agg_col.agg_min(&self.groups);
            if let Some(mut agg) = opt_agg {
                agg.rename(&new_name);
                cols.push(agg);
            }
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped series and compute the maximum value per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby(["date"])?.select(["temp"]).max()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+----------+
    /// | date       | temp_max |
    /// | ---        | ---      |
    /// | Date     | i32      |
    /// +============+==========+
    /// | 2020-08-23 | 9        |
    /// +------------+----------+
    /// | 2020-08-22 | 7        |
    /// +------------+----------+
    /// | 2020-08-21 | 20       |
    /// +------------+----------+
    /// ```
    pub fn max(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::Max);
            let opt_agg = agg_col.agg_max(&self.groups);
            if let Some(mut agg) = opt_agg {
                agg.rename(&new_name);
                cols.push(agg);
            }
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped `Series` and find the first value per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby(["date"])?.select(["temp"]).first()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+------------+
    /// | date       | temp_first |
    /// | ---        | ---        |
    /// | Date     | i32        |
    /// +============+============+
    /// | 2020-08-23 | 9          |
    /// +------------+------------+
    /// | 2020-08-22 | 7          |
    /// +------------+------------+
    /// | 2020-08-21 | 20         |
    /// +------------+------------+
    /// ```
    pub fn first(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::First);
            let mut agg = agg_col.agg_first(&self.groups);
            agg.rename(&new_name);
            cols.push(agg);
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped `Series` and return the last value per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby(["date"])?.select(["temp"]).last()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+------------+
    /// | date       | temp_last |
    /// | ---        | ---        |
    /// | Date     | i32        |
    /// +============+============+
    /// | 2020-08-23 | 9          |
    /// +------------+------------+
    /// | 2020-08-22 | 1          |
    /// +------------+------------+
    /// | 2020-08-21 | 10         |
    /// +------------+------------+
    /// ```
    pub fn last(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::Last);
            let mut agg = agg_col.agg_last(&self.groups);
            agg.rename(&new_name);
            cols.push(agg);
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped `Series` by counting the number of unique values.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby(["date"])?.select(["temp"]).n_unique()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+---------------+
    /// | date       | temp_n_unique |
    /// | ---        | ---           |
    /// | Date     | u32           |
    /// +============+===============+
    /// | 2020-08-23 | 1             |
    /// +------------+---------------+
    /// | 2020-08-22 | 2             |
    /// +------------+---------------+
    /// | 2020-08-21 | 2             |
    /// +------------+---------------+
    /// ```
    pub fn n_unique(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::NUnique);
            let opt_agg = agg_col.agg_n_unique(&self.groups);
            if let Some(mut agg) = opt_agg {
                agg.rename(&new_name);
                cols.push(agg.into_series());
            }
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped `Series` and determine the quantile per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// # use polars_arrow::prelude::QuantileInterpolOptions;
    ///
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby(["date"])?.select(["temp"]).quantile(0.2, QuantileInterpolOptions::default())
    /// }
    /// ```
    pub fn quantile(&self, quantile: f64, interpol: QuantileInterpolOptions) -> Result<DataFrame> {
        if !(0.0..=1.0).contains(&quantile) {
            return Err(PolarsError::ComputeError(
                "quantile should be within 0.0 and 1.0".into(),
            ));
        }
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name =
                fmt_groupby_column(agg_col.name(), GroupByMethod::Quantile(quantile, interpol));
            let opt_agg = agg_col.agg_quantile(&self.groups, quantile, interpol);
            if let Some(mut agg) = opt_agg {
                agg.rename(&new_name);
                cols.push(agg.into_series());
            }
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped `Series` and determine the median per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby(["date"])?.select(["temp"]).median()
    /// }
    /// ```
    pub fn median(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::Median);
            let opt_agg = agg_col.agg_median(&self.groups);
            if let Some(mut agg) = opt_agg {
                agg.rename(&new_name);
                cols.push(agg.into_series());
            }
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped `Series` and determine the variance per group.
    pub fn var(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::Var);
            let opt_agg = agg_col.agg_var(&self.groups);
            if let Some(mut agg) = opt_agg {
                agg.rename(&new_name);
                cols.push(agg.into_series());
            }
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped `Series` and determine the standard deviation per group.
    pub fn std(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::Std);
            let opt_agg = agg_col.agg_std(&self.groups);
            if let Some(mut agg) = opt_agg {
                agg.rename(&new_name);
                cols.push(agg.into_series());
            }
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped series and compute the number of values per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby(["date"])?.select(["temp"]).count()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+------------+
    /// | date       | temp_count |
    /// | ---        | ---        |
    /// | Date     | u32        |
    /// +============+============+
    /// | 2020-08-23 | 1          |
    /// +------------+------------+
    /// | 2020-08-22 | 2          |
    /// +------------+------------+
    /// | 2020-08-21 | 2          |
    /// +------------+------------+
    /// ```
    pub fn count(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;

        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::Count);
            let mut ca = self.groups.group_count();
            ca.rename(&new_name);
            cols.push(ca.into_series());
        }
        DataFrame::new(cols)
    }

    /// Get the groupby group indexes.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby(["date"])?.groups()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +--------------+------------+
    /// | date         | groups     |
    /// | ---          | ---        |
    /// | Date(days) | list [u32] |
    /// +==============+============+
    /// | 2020-08-23   | "[3]"      |
    /// +--------------+------------+
    /// | 2020-08-22   | "[2, 4]"   |
    /// +--------------+------------+
    /// | 2020-08-21   | "[0, 1]"   |
    /// +--------------+------------+
    /// ```
    pub fn groups(&self) -> Result<DataFrame> {
        let mut cols = self.keys();
        let mut column = self.groups.as_list_chunked();
        let new_name = fmt_groupby_column("", GroupByMethod::Groups);
        column.rename(&new_name);
        cols.push(column.into_series());
        DataFrame::new(cols)
    }

    /// Combine different aggregations on columns
    ///
    /// ## Operations
    ///
    /// * count
    /// * first
    /// * last
    /// * sum
    /// * min
    /// * max
    /// * mean
    /// * median
    ///
    /// # Example
    ///
    ///  ```rust
    ///  # use polars_core::prelude::*;
    ///  fn example(df: DataFrame) -> Result<DataFrame> {
    ///      df.groupby(["date"])?.agg(&[("temp", &["n_unique", "sum", "min"])])
    ///  }
    ///  ```
    ///  Returns:
    ///
    ///  ```text
    ///  +--------------+---------------+----------+----------+
    ///  | date         | temp_n_unique | temp_sum | temp_min |
    ///  | ---          | ---           | ---      | ---      |
    ///  | Date(days) | u32           | i32      | i32      |
    ///  +==============+===============+==========+==========+
    ///  | 2020-08-23   | 1             | 9        | 9        |
    ///  +--------------+---------------+----------+----------+
    ///  | 2020-08-22   | 2             | 8        | 1        |
    ///  +--------------+---------------+----------+----------+
    ///  | 2020-08-21   | 2             | 30       | 10       |
    ///  +--------------+---------------+----------+----------+
    ///  ```
    ///
    pub fn agg<Column, S, Slice>(&self, column_to_agg: &[(Column, Slice)]) -> Result<DataFrame>
    where
        S: AsRef<str>,
        S: AsRef<str>,
        Slice: AsRef<[S]>,
        Column: AsRef<str>,
    {
        // create a mapping from columns to aggregations on that column
        let mut map = HashMap::with_hasher(RandomState::new());
        column_to_agg.iter().for_each(|(column, aggregations)| {
            map.insert(column.as_ref(), aggregations.as_ref());
        });

        macro_rules! finish_agg_opt {
            ($self:ident, $name_fmt:expr, $agg_fn:ident, $agg_col:ident, $cols:ident) => {{
                let new_name = format!($name_fmt, $agg_col.name());
                let opt_agg = $agg_col.$agg_fn(&$self.groups);
                if let Some(mut agg) = opt_agg {
                    agg.rename(&new_name);
                    $cols.push(agg.into_series());
                }
            }};
        }
        macro_rules! finish_agg {
            ($self:ident, $name_fmt:expr, $agg_fn:ident, $agg_col:ident, $cols:ident) => {{
                let new_name = format!($name_fmt, $agg_col.name());
                let mut agg = $agg_col.$agg_fn(&$self.groups);
                agg.rename(&new_name);
                $cols.push(agg.into_series());
            }};
        }

        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in &agg_cols {
            if let Some(&aggregations) = map.get(agg_col.name()) {
                for aggregation_f in aggregations {
                    match aggregation_f.as_ref() {
                        "min" => finish_agg_opt!(self, "{}_min", agg_min, agg_col, cols),
                        "max" => finish_agg_opt!(self, "{}_max", agg_max, agg_col, cols),
                        "mean" => finish_agg_opt!(self, "{}_mean", agg_mean, agg_col, cols),
                        "sum" => finish_agg_opt!(self, "{}_sum", agg_sum, agg_col, cols),
                        "first" => finish_agg!(self, "{}_first", agg_first, agg_col, cols),
                        "last" => finish_agg!(self, "{}_last", agg_last, agg_col, cols),
                        "n_unique" => {
                            finish_agg_opt!(self, "{}_n_unique", agg_n_unique, agg_col, cols)
                        }
                        "median" => finish_agg_opt!(self, "{}_median", agg_median, agg_col, cols),
                        "std" => finish_agg_opt!(self, "{}_std", agg_std, agg_col, cols),
                        "var" => finish_agg_opt!(self, "{}_var", agg_var, agg_col, cols),
                        "count" => {
                            let new_name = format!("{}_count", agg_col.name());
                            let mut ca = self.groups.group_count();
                            ca.rename(&new_name);
                            cols.push(ca.into_series());
                        }
                        a => panic!("aggregation: {:?} is not supported", a),
                    }
                }
            }
        }
        DataFrame::new(cols)
    }

    /// Aggregate the groups of the groupby operation into lists.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     // GroupBy and aggregate to Lists
    ///     df.groupby(["date"])?.select(["temp"]).agg_list()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+------------------------+
    /// | date       | temp_agg_list          |
    /// | ---        | ---                    |
    /// | Date     | list [i32]             |
    /// +============+========================+
    /// | 2020-08-23 | "[Some(9)]"            |
    /// +------------+------------------------+
    /// | 2020-08-22 | "[Some(7), Some(1)]"   |
    /// +------------+------------------------+
    /// | 2020-08-21 | "[Some(20), Some(10)]" |
    /// +------------+------------------------+
    /// ```
    pub fn agg_list(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::List);
            if let Some(mut agg) = agg_col.agg_list(&self.groups) {
                agg.rename(&new_name);
                cols.push(agg);
            }
        }
        DataFrame::new(cols)
    }

    fn prepare_apply(&self) -> Result<DataFrame> {
        if let Some(agg) = &self.selected_agg {
            if agg.is_empty() {
                Ok(self.df.clone())
            } else {
                let mut new_cols = Vec::with_capacity(self.selected_keys.len() + agg.len());
                new_cols.extend_from_slice(&self.selected_keys);
                let cols = self.df.select_series(agg)?;
                new_cols.extend(cols.into_iter());
                Ok(DataFrame::new_no_checks(new_cols))
            }
        } else {
            Ok(self.df.clone())
        }
    }

    /// Apply a closure over the groups as a new DataFrame in parallel.
    pub fn par_apply<F>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(DataFrame) -> Result<DataFrame> + Send + Sync,
    {
        let df = self.prepare_apply()?;
        let dfs = self
            .get_groups()
            .idx_ref()
            .par_iter()
            .map(|t| {
                let sub_df = unsafe { df.take_iter_unchecked(t.1.iter().map(|i| *i as usize)) };
                f(sub_df)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut df = accumulate_dataframes_vertical(dfs)?;
        df.as_single_chunk();
        Ok(df)
    }

    /// Apply a closure over the groups as a new DataFrame.
    pub fn apply<F>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(DataFrame) -> Result<DataFrame> + Send + Sync,
    {
        let df = self.prepare_apply()?;
        let dfs = self
            .get_groups()
            .idx_ref()
            .iter()
            .map(|t| {
                let sub_df = unsafe { df.take_iter_unchecked(t.1.iter().map(|i| *i as usize)) };
                f(sub_df)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut df = accumulate_dataframes_vertical(dfs)?;
        df.as_single_chunk();
        Ok(df)
    }
}

#[derive(Copy, Clone, Debug)]
pub enum GroupByMethod {
    Min,
    Max,
    Median,
    Mean,
    First,
    Last,
    Sum,
    Groups,
    NUnique,
    Quantile(f64, QuantileInterpolOptions),
    Count,
    List,
    Std,
    Var,
}

// Formatting functions used in eager and lazy code for renaming grouped columns
pub fn fmt_groupby_column(name: &str, method: GroupByMethod) -> String {
    use GroupByMethod::*;
    match method {
        Min => format!("{}_min", name),
        Max => format!("{}_max", name),
        Median => format!("{}_median", name),
        Mean => format!("{}_mean", name),
        First => format!("{}_first", name),
        Last => format!("{}_last", name),
        Sum => format!("{}_sum", name),
        Groups => "groups".to_string(),
        NUnique => format!("{}_n_unique", name),
        Count => format!("{}_count", name),
        List => format!("{}_agg_list", name),
        Quantile(quantile, _interpol) => format!("{}_quantile_{:.2}", name, quantile),
        Std => format!("{}_agg_std", name),
        Var => format!("{}_agg_var", name),
    }
}

#[cfg(test)]
mod test {
    use crate::frame::groupby::{groupby, groupby_threaded_num};
    use crate::prelude::*;
    use crate::utils::split_ca;
    use num::traits::FloatConst;

    #[test]
    #[cfg(feature = "dtype-date")]
    #[cfg_attr(miri, ignore)]
    fn test_group_by() -> Result<()> {
        let s0 = Series::new(
            "date",
            &[
                "2020-08-21",
                "2020-08-21",
                "2020-08-22",
                "2020-08-23",
                "2020-08-22",
            ],
        );
        let s1 = Series::new("temp", [20, 10, 7, 9, 1]);
        let s2 = Series::new("rain", [0.2, 0.1, 0.3, 0.1, 0.01]);
        let df = DataFrame::new(vec![s0, s1, s2]).unwrap();

        let out = df.groupby_stable(["date"])?.select(["temp"]).count()?;
        assert_eq!(
            out.column("temp_count")?,
            &Series::new("temp_count", [2 as IdxSize, 2, 1])
        );

        // Select multiple
        let out = df
            .groupby_stable(["date"])?
            .select(&["temp", "rain"])
            .mean()?;
        assert_eq!(
            out.column("temp_mean")?,
            &Series::new("temp_mean", [15.0f64, 4.0, 9.0])
        );

        // Group by multiple
        let out = df
            .groupby_stable(&["date", "temp"])?
            .select(["rain"])
            .mean()?;
        assert!(out.column("rain_mean").is_ok());

        let out = df.groupby_stable(["date"])?.select(["temp"]).sum()?;
        assert_eq!(
            out.column("temp_sum")?,
            &Series::new("temp_sum", [30, 8, 9])
        );

        // implicit select all and only aggregate on methods that support that aggregation
        let gb = df.groupby(["date"]).unwrap().n_unique().unwrap();
        // check the group by column is filtered out.
        assert_eq!(gb.width(), 3);
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_static_groupby_by_12_columns() {
        // Build GroupBy DataFrame.
        let s0 = Series::new("G1", ["A", "A", "B", "B", "C"].as_ref());
        let s1 = Series::new("N", [1, 2, 2, 4, 2].as_ref());
        let s2 = Series::new("G2", ["k", "l", "m", "m", "l"].as_ref());
        let s3 = Series::new("G3", ["a", "b", "c", "c", "d"].as_ref());
        let s4 = Series::new("G4", ["1", "2", "3", "3", "4"].as_ref());
        let s5 = Series::new("G5", ["X", "Y", "Z", "Z", "W"].as_ref());
        let s6 = Series::new("G6", [false, true, true, true, false].as_ref());
        let s7 = Series::new("G7", ["r", "x", "q", "q", "o"].as_ref());
        let s8 = Series::new("G8", ["R", "X", "Q", "Q", "O"].as_ref());
        let s9 = Series::new("G9", [1, 2, 3, 3, 4].as_ref());
        let s10 = Series::new("G10", [".", "!", "?", "?", "/"].as_ref());
        let s11 = Series::new("G11", ["(", ")", "@", "@", "$"].as_ref());
        let s12 = Series::new("G12", ["-", "_", ";", ";", ","].as_ref());

        let df =
            DataFrame::new(vec![s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]).unwrap();

        let adf = df
            .groupby(&[
                "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12",
            ])
            .unwrap()
            .select(["N"])
            .sum()
            .unwrap();

        assert_eq!(
            Vec::from(&adf.column("N_sum").unwrap().i32().unwrap().sort(false)),
            &[Some(1), Some(2), Some(2), Some(6)]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_dynamic_groupby_by_13_columns() {
        // The content for every groupby series.
        let series_content = ["A", "A", "B", "B", "C"];

        // The name of every groupby series.
        let series_names = [
            "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13",
        ];

        // Vector to contain every series.
        let mut series = Vec::with_capacity(14);

        // Create a series for every group name.
        for series_name in &series_names {
            let serie = Series::new(series_name, series_content.as_ref());
            series.push(serie);
        }

        // Create a series for the aggregation column.
        let serie = Series::new("N", [1, 2, 3, 3, 4].as_ref());
        series.push(serie);

        // Creat the dataframe with the computed series.
        let df = DataFrame::new(series).unwrap();

        // Compute the aggregated DataFrame by the 13 columns defined in `series_names`.
        let adf = df
            .groupby(&series_names)
            .unwrap()
            .select(["N"])
            .sum()
            .unwrap();

        // Check that the results of the group-by are correct. The content of every column
        // is equal, then, the grouped columns shall be equal and in the same order.
        for series_name in &series_names {
            assert_eq!(
                Vec::from(&adf.column(series_name).unwrap().utf8().unwrap().sort(false)),
                &[Some("A"), Some("B"), Some("C")]
            );
        }

        // Check the aggregated column is the expected one.
        assert_eq!(
            Vec::from(&adf.column("N_sum").unwrap().i32().unwrap().sort(false)),
            &[Some(3), Some(4), Some(6)]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_groupby_floats() {
        let df = df! {"flt" => [1., 1., 2., 2., 3.],
                    "val" => [1, 1, 1, 1, 1]
        }
        .unwrap();
        let res = df.groupby(["flt"]).unwrap().sum().unwrap();
        let res = res.sort(["flt"], false).unwrap();
        assert_eq!(
            Vec::from(res.column("val_sum").unwrap().i32().unwrap()),
            &[Some(2), Some(2), Some(1)]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[cfg(feature = "dtype-categorical")]
    fn test_groupby_categorical() {
        let mut df = df! {"foo" => ["a", "a", "b", "b", "c"],
                    "ham" => ["a", "a", "b", "b", "c"],
                    "bar" => [1, 1, 1, 1, 1]
        }
        .unwrap();

        df.apply("foo", |s| s.cast(&DataType::Categorical(None)).unwrap())
            .unwrap();

        // check multiple keys and categorical
        let res = df
            .groupby_stable(["foo", "ham"])
            .unwrap()
            .select(["bar"])
            .sum()
            .unwrap();

        assert_eq!(
            Vec::from(res.column("bar_sum").unwrap().i32().unwrap()),
            &[Some(2), Some(2), Some(1)]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_groupby_apply() {
        let df = df! {
            "a" => [1, 1, 2, 2, 2],
            "b" => [1, 2, 3, 4, 5]
        }
        .unwrap();

        let out = df.groupby(["a"]).unwrap().apply(Ok).unwrap();
        assert!(out.sort(["b"], false).unwrap().frame_equal(&df));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_groupby_threaded() {
        for slice in &[
            vec![1, 2, 3, 4, 4, 4, 2, 1],
            vec![1, 2, 3, 4, 4, 4, 2, 1, 1],
            vec![1, 2, 3, 4, 4, 4],
        ] {
            let ca = UInt32Chunked::new("", slice);
            let split = split_ca(&ca, 4).unwrap();

            let a = groupby(ca.into_iter(), true).into_idx();

            let keys = split.iter().map(|ca| ca.cont_slice().unwrap()).collect();
            let b = groupby_threaded_num(keys, 0, split.len() as u64, true).into_idx();

            assert_eq!(a, b);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_groupby_null_handling() -> Result<()> {
        let df = df!(
            "a" => ["a", "a", "a", "b", "b"],
            "b" => [Some(1), Some(2), None, None, Some(1)]
        )?;
        let out = df.groupby_stable(["a"])?.mean()?;

        assert_eq!(
            Vec::from(out.column("b_mean")?.f64()?),
            &[Some(1.5), Some(1.0)]
        );
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_groupby_var() -> Result<()> {
        // check variance and proper coercion to f64
        let df = df![
            "g" => ["foo", "foo", "bar"],
            "flt" => [1.0, 2.0, 3.0],
            "int" => [1, 2, 3]
        ]?;

        let out = df.groupby(["g"])?.select(["int"]).var()?;
        assert_eq!(
            out.column("int_agg_var")?.f64()?.sort(false).get(0),
            Some(0.5)
        );
        let out = df.groupby(["g"])?.select(["int"]).std()?;
        let val = out
            .column("int_agg_std")?
            .f64()?
            .sort(false)
            .get(0)
            .unwrap();
        let expected = f64::FRAC_1_SQRT_2();
        assert!((val - expected).abs() < 0.000001);
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[cfg(feature = "dtype-categorical")]
    fn test_groupby_null_group() -> Result<()> {
        // check if null is own group
        let mut df = df![
            "g" => [Some("foo"), Some("foo"), Some("bar"), None, None],
            "flt" => [1.0, 2.0, 3.0, 1.0, 1.0],
            "int" => [1, 2, 3, 1, 1]
        ]?;

        df.try_apply("g", |s| s.cast(&DataType::Categorical(None)))?;

        let _ = df.groupby(["g"])?.sum()?;
        Ok(())
    }
}
