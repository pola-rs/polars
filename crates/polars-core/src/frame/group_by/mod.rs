use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use num_traits::NumCast;
use polars_utils::hashing::DirtyHash;
use rayon::prelude::*;

use self::hashing::*;
use crate::prelude::*;
use crate::utils::{_set_partition_size, accumulate_dataframes_vertical};
use crate::POOL;

pub mod aggregations;
pub mod expr;
pub(crate) mod hashing;
mod into_groups;
mod perfect;
mod proxy;

pub use into_groups::*;
pub use proxy::*;

use crate::prelude::sort::arg_sort_multiple::{
    encode_rows_unordered, encode_rows_vertical_par_unordered,
};

impl DataFrame {
    pub fn group_by_with_series(
        &self,
        mut by: Vec<Series>,
        multithreaded: bool,
        sorted: bool,
    ) -> PolarsResult<GroupBy> {
        polars_ensure!(
            !by.is_empty(),
            ComputeError: "at least one key is required in a group_by operation"
        );
        let minimal_by_len = by.iter().map(|s| s.len()).min().expect("at least 1 key");
        let df_height = self.height();

        // we only throw this error if self.width > 0
        // so that we can still call this on a dummy dataframe where we provide the keys
        if (minimal_by_len != df_height) && (self.width() > 0) {
            polars_ensure!(
                minimal_by_len == 1,
                ShapeMismatch: "series used as keys should have the same length as the DataFrame"
            );
            for by_key in by.iter_mut() {
                if by_key.len() == minimal_by_len {
                    *by_key = by_key.new_from_index(0, df_height)
                }
            }
        };

        let groups = if by.len() == 1 {
            let series = &by[0];
            series.group_tuples(multithreaded, sorted)
        } else if by.iter().any(|s| s.dtype().is_object()) {
            #[cfg(feature = "object")]
            {
                let mut df = DataFrame::new(by.clone()).unwrap();
                let n = df.height();
                let rows = df.to_av_rows();
                let iter = (0..n).map(|i| rows.get(i));
                Ok(group_by(iter, sorted))
            }
            #[cfg(not(feature = "object"))]
            {
                unreachable!()
            }
        } else {
            // Skip null dtype.
            let by = by
                .iter()
                .filter(|s| !s.dtype().is_null())
                .cloned()
                .collect::<Vec<_>>();
            if by.is_empty() {
                let groups = if self.is_empty() {
                    vec![]
                } else {
                    vec![[0, self.height() as IdxSize]]
                };
                Ok(GroupsProxy::Slice {
                    groups,
                    rolling: false,
                })
            } else {
                let rows = if multithreaded {
                    encode_rows_vertical_par_unordered(&by)
                } else {
                    encode_rows_unordered(&by)
                }?
                .into_series();
                rows.group_tuples(multithreaded, sorted)
            }
        };
        Ok(GroupBy::new(self, by, groups?, None))
    }

    /// Group DataFrame using a Series column.
    ///
    /// # Example
    ///
    /// ```
    /// use polars_core::prelude::*;
    /// fn group_by_sum(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     df.group_by(["column_name"])?
    ///     .select(["agg_column_name"])
    ///     .sum()
    /// }
    /// ```
    pub fn group_by<I, S>(&self, by: I) -> PolarsResult<GroupBy>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let selected_keys = self.select_series(by)?;
        self.group_by_with_series(selected_keys, true, false)
    }

    /// Group DataFrame using a Series column.
    /// The groups are ordered by their smallest row index.
    pub fn group_by_stable<I, S>(&self, by: I) -> PolarsResult<GroupBy>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let selected_keys = self.select_series(by)?;
        self.group_by_with_series(selected_keys, true, true)
    }
}

/// Returned by a group_by operation on a DataFrame. This struct supports
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
/// | Date       | i32  | f64  |
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
    pub df: &'df DataFrame,
    pub(crate) selected_keys: Vec<Series>,
    // [first idx, [other idx]]
    groups: GroupsProxy,
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
    ///     (first_idx, [`Vec<indexes>`])
    ///     Where second value in the tuple is a vector with all matching indexes.
    pub fn get_groups(&self) -> &GroupsProxy {
        &self.groups
    }

    /// Get the internal representation of the GroupBy operation.
    /// The Vec returned contains:
    ///     (first_idx, [`Vec<indexes>`])
    ///     Where second value in the tuple is a vector with all matching indexes.
    ///
    /// # Safety
    /// Groups should always be in bounds of the `DataFrame` hold by this `[GroupBy]`.
    /// If you mutate it, you must hold that invariant.
    pub unsafe fn get_groups_mut(&mut self) -> &mut GroupsProxy {
        &mut self.groups
    }

    pub fn take_groups(self) -> GroupsProxy {
        self.groups
    }

    pub fn take_groups_mut(&mut self) -> GroupsProxy {
        std::mem::take(&mut self.groups)
    }

    pub fn keys_sliced(&self, slice: Option<(i64, usize)>) -> Vec<Series> {
        #[allow(unused_assignments)]
        // needed to keep the lifetimes valid for this scope
        let mut groups_owned = None;

        let groups = if let Some((offset, len)) = slice {
            groups_owned = Some(self.groups.slice(offset, len));
            groups_owned.as_deref().unwrap()
        } else {
            &self.groups
        };

        POOL.install(|| {
            self.selected_keys
                .par_iter()
                .map(|s| {
                    match groups {
                        GroupsProxy::Idx(groups) => {
                            // SAFETY: groups are always in bounds.
                            let mut out = unsafe { s.take_slice_unchecked(groups.first()) };
                            if groups.sorted {
                                out.set_sorted_flag(s.is_sorted_flag());
                            };
                            out
                        },
                        GroupsProxy::Slice { groups, rolling } => {
                            if *rolling && !groups.is_empty() {
                                // Groups can be sliced.
                                let offset = groups[0][0];
                                let [upper_offset, upper_len] = groups[groups.len() - 1];
                                return s.slice(
                                    offset as i64,
                                    ((upper_offset + upper_len) - offset) as usize,
                                );
                            }

                            let indices = groups.iter().map(|&[first, _len]| first).collect_ca("");
                            // SAFETY: groups are always in bounds.
                            let mut out = unsafe { s.take_unchecked(&indices) };
                            // Sliced groups are always in order of discovery.
                            out.set_sorted_flag(s.is_sorted_flag());
                            out
                        },
                    }
                })
                .collect()
        })
    }

    pub fn keys(&self) -> Vec<Series> {
        self.keys_sliced(None)
    }

    fn prepare_agg(&self) -> PolarsResult<(Vec<Series>, Vec<Series>)> {
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
            },
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
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.group_by(["date"])?.select(&["temp", "rain"]).mean()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+-----------+-----------+
    /// | date       | temp_mean | rain_mean |
    /// | ---        | ---       | ---       |
    /// | Date       | f64       | f64       |
    /// +============+===========+===========+
    /// | 2020-08-23 | 9         | 0.1       |
    /// +------------+-----------+-----------+
    /// | 2020-08-22 | 4         | 0.155     |
    /// +------------+-----------+-----------+
    /// | 2020-08-21 | 15        | 0.15      |
    /// +------------+-----------+-----------+
    /// ```
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn mean(&self) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;

        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(agg_col.name(), GroupByMethod::Mean);
            let mut agg = unsafe { agg_col.agg_mean(&self.groups) };
            agg.rename(&new_name);
            cols.push(agg);
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped series and compute the sum per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.group_by(["date"])?.select(["temp"]).sum()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+----------+
    /// | date       | temp_sum |
    /// | ---        | ---      |
    /// | Date       | i32      |
    /// +============+==========+
    /// | 2020-08-23 | 9        |
    /// +------------+----------+
    /// | 2020-08-22 | 8        |
    /// +------------+----------+
    /// | 2020-08-21 | 30       |
    /// +------------+----------+
    /// ```
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn sum(&self) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;

        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(agg_col.name(), GroupByMethod::Sum);
            let mut agg = unsafe { agg_col.agg_sum(&self.groups) };
            agg.rename(&new_name);
            cols.push(agg);
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped series and compute the minimal value per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.group_by(["date"])?.select(["temp"]).min()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+----------+
    /// | date       | temp_min |
    /// | ---        | ---      |
    /// | Date       | i32      |
    /// +============+==========+
    /// | 2020-08-23 | 9        |
    /// +------------+----------+
    /// | 2020-08-22 | 1        |
    /// +------------+----------+
    /// | 2020-08-21 | 10       |
    /// +------------+----------+
    /// ```
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn min(&self) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(agg_col.name(), GroupByMethod::Min);
            let mut agg = unsafe { agg_col.agg_min(&self.groups) };
            agg.rename(&new_name);
            cols.push(agg);
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped series and compute the maximum value per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.group_by(["date"])?.select(["temp"]).max()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+----------+
    /// | date       | temp_max |
    /// | ---        | ---      |
    /// | Date       | i32      |
    /// +============+==========+
    /// | 2020-08-23 | 9        |
    /// +------------+----------+
    /// | 2020-08-22 | 7        |
    /// +------------+----------+
    /// | 2020-08-21 | 20       |
    /// +------------+----------+
    /// ```
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn max(&self) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(agg_col.name(), GroupByMethod::Max);
            let mut agg = unsafe { agg_col.agg_max(&self.groups) };
            agg.rename(&new_name);
            cols.push(agg);
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped `Series` and find the first value per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.group_by(["date"])?.select(["temp"]).first()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+------------+
    /// | date       | temp_first |
    /// | ---        | ---        |
    /// | Date       | i32        |
    /// +============+============+
    /// | 2020-08-23 | 9          |
    /// +------------+------------+
    /// | 2020-08-22 | 7          |
    /// +------------+------------+
    /// | 2020-08-21 | 20         |
    /// +------------+------------+
    /// ```
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn first(&self) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(agg_col.name(), GroupByMethod::First);
            let mut agg = unsafe { agg_col.agg_first(&self.groups) };
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
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.group_by(["date"])?.select(["temp"]).last()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+------------+
    /// | date       | temp_last |
    /// | ---        | ---        |
    /// | Date       | i32        |
    /// +============+============+
    /// | 2020-08-23 | 9          |
    /// +------------+------------+
    /// | 2020-08-22 | 1          |
    /// +------------+------------+
    /// | 2020-08-21 | 10         |
    /// +------------+------------+
    /// ```
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn last(&self) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(agg_col.name(), GroupByMethod::Last);
            let mut agg = unsafe { agg_col.agg_last(&self.groups) };
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
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.group_by(["date"])?.select(["temp"]).n_unique()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+---------------+
    /// | date       | temp_n_unique |
    /// | ---        | ---           |
    /// | Date       | u32           |
    /// +============+===============+
    /// | 2020-08-23 | 1             |
    /// +------------+---------------+
    /// | 2020-08-22 | 2             |
    /// +------------+---------------+
    /// | 2020-08-21 | 2             |
    /// +------------+---------------+
    /// ```
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn n_unique(&self) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(agg_col.name(), GroupByMethod::NUnique);
            let mut agg = unsafe { agg_col.agg_n_unique(&self.groups) };
            agg.rename(&new_name);
            cols.push(agg.into_series());
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped [`Series`] and determine the quantile per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// # use arrow::legacy::prelude::QuantileInterpolOptions;
    ///
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.group_by(["date"])?.select(["temp"]).quantile(0.2, QuantileInterpolOptions::default())
    /// }
    /// ```
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn quantile(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<DataFrame> {
        polars_ensure!(
            (0.0..=1.0).contains(&quantile),
            ComputeError: "`quantile` should be within 0.0 and 1.0"
        );
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name =
                fmt_group_by_column(agg_col.name(), GroupByMethod::Quantile(quantile, interpol));
            let mut agg = unsafe { agg_col.agg_quantile(&self.groups, quantile, interpol) };
            agg.rename(&new_name);
            cols.push(agg.into_series());
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped [`Series`] and determine the median per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.group_by(["date"])?.select(["temp"]).median()
    /// }
    /// ```
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn median(&self) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(agg_col.name(), GroupByMethod::Median);
            let mut agg = unsafe { agg_col.agg_median(&self.groups) };
            agg.rename(&new_name);
            cols.push(agg.into_series());
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped [`Series`] and determine the variance per group.
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn var(&self, ddof: u8) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(agg_col.name(), GroupByMethod::Var(ddof));
            let mut agg = unsafe { agg_col.agg_var(&self.groups, ddof) };
            agg.rename(&new_name);
            cols.push(agg.into_series());
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped [`Series`] and determine the standard deviation per group.
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn std(&self, ddof: u8) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(agg_col.name(), GroupByMethod::Std(ddof));
            let mut agg = unsafe { agg_col.agg_std(&self.groups, ddof) };
            agg.rename(&new_name);
            cols.push(agg.into_series());
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped series and compute the number of values per group.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.group_by(["date"])?.select(["temp"]).count()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+------------+
    /// | date       | temp_count |
    /// | ---        | ---        |
    /// | Date       | u32        |
    /// +============+============+
    /// | 2020-08-23 | 1          |
    /// +------------+------------+
    /// | 2020-08-22 | 2          |
    /// +------------+------------+
    /// | 2020-08-21 | 2          |
    /// +------------+------------+
    /// ```
    pub fn count(&self) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;

        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(
                agg_col.name(),
                GroupByMethod::Count {
                    include_nulls: true,
                },
            );
            let mut ca = self.groups.group_count();
            ca.rename(&new_name);
            cols.push(ca.into_series());
        }
        DataFrame::new(cols)
    }

    /// Get the group_by group indexes.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.group_by(["date"])?.groups()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +--------------+------------+
    /// | date         | groups     |
    /// | ---          | ---        |
    /// | Date(days)   | list [u32] |
    /// +==============+============+
    /// | 2020-08-23   | "[3]"      |
    /// +--------------+------------+
    /// | 2020-08-22   | "[2, 4]"   |
    /// +--------------+------------+
    /// | 2020-08-21   | "[0, 1]"   |
    /// +--------------+------------+
    /// ```
    pub fn groups(&self) -> PolarsResult<DataFrame> {
        let mut cols = self.keys();
        let mut column = self.groups.as_list_chunked();
        let new_name = fmt_group_by_column("", GroupByMethod::Groups);
        column.rename(&new_name);
        cols.push(column.into_series());
        DataFrame::new(cols)
    }

    /// Aggregate the groups of the group_by operation into lists.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     // GroupBy and aggregate to Lists
    ///     df.group_by(["date"])?.select(["temp"]).agg_list()
    /// }
    /// ```
    /// Returns:
    ///
    /// ```text
    /// +------------+------------------------+
    /// | date       | temp_agg_list          |
    /// | ---        | ---                    |
    /// | Date       | list [i32]             |
    /// +============+========================+
    /// | 2020-08-23 | "[Some(9)]"            |
    /// +------------+------------------------+
    /// | 2020-08-22 | "[Some(7), Some(1)]"   |
    /// +------------+------------------------+
    /// | 2020-08-21 | "[Some(20), Some(10)]" |
    /// +------------+------------------------+
    /// ```
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn agg_list(&self) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(agg_col.name(), GroupByMethod::Implode);
            let mut agg = unsafe { agg_col.agg_list(&self.groups) };
            agg.rename(&new_name);
            cols.push(agg);
        }
        DataFrame::new(cols)
    }

    fn prepare_apply(&self) -> PolarsResult<DataFrame> {
        polars_ensure!(self.df.height() > 0, ComputeError: "cannot group_by + apply on empty 'DataFrame'");
        if let Some(agg) = &self.selected_agg {
            if agg.is_empty() {
                Ok(self.df.clone())
            } else {
                let mut new_cols = Vec::with_capacity(self.selected_keys.len() + agg.len());
                new_cols.extend_from_slice(&self.selected_keys);
                let cols = self.df.select_series(agg)?;
                new_cols.extend(cols);
                Ok(unsafe { DataFrame::new_no_checks(new_cols) })
            }
        } else {
            Ok(self.df.clone())
        }
    }

    /// Apply a closure over the groups as a new [`DataFrame`] in parallel.
    #[deprecated(since = "0.24.1", note = "use polars.lazy aggregations")]
    pub fn par_apply<F>(&self, f: F) -> PolarsResult<DataFrame>
    where
        F: Fn(DataFrame) -> PolarsResult<DataFrame> + Send + Sync,
    {
        let df = self.prepare_apply()?;
        let dfs = self
            .get_groups()
            .par_iter()
            .map(|g| {
                // SAFETY:
                // groups are in bounds
                let sub_df = unsafe { take_df(&df, g) };
                f(sub_df)
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        let mut df = accumulate_dataframes_vertical(dfs)?;
        df.as_single_chunk_par();
        Ok(df)
    }

    /// Apply a closure over the groups as a new [`DataFrame`].
    pub fn apply<F>(&self, mut f: F) -> PolarsResult<DataFrame>
    where
        F: FnMut(DataFrame) -> PolarsResult<DataFrame> + Send + Sync,
    {
        let df = self.prepare_apply()?;
        let dfs = self
            .get_groups()
            .iter()
            .map(|g| {
                // SAFETY:
                // groups are in bounds
                let sub_df = unsafe { take_df(&df, g) };
                f(sub_df)
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        let mut df = accumulate_dataframes_vertical(dfs)?;
        df.as_single_chunk_par();
        Ok(df)
    }
}

unsafe fn take_df(df: &DataFrame, g: GroupsIndicator) -> DataFrame {
    match g {
        GroupsIndicator::Idx(idx) => df.take_slice_unchecked(idx.1),
        GroupsIndicator::Slice([first, len]) => df.slice(first as i64, len as usize),
    }
}

#[derive(Copy, Clone, Debug)]
pub enum GroupByMethod {
    Min,
    NanMin,
    Max,
    NanMax,
    Median,
    Mean,
    First,
    Last,
    Sum,
    Groups,
    NUnique,
    Quantile(f64, QuantileInterpolOptions),
    Count { include_nulls: bool },
    Implode,
    Std(u8),
    Var(u8),
}

impl Display for GroupByMethod {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use GroupByMethod::*;
        let s = match self {
            Min => "min",
            NanMin => "nan_min",
            Max => "max",
            NanMax => "nan_max",
            Median => "median",
            Mean => "mean",
            First => "first",
            Last => "last",
            Sum => "sum",
            Groups => "groups",
            NUnique => "n_unique",
            Quantile(_, _) => "quantile",
            Count { .. } => "count",
            Implode => "list",
            Std(_) => "std",
            Var(_) => "var",
        };
        write!(f, "{s}")
    }
}

// Formatting functions used in eager and lazy code for renaming grouped columns
pub fn fmt_group_by_column(name: &str, method: GroupByMethod) -> String {
    use GroupByMethod::*;
    match method {
        Min => format!("{name}_min"),
        Max => format!("{name}_max"),
        NanMin => format!("{name}_nan_min"),
        NanMax => format!("{name}_nan_max"),
        Median => format!("{name}_median"),
        Mean => format!("{name}_mean"),
        First => format!("{name}_first"),
        Last => format!("{name}_last"),
        Sum => format!("{name}_sum"),
        Groups => "groups".to_string(),
        NUnique => format!("{name}_n_unique"),
        Count { .. } => format!("{name}_count"),
        Implode => format!("{name}_agg_list"),
        Quantile(quantile, _interpol) => format!("{name}_quantile_{quantile:.2}"),
        Std(_) => format!("{name}_agg_std"),
        Var(_) => format!("{name}_agg_var"),
    }
}

#[cfg(test)]
mod test {
    use num_traits::FloatConst;

    use crate::prelude::*;

    #[test]
    #[cfg(feature = "dtype-date")]
    #[cfg_attr(miri, ignore)]
    fn test_group_by() -> PolarsResult<()> {
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

        let out = df.group_by_stable(["date"])?.select(["temp"]).count()?;
        assert_eq!(
            out.column("temp_count")?,
            &Series::new("temp_count", [2 as IdxSize, 2, 1])
        );

        // Use of deprecated mean() for testing purposes
        #[allow(deprecated)]
        // Select multiple
        let out = df
            .group_by_stable(["date"])?
            .select(["temp", "rain"])
            .mean()?;
        assert_eq!(
            out.column("temp_mean")?,
            &Series::new("temp_mean", [15.0f64, 4.0, 9.0])
        );

        // Use of deprecated `mean()` for testing purposes
        #[allow(deprecated)]
        // Group by multiple
        let out = df
            .group_by_stable(["date", "temp"])?
            .select(["rain"])
            .mean()?;
        assert!(out.column("rain_mean").is_ok());

        // Use of deprecated `sum()` for testing purposes
        #[allow(deprecated)]
        let out = df.group_by_stable(["date"])?.select(["temp"]).sum()?;
        assert_eq!(
            out.column("temp_sum")?,
            &Series::new("temp_sum", [30, 8, 9])
        );

        // Use of deprecated `n_unique()` for testing purposes
        #[allow(deprecated)]
        // implicit select all and only aggregate on methods that support that aggregation
        let gb = df.group_by(["date"]).unwrap().n_unique().unwrap();
        // check the group by column is filtered out.
        assert_eq!(gb.width(), 3);
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_static_group_by_by_12_columns() {
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

        // Use of deprecated `sum()` for testing purposes
        #[allow(deprecated)]
        let adf = df
            .group_by([
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
    fn test_dynamic_group_by_by_13_columns() {
        // The content for every group_by series.
        let series_content = ["A", "A", "B", "B", "C"];

        // The name of every group_by series.
        let series_names = [
            "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13",
        ];

        // Vector to contain every series.
        let mut series = Vec::with_capacity(14);

        // Create a series for every group name.
        for series_name in &series_names {
            let group_series = Series::new(series_name, series_content.as_ref());
            series.push(group_series);
        }

        // Create a series for the aggregation column.
        let agg_series = Series::new("N", [1, 2, 3, 3, 4].as_ref());
        series.push(agg_series);

        // Create the dataframe with the computed series.
        let df = DataFrame::new(series).unwrap();

        // Use of deprecated `sum()` for testing purposes
        #[allow(deprecated)]
        // Compute the aggregated DataFrame by the 13 columns defined in `series_names`.
        let adf = df
            .group_by(series_names)
            .unwrap()
            .select(["N"])
            .sum()
            .unwrap();

        // Check that the results of the group-by are correct. The content of every column
        // is equal, then, the grouped columns shall be equal and in the same order.
        for series_name in &series_names {
            assert_eq!(
                Vec::from(&adf.column(series_name).unwrap().str().unwrap().sort(false)),
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
    fn test_group_by_floats() {
        let df = df! {"flt" => [1., 1., 2., 2., 3.],
                    "val" => [1, 1, 1, 1, 1]
        }
        .unwrap();
        // Use of deprecated `sum()` for testing purposes
        #[allow(deprecated)]
        let res = df.group_by(["flt"]).unwrap().sum().unwrap();
        let res = res.sort(["flt"], SortMultipleOptions::default()).unwrap();
        assert_eq!(
            Vec::from(res.column("val_sum").unwrap().i32().unwrap()),
            &[Some(2), Some(2), Some(1)]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[cfg(feature = "dtype-categorical")]
    fn test_group_by_categorical() {
        let mut df = df! {"foo" => ["a", "a", "b", "b", "c"],
                    "ham" => ["a", "a", "b", "b", "c"],
                    "bar" => [1, 1, 1, 1, 1]
        }
        .unwrap();

        df.apply("foo", |s| {
            s.cast(&DataType::Categorical(None, Default::default()))
                .unwrap()
        })
        .unwrap();

        // Use of deprecated `sum()` for testing purposes
        #[allow(deprecated)]
        // check multiple keys and categorical
        let res = df
            .group_by_stable(["foo", "ham"])
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
    fn test_group_by_null_handling() -> PolarsResult<()> {
        let df = df!(
            "a" => ["a", "a", "a", "b", "b"],
            "b" => [Some(1), Some(2), None, None, Some(1)]
        )?;
        // Use of deprecated `mean()` for testing purposes
        #[allow(deprecated)]
        let out = df.group_by_stable(["a"])?.mean()?;

        assert_eq!(
            Vec::from(out.column("b_mean")?.f64()?),
            &[Some(1.5), Some(1.0)]
        );
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_group_by_var() -> PolarsResult<()> {
        // check variance and proper coercion to f64
        let df = df![
            "g" => ["foo", "foo", "bar"],
            "flt" => [1.0, 2.0, 3.0],
            "int" => [1, 2, 3]
        ]?;

        // Use of deprecated `sum()` for testing purposes
        #[allow(deprecated)]
        let out = df.group_by_stable(["g"])?.select(["int"]).var(1)?;

        assert_eq!(out.column("int_agg_var")?.f64()?.get(0), Some(0.5));
        // Use of deprecated `std()` for testing purposes
        #[allow(deprecated)]
        let out = df.group_by_stable(["g"])?.select(["int"]).std(1)?;
        let val = out.column("int_agg_std")?.f64()?.get(0).unwrap();
        let expected = f64::FRAC_1_SQRT_2();
        assert!((val - expected).abs() < 0.000001);
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[cfg(feature = "dtype-categorical")]
    fn test_group_by_null_group() -> PolarsResult<()> {
        // check if null is own group
        let mut df = df![
            "g" => [Some("foo"), Some("foo"), Some("bar"), None, None],
            "flt" => [1.0, 2.0, 3.0, 1.0, 1.0],
            "int" => [1, 2, 3, 1, 1]
        ]?;

        df.try_apply("g", |s| {
            s.cast(&DataType::Categorical(None, Default::default()))
        })?;

        // Use of deprecated `sum()` for testing purposes
        #[allow(deprecated)]
        let _ = df.group_by(["g"])?.sum()?;
        Ok(())
    }
}
