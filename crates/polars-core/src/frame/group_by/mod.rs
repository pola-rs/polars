use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use num_traits::NumCast;
use polars_compute::rolling::QuantileMethod;
use polars_utils::broadcast::broadcast_len;
use polars_utils::format_pl_smallstr;
use polars_utils::hashing::DirtyHash;
use rayon::prelude::*;

use self::hashing::*;
use crate::prelude::*;
use crate::runtime::RAYON;
use crate::utils::{_set_partition_size, accumulate_dataframes_vertical};

pub mod aggregations;
pub(crate) mod hashing;
mod into_groups;
mod position;

pub use into_groups::*;
pub use position::*;

use crate::chunked_array::ops::row_encode::{
    encode_rows_unordered, encode_rows_vertical_par_unordered,
};

impl DataFrame {
    pub fn group_by_with_series(
        &self,
        mut by: Vec<Column>,
        multithreaded: bool,
        sorted: bool,
    ) -> PolarsResult<GroupBy<'_>> {
        polars_ensure!(
            !by.is_empty(),
            ComputeError: "at least one key is required in a group_by operation"
        );

        // Ensure all 'by' columns have the same common_height
        // The condition self.width > 0 ensures we can still call this on a
        // dummy dataframe where we provide the keys
        let common_height = if self.width() > 0 {
            self.height()
        } else {
            broadcast_len(by.iter()).context("group_by key")?
        };
        for by_key in by.iter_mut() {
            by_key
                .broadcast_in_place_to(common_height)
                .context("group_by keys should have the same length as the DataFrame")?;
        }

        let groups = if by.len() == 1 {
            let column = &by[0];
            column
                .as_materialized_series()
                .group_tuples(multithreaded, sorted)
        } else if by.iter().any(|s| s.dtype().is_object()) {
            #[cfg(feature = "object")]
            {
                let mut df = DataFrame::new(self.height(), by.clone()).unwrap();
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
                let groups = if self.height() == 0 {
                    vec![]
                } else {
                    vec![[0, self.height() as IdxSize]]
                };

                Ok(GroupsType::new_slice(groups, false, true))
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
        Ok(GroupBy::new(self, by, groups?.into_sliceable(), None))
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
    pub fn group_by<I, S>(&self, by: I) -> PolarsResult<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let selected_keys = self.select_to_vec(by)?;
        self.group_by_with_series(selected_keys, true, false)
    }

    /// Group DataFrame using a Series column.
    /// The groups are ordered by their smallest row index.
    pub fn group_by_stable<I, S>(&self, by: I) -> PolarsResult<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let selected_keys = self.select_to_vec(by)?;
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
/// let s1 = Series::new("temp".into(), [20, 10, 7, 9, 1]);
/// // create rain series
/// let s2 = Series::new("rain".into(), [0.2, 0.1, 0.3, 0.1, 0.01]);
/// // create a new DataFrame
/// let df = DataFrame::new_infer_height(vec![s0, s1, s2]).unwrap();
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
pub struct GroupBy<'a> {
    pub df: &'a DataFrame,
    pub(crate) selected_keys: Vec<Column>,
    // [first idx, [other idx]]
    groups: GroupPositions,
    // columns selected for aggregation
    pub(crate) selected_agg: Option<Vec<PlSmallStr>>,
}

impl<'a> GroupBy<'a> {
    pub fn new(
        df: &'a DataFrame,
        by: Vec<Column>,
        groups: GroupPositions,
        selected_agg: Option<Vec<PlSmallStr>>,
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
    pub fn select<I: IntoIterator<Item = S>, S: Into<PlSmallStr>>(mut self, selection: I) -> Self {
        self.selected_agg = Some(selection.into_iter().map(|s| s.into()).collect());
        self
    }

    /// Get the internal representation of the GroupBy operation.
    /// The Vec returned contains:
    ///     (first_idx, [`Vec<indexes>`])
    ///     Where second value in the tuple is a vector with all matching indexes.
    pub fn get_groups(&self) -> &GroupPositions {
        &self.groups
    }

    /// Get the internal representation of the GroupBy operation.
    /// The Vec returned contains:
    ///     (first_idx, [`Vec<indexes>`])
    ///     Where second value in the tuple is a vector with all matching indexes.
    ///
    /// # Safety
    /// Groups should always be in bounds of the `DataFrame` hold by this [`GroupBy`].
    /// If you mutate it, you must hold that invariant.
    pub unsafe fn get_groups_mut(&mut self) -> &mut GroupPositions {
        &mut self.groups
    }

    pub fn into_groups(self) -> GroupPositions {
        self.groups
    }

    pub fn keys_sliced(&self, slice: Option<(i64, usize)>) -> Vec<Column> {
        #[allow(unused_assignments)]
        // needed to keep the lifetimes valid for this scope
        let mut groups_owned = None;

        let groups = if let Some((offset, len)) = slice {
            groups_owned = Some(self.groups.slice(offset, len));
            groups_owned.as_deref().unwrap()
        } else {
            &self.groups
        };
        RAYON.install(|| {
            self.selected_keys
                .par_iter()
                .map(Column::as_materialized_series)
                .map(|s| {
                    match groups {
                        GroupsType::Idx(groups) => {
                            // SAFETY: groups are always in bounds.
                            let mut out = unsafe { s.take_slice_unchecked(groups.first()) };
                            if groups.sorted_by_first_idx {
                                out.set_sorted_flag(s.is_sorted_flag());
                            };
                            out
                        },
                        GroupsType::Slice {
                            groups,
                            overlapping,
                            monotonic: _,
                        } => {
                            if *overlapping && !groups.is_empty() {
                                // Groups can be sliced.
                                let offset = groups[0][0];
                                let [upper_offset, upper_len] = groups[groups.len() - 1];
                                return s.slice(
                                    offset as i64,
                                    ((upper_offset + upper_len) - offset) as usize,
                                );
                            }

                            let indices = groups
                                .iter()
                                .map(|&[first, _len]| first)
                                .collect_ca(PlSmallStr::EMPTY);
                            // SAFETY: groups are always in bounds.
                            let mut out = unsafe { s.take_unchecked(&indices) };
                            // Sliced groups are always in order of discovery.
                            out.set_sorted_flag(s.is_sorted_flag());
                            out
                        },
                    }
                })
                .map(Column::from)
                .collect()
        })
    }

    pub fn keys(&self) -> Vec<Column> {
        self.keys_sliced(None)
    }

    fn prepare_agg(&self) -> PolarsResult<(Vec<Column>, Vec<Column>)> {
        let keys = self.keys();

        let agg_col = match &self.selected_agg {
            Some(selection) => self.df.select_to_vec(selection),
            None => {
                let by: Vec<_> = self.selected_keys.iter().map(|s| s.name()).collect();
                let selection = self
                    .df
                    .columns()
                    .iter()
                    .map(|s| s.name())
                    .filter(|a| !by.contains(a))
                    .cloned()
                    .collect::<Vec<_>>();

                self.df.select_to_vec(selection.as_slice())
            },
        }?;

        Ok((keys, agg_col))
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
                agg_col.name().as_str(),
                GroupByMethod::Count {
                    include_nulls: true,
                },
            );
            let mut ca = self.groups.group_count();
            ca.rename(new_name);
            cols.push(ca.into_column());
        }
        DataFrame::new_infer_height(cols)
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
        column.rename(new_name);
        cols.push(column.into_column());
        DataFrame::new_infer_height(cols)
    }

    fn prepare_apply(&self) -> PolarsResult<DataFrame> {
        if let Some(agg) = &self.selected_agg {
            if agg.is_empty() {
                Ok(self.df.clone())
            } else {
                let mut new_cols = Vec::with_capacity(self.selected_keys.len() + agg.len());
                new_cols.extend_from_slice(&self.selected_keys);
                let cols = self.df.select_to_vec(agg.as_slice())?;
                new_cols.extend(cols);
                Ok(unsafe { DataFrame::new_unchecked(self.df.height(), new_cols) })
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
        polars_ensure!(self.df.height() > 0, ComputeError: "cannot group_by + apply on empty 'DataFrame'");
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
        df.rechunk_mut_par();
        Ok(df)
    }

    /// Apply a closure over the groups as a new [`DataFrame`].
    pub fn apply<F>(&self, f: F) -> PolarsResult<DataFrame>
    where
        F: FnMut(DataFrame) -> PolarsResult<DataFrame> + Send + Sync,
    {
        self.apply_sliced(None, f, None)
    }

    pub fn apply_sliced<F>(
        &self,
        slice: Option<(i64, usize)>,
        mut f: F,
        schema: Option<&SchemaRef>,
    ) -> PolarsResult<DataFrame>
    where
        F: FnMut(DataFrame) -> PolarsResult<DataFrame> + Send + Sync,
    {
        if self.df.height() == 0 {
            // return empty dataframe with correct schema
            if let Some(schema) = schema {
                return Ok(DataFrame::empty_with_arc_schema(schema.clone()));
            }

            polars_bail!(ComputeError: "cannot group_by + apply on empty 'DataFrame'");
        }

        let df = self.prepare_apply()?;
        let max_height = if let Some((offset, len)) = slice {
            offset.try_into().unwrap_or(usize::MAX).saturating_add(len)
        } else {
            usize::MAX
        };
        let mut height = 0;
        let mut dfs = Vec::with_capacity(self.get_groups().len());
        for g in self.get_groups().iter() {
            // SAFETY: groups are in bounds.
            let sub_df = unsafe { take_df(&df, g) };
            let df = f(sub_df)?;
            height += df.height();
            dfs.push(df);

            // Even if max_height is zero we need at least one df, so check
            // after first push.
            if height >= max_height {
                break;
            }
        }

        let mut df = accumulate_dataframes_vertical(dfs)?;
        if let Some((offset, len)) = slice {
            df = df.slice(offset, len);
        }
        Ok(df)
    }

    pub fn sliced(mut self, slice: Option<(i64, usize)>) -> Self {
        match slice {
            None => self,
            Some((offset, length)) => {
                self.groups = self.groups.slice(offset, length);
                self.selected_keys = self.keys_sliced(slice);
                self
            },
        }
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
    FirstNonNull,
    Last,
    LastNonNull,
    Item { allow_empty: bool },
    Sum,
    Groups,
    NUnique,
    Quantile(f64, QuantileMethod),
    Count { include_nulls: bool },
    Implode { maintain_order: bool },
    Std(u8),
    Var(u8),
    ArgMin,
    ArgMax,
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
            FirstNonNull => "first_non_null",
            Last => "last",
            LastNonNull => "last_non_null",
            Item { .. } => "item",
            Sum => "sum",
            Groups => "groups",
            NUnique => "n_unique",
            Quantile(_, _) => "quantile",
            Count { .. } => "count",
            Implode { .. } => "implode",
            Std(_) => "std",
            Var(_) => "var",
            ArgMin => "arg_min",
            ArgMax => "arg_max",
        };
        write!(f, "{s}")
    }
}

// Formatting functions used in eager and lazy code for renaming grouped columns
pub fn fmt_group_by_column(name: &str, method: GroupByMethod) -> PlSmallStr {
    use GroupByMethod::*;
    match method {
        Min => format_pl_smallstr!("{name}_min"),
        Max => format_pl_smallstr!("{name}_max"),
        NanMin => format_pl_smallstr!("{name}_nan_min"),
        NanMax => format_pl_smallstr!("{name}_nan_max"),
        Median => format_pl_smallstr!("{name}_median"),
        Mean => format_pl_smallstr!("{name}_mean"),
        First => format_pl_smallstr!("{name}_first"),
        FirstNonNull => format_pl_smallstr!("{name}_first_non_null"),
        Last => format_pl_smallstr!("{name}_last"),
        LastNonNull => format_pl_smallstr!("{name}_last_non_null"),
        Item { .. } => format_pl_smallstr!("{name}_item"),
        Sum => format_pl_smallstr!("{name}_sum"),
        Groups => PlSmallStr::from_static("groups"),
        NUnique => format_pl_smallstr!("{name}_n_unique"),
        Count { .. } => format_pl_smallstr!("{name}_count"),
        Implode { .. } => format_pl_smallstr!("{name}_agg_list"),
        Quantile(quantile, _interpol) => format_pl_smallstr!("{name}_quantile_{quantile:.2}"),
        Std(_) => format_pl_smallstr!("{name}_agg_std"),
        Var(_) => format_pl_smallstr!("{name}_agg_var"),
        ArgMin => format_pl_smallstr!("{name}_arg_min"),
        ArgMax => format_pl_smallstr!("{name}_arg_max"),
    }
}
