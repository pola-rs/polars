use self::hashing::*;
use crate::chunked_array::builder::PrimitiveChunkedBuilder;
use crate::frame::select::Selection;
use crate::prelude::*;
#[cfg(feature = "groupby_list")]
use crate::utils::Wrap;
use crate::utils::{
    accumulate_dataframes_vertical, copy_from_slice_unchecked, set_partition_size, split_ca, NoNull,
};
use crate::vector_hasher::{get_null_hash_value, AsU64, StrHash};
use crate::POOL;
use ahash::{CallHasher, RandomState};
use hashbrown::HashMap;
use num::NumCast;
use rayon::prelude::*;
use std::fmt::Debug;
use std::hash::Hash;
#[cfg(feature = "dtype-categorical")]
use std::ops::Deref;

pub mod aggregations;
pub(crate) mod hashing;
#[cfg(feature = "pivot")]
pub(crate) mod pivot;
#[cfg(feature = "downsample")]
pub mod resample;

pub type GroupTuples = Vec<(u32, Vec<u32>)>;
pub type GroupedMap<T> = HashMap<T, Vec<u32>, RandomState>;

/// Used to create the tuples for a groupby operation.
pub trait IntoGroupTuples {
    /// Create the tuples need for a groupby operation.
    ///     * The first value in the tuple is the first index of the group.
    ///     * The second value in the tuple is are the indexes of the groups including the first value.
    fn group_tuples(&self, _multithreaded: bool) -> GroupTuples {
        unimplemented!()
    }
}

fn group_multithreaded<T>(ca: &ChunkedArray<T>) -> bool {
    // TODO! change to something sensible
    ca.len() > 1000
}

fn num_group_tuples<T>(ca: &ChunkedArray<T>, multithreaded: bool) -> GroupTuples
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq + Send + AsU64,
    Option<T::Native>: AsU64,
{
    #[cfg(feature = "dtype-categorical")]
    let group_size_hint = if let Some(m) = &ca.categorical_map {
        ca.len() / m.len()
    } else {
        0
    };
    #[cfg(not(feature = "dtype-categorical"))]
    let group_size_hint = 0;
    if multithreaded && group_multithreaded(ca) {
        let n_partitions = set_partition_size() as u64;

        // use the arrays as iterators
        if ca.chunks.len() == 1 {
            if ca.null_count() == 0 {
                let keys = vec![ca.cont_slice().unwrap()];
                groupby_threaded_num(keys, group_size_hint, n_partitions)
            } else {
                let keys = ca
                    .downcast_iter()
                    .map(|arr| arr.into_iter().map(|x| x.copied()).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                groupby_threaded_num(keys, group_size_hint, n_partitions)
            }
            // use the polars-iterators
        } else if ca.null_count() == 0 {
            let keys = vec![ca.into_no_null_iter().collect::<Vec<_>>()];
            groupby_threaded_num(keys, group_size_hint, n_partitions)
        } else {
            let keys = vec![ca.into_iter().collect::<Vec<_>>()];
            groupby_threaded_num(keys, group_size_hint, n_partitions)
        }
    } else if ca.null_count() == 0 {
        groupby(ca.into_no_null_iter())
    } else {
        groupby(ca.into_iter())
    }
}

impl<T> IntoGroupTuples for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
{
    fn group_tuples(&self, multithreaded: bool) -> GroupTuples {
        match self.dtype() {
            DataType::UInt64 => {
                // convince the compiler that we are this type.
                let ca: &UInt64Chunked = unsafe {
                    &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt64Type>)
                };
                num_group_tuples(ca, multithreaded)
            }
            DataType::UInt32 => {
                // convince the compiler that we are this type.
                let ca: &UInt32Chunked = unsafe {
                    &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt32Type>)
                };
                num_group_tuples(ca, multithreaded)
            }
            DataType::Int64 | DataType::Float64 => {
                let ca = self.bit_repr_large();
                num_group_tuples(&ca, multithreaded)
            }
            DataType::Int32 | DataType::Float32 => {
                let ca = self.bit_repr_small();
                num_group_tuples(&ca, multithreaded)
            }
            _ => {
                let ca = self.cast(&DataType::UInt32).unwrap();
                let ca = ca.u32().unwrap();
                num_group_tuples(ca, multithreaded)
            }
        }
    }
}
impl IntoGroupTuples for BooleanChunked {
    fn group_tuples(&self, multithreaded: bool) -> GroupTuples {
        let ca = self.cast(&DataType::UInt32).unwrap();
        let ca = ca.u32().unwrap();
        ca.group_tuples(multithreaded)
    }
}

impl IntoGroupTuples for Utf8Chunked {
    fn group_tuples(&self, multithreaded: bool) -> GroupTuples {
        let hb = RandomState::default();
        let null_h = get_null_hash_value(hb.clone());

        if multithreaded {
            let n_partitions = set_partition_size();

            let split = split_ca(self, n_partitions).unwrap();

            let str_hashes = POOL.install(|| {
                split
                    .par_iter()
                    .map(|ca| {
                        ca.into_iter()
                            .map(|opt_s| {
                                let hash = match opt_s {
                                    Some(s) => str::get_hash(s, &hb),
                                    None => null_h,
                                };
                                StrHash::new(opt_s, hash)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
            groupby_threaded_num(str_hashes, 0, n_partitions as u64)
        } else {
            let str_hashes = self
                .into_iter()
                .map(|opt_s| {
                    let hash = match opt_s {
                        Some(s) => str::get_hash(s, &hb),
                        None => null_h,
                    };
                    StrHash::new(opt_s, hash)
                })
                .collect::<Vec<_>>();
            groupby(str_hashes.iter())
        }
    }
}

#[cfg(feature = "dtype-categorical")]
impl IntoGroupTuples for CategoricalChunked {
    fn group_tuples(&self, multithreaded: bool) -> GroupTuples {
        self.deref().group_tuples(multithreaded)
    }
}

impl IntoGroupTuples for ListChunked {
    #[cfg(feature = "groupby_list")]
    fn group_tuples(&self, _multithreaded: bool) -> GroupTuples {
        groupby(self.into_iter().map(|opt_s| opt_s.map(Wrap)))
    }
}

#[cfg(feature = "object")]
impl<T> IntoGroupTuples for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn group_tuples(&self, _multithreaded: bool) -> GroupTuples {
        groupby(self.into_iter())
    }
}

/// Used to tightly two 32 bit values and null information
/// Only the bit values matter, not the meaning of the bits
#[inline]
fn pack_u32_tuples(opt_l: Option<u32>, opt_r: Option<u32>) -> [u8; 9] {
    // 4 bytes for first value
    // 4 bytes for second value
    // last bytes' bits are used to indicate missing values
    let mut val = [0u8; 9];
    let s = &mut val;
    match (opt_l, opt_r) {
        (Some(l), Some(r)) => {
            // write to first 4 places
            unsafe { copy_from_slice_unchecked(&l.to_ne_bytes(), &mut s[..4]) }
            // write to second chunk of 4 places
            unsafe { copy_from_slice_unchecked(&r.to_ne_bytes(), &mut s[4..8]) }
            // leave last byte as is
        }
        (Some(l), None) => {
            unsafe { copy_from_slice_unchecked(&l.to_ne_bytes(), &mut s[..4]) }
            // set right null bit
            s[8] = 1;
        }
        (None, Some(r)) => {
            unsafe { copy_from_slice_unchecked(&r.to_ne_bytes(), &mut s[4..8]) }
            // set left null bit
            s[8] = 1 << 1;
        }
        (None, None) => {
            // set two null bits
            s[8] = 3;
        }
    }
    val
}

/// Used to tightly two 64 bit values and null information
/// Only the bit values matter, not the meaning of the bits
#[inline]
fn pack_u64_tuples(opt_l: Option<u64>, opt_r: Option<u64>) -> [u8; 17] {
    // 8 bytes for first value
    // 8 bytes for second value
    // last bytes' bits are used to indicate missing values
    let mut val = [0u8; 17];
    let s = &mut val;
    match (opt_l, opt_r) {
        (Some(l), Some(r)) => {
            // write to first 4 places
            unsafe { copy_from_slice_unchecked(&l.to_ne_bytes(), &mut s[..8]) }
            // write to second chunk of 4 places
            unsafe { copy_from_slice_unchecked(&r.to_ne_bytes(), &mut s[8..16]) }
            // leave last byte as is
        }
        (Some(l), None) => {
            unsafe { copy_from_slice_unchecked(&l.to_ne_bytes(), &mut s[..8]) }
            // set right null bit
            s[16] = 1;
        }
        (None, Some(r)) => {
            unsafe { copy_from_slice_unchecked(&r.to_ne_bytes(), &mut s[8..16]) }
            // set left null bit
            s[16] = 1 << 1;
        }
        (None, None) => {
            // set two null bits
            s[16] = 3;
        }
    }
    val
}

/// Used to tightly one 32 bit and a 64 bit valued type and null information
/// Only the bit values matter, not the meaning of the bits
#[inline]
fn pack_u32_u64_tuples(opt_l: Option<u32>, opt_r: Option<u64>) -> [u8; 13] {
    // 8 bytes for first value
    // 8 bytes for second value
    // last bytes' bits are used to indicate missing values
    let mut val = [0u8; 13];
    let s = &mut val;
    match (opt_l, opt_r) {
        (Some(l), Some(r)) => {
            // write to first 4 places
            unsafe { copy_from_slice_unchecked(&l.to_ne_bytes(), &mut s[..4]) }
            // write to second chunk of 4 places
            unsafe { copy_from_slice_unchecked(&r.to_ne_bytes(), &mut s[4..12]) }
            // leave last byte as is
        }
        (Some(l), None) => {
            unsafe { copy_from_slice_unchecked(&l.to_ne_bytes(), &mut s[..4]) }
            // set right null bit
            s[12] = 1;
        }
        (None, Some(r)) => {
            unsafe { copy_from_slice_unchecked(&r.to_ne_bytes(), &mut s[4..12]) }
            // set left null bit
            s[12] = 1 << 1;
        }
        (None, None) => {
            // set two null bits
            s[12] = 3;
        }
    }
    val
}

impl DataFrame {
    pub fn groupby_with_series(&self, by: Vec<Series>, multithreaded: bool) -> Result<GroupBy> {
        macro_rules! finish_packed_bit_path {
            ($ca0:expr, $ca1:expr, $pack_fn:expr) => {{
                let n_partitions = set_partition_size();

                // we split so that we can prepare the data over multiple threads.
                // pack the bit values together and add a final byte that will be 0
                // when there are no null values.
                // otherwise we use two bits of this byte to represent null values.
                let split_0 = split_ca(&$ca0, n_partitions).unwrap();
                let split_1 = split_ca(&$ca1, n_partitions).unwrap();
                let keys = POOL.install(|| {
                    split_0
                        .into_par_iter()
                        .zip(split_1.into_par_iter())
                        .map(|(ca0, ca1)| {
                            ca0.into_iter()
                                .zip(ca1.into_iter())
                                .map(|(l, r)| $pack_fn(l, r))
                                .trust_my_length(ca0.len())
                                .collect_trusted::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                });

                return Ok(GroupBy::new(
                    self,
                    by,
                    groupby_threaded_num(keys, 0, n_partitions as u64),
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
        let keys_df = DataFrame::new(
            by.iter()
                .map(|s| match s.dtype() {
                    Categorical | Int8 | UInt8 | Int16 | UInt16 => {
                        s.cast(&DataType::UInt32).unwrap()
                    }
                    Float32 => s.bit_repr_small().into_series(),
                    // otherwise we use the vec hash for float
                    Float64 => s.bit_repr_large().into_series(),
                    _ => {
                        // is date like
                        if !s.is_numeric() && s.is_numeric_physical() {
                            s.to_physical_repr()
                        } else {
                            s.clone()
                        }
                    }
                })
                .collect(),
        )?;

        let groups = match by.len() {
            1 => {
                let series = &by[0];
                series.group_tuples(multithreaded)
            }
            _ => {
                // multiple keys is always multi-threaded
                // reduce code paths
                let s0 = &keys_df.get_columns()[0];
                let s1 = &keys_df.get_columns()[1];

                // fast path for numeric data
                // uses the bit values to tightly pack those into arrays.
                if by.len() == 2 && s0.is_numeric() && s1.is_numeric() {
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
                }

                let n_partitions = set_partition_size();
                groupby_threaded_multiple_keys_flat(keys_df, n_partitions)
            }
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
    ///     df.groupby("column_name")?
    ///     .select("agg_column_name")
    ///     .sum()
    /// }
    /// ```
    pub fn groupby<'g, J, S: Selection<'g, J>>(&self, by: S) -> Result<GroupBy> {
        let selected_keys = self.select_series(by)?;
        self.groupby_with_series(selected_keys, true)
    }

    /// Group DataFrame using a Series column.
    /// The groups are ordered by their smallest row index.
    pub fn groupby_stable<'g, J, S: Selection<'g, J>>(&self, by: S) -> Result<GroupBy> {
        let mut gb = self.groupby(by)?;
        gb.groups.sort_unstable_by_key(|t| t.0);
        Ok(gb)
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
/// let s1 = Series::new("temp", [20, 10, 7, 9, 1].as_ref());
/// // create rain series
/// let s2 = Series::new("rain", [0.2, 0.1, 0.3, 0.1, 0.01].as_ref());
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
pub struct GroupBy<'df, 'selection_str> {
    df: &'df DataFrame,
    pub(crate) selected_keys: Vec<Series>,
    // [first idx, [other idx]]
    pub(crate) groups: GroupTuples,
    // columns selected for aggregation
    pub(crate) selected_agg: Option<Vec<&'selection_str str>>,
}

impl<'df, 'selection_str> GroupBy<'df, 'selection_str> {
    pub fn new(
        df: &'df DataFrame,
        by: Vec<Series>,
        groups: GroupTuples,
        selected_agg: Option<Vec<&'selection_str str>>,
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
    pub fn select<S, J>(mut self, selection: S) -> Self
    where
        S: Selection<'selection_str, J>,
    {
        self.selected_agg = Some(selection.to_selection_vec());
        self
    }

    /// Get the internal representation of the GroupBy operation.
    /// The Vec returned contains:
    ///     (first_idx, Vec<indexes>)
    ///     Where second value in the tuple is a vector with all matching indexes.
    pub fn get_groups(&self) -> &GroupTuples {
        &self.groups
    }

    /// Get the internal representation of the GroupBy operation.
    /// The Vec returned contains:
    ///     (first_idx, Vec<indexes>)
    ///     Where second value in the tuple is a vector with all matching indexes.
    pub fn get_groups_mut(&mut self) -> &mut GroupTuples {
        &mut self.groups
    }

    pub fn keys(&self) -> Vec<Series> {
        POOL.install(|| {
            self.selected_keys
                .par_iter()
                .map(|s| {
                    // Safety
                    // groupby indexes are in bound.
                    unsafe {
                        s.take_iter_unchecked(&mut self.groups.iter().map(|(idx, _)| *idx as usize))
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
    ///     df.groupby("date")?.select(&["temp", "rain"]).mean()
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
    ///     df.groupby("date")?.select("temp").sum()
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
    ///     df.groupby("date")?.select("temp").min()
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
    ///     df.groupby("date")?.select("temp").max()
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
    ///     df.groupby("date")?.select("temp").first()
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
    ///     df.groupby("date")?.select("temp").last()
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
    ///     df.groupby("date")?.select("temp").n_unique()
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
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby("date")?.select("temp").quantile(0.2)
    /// }
    /// ```
    pub fn quantile(&self, quantile: f64) -> Result<DataFrame> {
        if !(0.0..=1.0).contains(&quantile) {
            return Err(PolarsError::ComputeError(
                "quantile should be within 0.0 and 1.0".into(),
            ));
        }
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = fmt_groupby_column(agg_col.name(), GroupByMethod::Quantile(quantile));
            let opt_agg = agg_col.agg_quantile(&self.groups, quantile);
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
    ///     df.groupby("date")?.select("temp").median()
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
    ///     df.groupby("date")?.select("temp").count()
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
            let mut builder =
                PrimitiveChunkedBuilder::<UInt32Type>::new(&new_name, self.groups.len());
            for (_first, idx) in &self.groups {
                builder.append_value(idx.len() as u32);
            }
            let ca = builder.finish();
            cols.push(ca.into_series())
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
    ///     df.groupby("date")?.groups()
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

        let mut column: ListChunked = self
            .groups
            .iter()
            .map(|(_first, idx)| {
                let ca: NoNull<UInt32Chunked> = idx.iter().map(|&v| v as u32).collect();
                ca.into_inner().into_series()
            })
            .collect();
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
    ///      df.groupby("date")?.agg(&[("temp", &["n_unique", "sum", "min"])])
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
                            let mut builder = PrimitiveChunkedBuilder::<UInt32Type>::new(
                                &new_name,
                                self.groups.len(),
                            );
                            for (_first, idx) in &self.groups {
                                builder.append_value(idx.len() as u32);
                            }
                            let ca = builder.finish();
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
    ///     df.groupby("date")?.select("temp").agg_list()
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

    /// Apply a closure over the groups as a new DataFrame.
    pub fn apply<F>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(DataFrame) -> Result<DataFrame> + Send + Sync,
    {
        let df = if let Some(agg) = &self.selected_agg {
            if agg.is_empty() {
                self.df.clone()
            } else {
                let mut new_cols = Vec::with_capacity(self.selected_keys.len() + agg.len());
                new_cols.extend_from_slice(&self.selected_keys);
                let cols = self.df.select_series(agg)?;
                new_cols.extend(cols.into_iter());
                DataFrame::new_no_checks(new_cols)
            }
        } else {
            self.df.clone()
        };

        let dfs = self
            .get_groups()
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
}

#[derive(Copy, Clone)]
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
    Quantile(f64),
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
        Quantile(quantile) => format!("{}_quantile_{:.2}", name, quantile),
        Std => format!("{}_agg_std", name),
        Var => format!("{}_agg_var", name),
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::frame::groupby::{groupby, groupby_threaded_num};
    use crate::prelude::*;
    use crate::utils::split_ca;
    use num::traits::FloatConst;

    #[test]
    #[cfg(feature = "dtype-date")]
    #[cfg_attr(miri, ignore)]
    fn test_group_by() {
        let s0 = DateChunked::parse_from_str_slice(
            "date",
            &[
                "2020-08-21",
                "2020-08-21",
                "2020-08-22",
                "2020-08-23",
                "2020-08-22",
            ],
            "%Y-%m-%d",
        )
        .into_series();
        let s1 = Series::new("temp", [20, 10, 7, 9, 1].as_ref());
        let s2 = Series::new("rain", [0.2, 0.1, 0.3, 0.1, 0.01].as_ref());
        let df = DataFrame::new(vec![s0, s1, s2]).unwrap();
        println!("{:?}", df);

        println!(
            "{:?}",
            df.groupby("date").unwrap().select("temp").count().unwrap()
        );
        // Select multiple
        println!(
            "{:?}",
            df.groupby("date")
                .unwrap()
                .select(&["temp", "rain"])
                .mean()
                .unwrap()
        );
        // Group by multiple
        println!(
            "multiple keys {:?}",
            df.groupby(&["date", "temp"])
                .unwrap()
                .select("rain")
                .mean()
                .unwrap()
        );
        println!(
            "{:?}",
            df.groupby("date").unwrap().select("temp").sum().unwrap()
        );
        println!(
            "{:?}",
            df.groupby("date").unwrap().select("temp").min().unwrap()
        );
        println!(
            "{:?}",
            df.groupby("date").unwrap().select("temp").max().unwrap()
        );
        println!(
            "{:?}",
            df.groupby("date")
                .unwrap()
                .select("temp")
                .agg_list()
                .unwrap()
        );
        println!(
            "{:?}",
            df.groupby("date").unwrap().select("temp").first().unwrap()
        );
        println!(
            "{:?}",
            df.groupby("date").unwrap().select("temp").last().unwrap()
        );
        println!(
            "{:?}",
            df.groupby("date")
                .unwrap()
                .select("temp")
                .n_unique()
                .unwrap()
        );
        println!(
            "{:?}",
            df.groupby("date")
                .unwrap()
                .select("temp")
                .quantile(0.2)
                .unwrap()
        );
        println!(
            "{:?}",
            df.groupby("date").unwrap().select("temp").median().unwrap()
        );
        // implicit select all and only aggregate on methods that support that aggregation
        let gb = df.groupby("date").unwrap().n_unique().unwrap();
        println!("{:?}", df.groupby("date").unwrap().n_unique().unwrap());
        // check the group by column is filtered out.
        assert_eq!(gb.width(), 2);
        println!(
            "{:?}",
            df.groupby("date")
                .unwrap()
                .agg(&[("temp", &["n_unique", "sum", "min"])])
                .unwrap()
        );
        println!("{:?}", df.groupby("date").unwrap().groups().unwrap());
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
        println!("{:?}", df);

        let adf = df
            .groupby(&[
                "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12",
            ])
            .unwrap()
            .select("N")
            .sum()
            .unwrap();

        println!("{:?}", adf);

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
        println!("{:?}", df);

        // Compute the aggregated DataFrame by the 13 columns defined in `series_names`.
        let adf = df
            .groupby(&series_names)
            .unwrap()
            .select("N")
            .sum()
            .unwrap();
        println!("{:?}", adf);

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
        let res = df.groupby("flt").unwrap().sum().unwrap();
        let res = res.sort("flt", false).unwrap();
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

        df.apply("foo", |s| s.cast(&DataType::Categorical).unwrap())
            .unwrap();

        // check multiple keys and categorical
        let res = df
            .groupby_stable(&["foo", "ham"])
            .unwrap()
            .select("bar")
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

        let out = df.groupby("a").unwrap().apply(Ok).unwrap();
        assert!(out.sort("b", false).unwrap().frame_equal(&df));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_groupby_threaded() {
        for slice in &[
            vec![1, 2, 3, 4, 4, 4, 2, 1],
            vec![1, 2, 3, 4, 4, 4, 2, 1, 1],
            vec![1, 2, 3, 4, 4, 4],
        ] {
            let ca = UInt32Chunked::new_from_slice("", slice);
            let split = split_ca(&ca, 4).unwrap();

            let a = groupby(ca.into_iter()).into_iter().sorted().collect_vec();

            let keys = split.iter().map(|ca| ca.cont_slice().unwrap()).collect();
            let b = groupby_threaded_num(keys, 0, split.len() as u64)
                .into_iter()
                .sorted()
                .collect_vec();

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
        let out = df.groupby_stable("a")?.mean()?;

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

        let out = df.groupby("g")?.select("int").var()?;
        assert_eq!(
            out.column("int_agg_var")?.f64()?.sort(false).get(1),
            Some(0.5)
        );
        let out = df.groupby("g")?.select("int").std()?;
        let val = out
            .column("int_agg_std")?
            .f64()?
            .sort(false)
            .get(1)
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

        df.may_apply("g", |s| s.cast(&DataType::Categorical))?;

        let out = df.groupby("g")?.sum()?;
        dbg!(out);
        Ok(())
    }
}
