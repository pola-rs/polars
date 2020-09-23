use super::hash_join::prepare_hashed_relation;
use crate::chunked_array::builder::PrimitiveChunkedBuilder;
use crate::frame::select::Selection;
use crate::prelude::*;
use crate::utils::Xob;
use arrow::array::{PrimitiveBuilder, StringBuilder};
use enum_dispatch::enum_dispatch;
use fnv::FnvBuildHasher;
use itertools::Itertools;
use num::{Num, NumCast, ToPrimitive, Zero};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::{
    fmt::{Debug, Formatter},
    ops::Add,
};

fn groupby<T>(a: impl Iterator<Item = T>) -> Vec<(usize, Vec<usize>)>
where
    T: Hash + Eq + Copy,
{
    let hash_tbl = prepare_hashed_relation(a);

    hash_tbl
        .into_iter()
        .map(|(_, indexes)| {
            let first = unsafe { *indexes.get_unchecked(0) };
            (first, indexes)
        })
        .collect()
}

#[enum_dispatch(Series)]
trait IntoGroupTuples {
    fn group_tuples(&self) -> Vec<(usize, Vec<usize>)> {
        unimplemented!()
    }
}

impl<T> IntoGroupTuples for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Eq + Hash,
{
    fn group_tuples(&self) -> Vec<(usize, Vec<usize>)> {
        if let Ok(slice) = self.cont_slice() {
            groupby(slice.iter())
        } else {
            groupby(self.into_iter())
        }
    }
}
impl IntoGroupTuples for BooleanChunked {
    fn group_tuples(&self) -> Vec<(usize, Vec<usize>)> {
        if self.is_optimal_aligned() {
            groupby(self.into_no_null_iter())
        } else {
            groupby(self.into_iter())
        }
    }
}

impl IntoGroupTuples for Utf8Chunked {
    fn group_tuples(&self) -> Vec<(usize, Vec<usize>)> {
        if self.is_optimal_aligned() {
            groupby(self.into_no_null_iter())
        } else {
            groupby(self.into_iter())
        }
    }
}

impl IntoGroupTuples for Float64Chunked {}
impl IntoGroupTuples for Float32Chunked {}
impl IntoGroupTuples for LargeListChunked {}

/// Utility enum used for grouping on multiple columns
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
enum Groupable<'a> {
    Boolean(bool),
    Utf8(&'a str),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
}

impl<'a> Debug for Groupable<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use Groupable::*;
        match self {
            Boolean(v) => write!(f, "{}", v),
            Utf8(v) => write!(f, "{}", v),
            UInt8(v) => write!(f, "{}", v),
            UInt16(v) => write!(f, "{}", v),
            UInt32(v) => write!(f, "{}", v),
            UInt64(v) => write!(f, "{}", v),
            Int8(v) => write!(f, "{}", v),
            Int16(v) => write!(f, "{}", v),
            Int32(v) => write!(f, "{}", v),
            Int64(v) => write!(f, "{}", v),
        }
    }
}

impl Series {
    fn as_groupable_iter<'a>(&'a self) -> Result<Box<dyn Iterator<Item = Option<Groupable>> + 'a>> {
        macro_rules! as_groupable_iter {
            ($ca:expr, $variant:ident ) => {{
                let bx = Box::new(
                    $ca.into_iter()
                        .map(|opt_b| opt_b.map(|b| Groupable::$variant(b))),
                );
                Ok(bx)
            }};
        }
        match self {
            Series::Bool(ca) => as_groupable_iter!(ca, Boolean),
            Series::UInt8(ca) => as_groupable_iter!(ca, UInt8),
            Series::UInt16(ca) => as_groupable_iter!(ca, UInt16),
            Series::UInt32(ca) => as_groupable_iter!(ca, UInt32),
            Series::UInt64(ca) => as_groupable_iter!(ca, UInt64),
            Series::Int8(ca) => as_groupable_iter!(ca, Int8),
            Series::Int16(ca) => as_groupable_iter!(ca, Int16),
            Series::Int32(ca) => as_groupable_iter!(ca, Int32),
            Series::Int64(ca) => as_groupable_iter!(ca, Int64),
            Series::Date32(ca) => as_groupable_iter!(ca, Int32),
            Series::Date64(ca) => as_groupable_iter!(ca, Int64),
            Series::TimestampSecond(ca) => as_groupable_iter!(ca, Int64),
            Series::TimestampMillisecond(ca) => as_groupable_iter!(ca, Int64),
            Series::TimestampNanosecond(ca) => as_groupable_iter!(ca, Int64),
            Series::TimestampMicrosecond(ca) => as_groupable_iter!(ca, Int64),
            Series::Time32Second(ca) => as_groupable_iter!(ca, Int32),
            Series::Time32Millisecond(ca) => as_groupable_iter!(ca, Int32),
            Series::Time64Nanosecond(ca) => as_groupable_iter!(ca, Int64),
            Series::Time64Microsecond(ca) => as_groupable_iter!(ca, Int64),
            Series::DurationNanosecond(ca) => as_groupable_iter!(ca, Int64),
            Series::DurationMicrosecond(ca) => as_groupable_iter!(ca, Int64),
            Series::DurationMillisecond(ca) => as_groupable_iter!(ca, Int64),
            Series::DurationSecond(ca) => as_groupable_iter!(ca, Int64),
            Series::IntervalDayTime(ca) => as_groupable_iter!(ca, Int64),
            Series::IntervalYearMonth(ca) => as_groupable_iter!(ca, Int32),
            Series::Utf8(ca) => as_groupable_iter!(ca, Utf8),
            _ => Err(PolarsError::Other("Column is not groupable".into())),
        }
    }
}

impl DataFrame {
    /// Group DataFrame using a Series column.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn groupby_sum(df: &DataFrame) -> Result<DataFrame> {
    ///     df.groupby("column_name")?
    ///     .select("agg_column_name")
    ///     .sum()
    /// }
    /// ```
    pub fn groupby<'g, J, S: Selection<'g, J>>(&self, by: S) -> Result<GroupBy> {
        let selected_keys = self.select_series(by)?;

        let groups = match selected_keys.len() {
            1 => selected_keys[0].group_tuples(),
            2 => {
                let iter = selected_keys[0]
                    .as_groupable_iter()?
                    .zip(selected_keys[1].as_groupable_iter()?);
                groupby(iter)
            }
            3 => {
                let iter = selected_keys[0]
                    .as_groupable_iter()?
                    .zip(selected_keys[1].as_groupable_iter()?)
                    .zip(selected_keys[2].as_groupable_iter()?);
                groupby(iter)
            }
            4 => {
                let iter = selected_keys[0]
                    .as_groupable_iter()?
                    .zip(selected_keys[1].as_groupable_iter()?)
                    .zip(selected_keys[2].as_groupable_iter()?)
                    .zip(selected_keys[3].as_groupable_iter()?);
                groupby(iter)
            }
            5 => {
                let iter = selected_keys[0]
                    .as_groupable_iter()?
                    .zip(selected_keys[1].as_groupable_iter()?)
                    .zip(selected_keys[2].as_groupable_iter()?)
                    .zip(selected_keys[3].as_groupable_iter()?)
                    .zip(selected_keys[4].as_groupable_iter()?);
                groupby(iter)
            }
            _ => {
                return Err(PolarsError::Other(
                    "more than 5 combined keys are currently not supported".into(),
                ));
            }
        };

        Ok(GroupBy {
            df: self,
            selected_keys,
            groups,
            selected_agg: None,
        })
    }
}

/// Returned by a groupby operation on a DataFrame. This struct supports
/// several aggregations.
///
/// Until described otherwise, the examples in this struct are performed on the following DataFrame:
///
/// ```rust
/// use polars::prelude::*;
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
/// let s0 = Date32Chunked::parse_from_str_slice("date", dates, fmt)
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
/// | date32     | i32  | f64  |
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
    selected_keys: Vec<Series>,
    // [first idx, [other idx]]
    groups: Vec<(usize, Vec<usize>)>,
    // columns selected for aggregation
    selected_agg: Option<Vec<&'selection_str str>>,
}

#[enum_dispatch(Series)]
trait NumericAggSync {
    fn agg_mean(&self, _groups: &Vec<(usize, Vec<usize>)>) -> Option<Series> {
        None
    }
    fn agg_min(&self, _groups: &Vec<(usize, Vec<usize>)>) -> Option<Series> {
        None
    }
    fn agg_max(&self, _groups: &Vec<(usize, Vec<usize>)>) -> Option<Series> {
        None
    }
    fn agg_sum(&self, _groups: &Vec<(usize, Vec<usize>)>) -> Option<Series> {
        None
    }
}

impl NumericAggSync for BooleanChunked {}
impl NumericAggSync for Utf8Chunked {}
impl NumericAggSync for LargeListChunked {}

impl<T> NumericAggSync for ChunkedArray<T>
where
    T: PolarsNumericType + Sync,
    T::Native: std::ops::Add<Output = T::Native> + Num + NumCast,
{
    fn agg_mean(&self, groups: &Vec<(usize, Vec<usize>)>) -> Option<Series> {
        Some(Series::Float64(
            groups
                .par_iter()
                .map(|(_first, idx)| {
                    // Fast path
                    if let Ok(slice) = self.cont_slice() {
                        let mut sum = 0.;
                        for i in idx {
                            sum = sum + slice[*i].to_f64().unwrap()
                        }
                        Some(sum / idx.len() as f64)
                    } else {
                        let take = unsafe {
                            self.take_unchecked(idx.into_iter().copied(), Some(self.len()))
                        };
                        let opt_sum: Option<T::Native> = take.sum();
                        opt_sum.map(|sum| sum.to_f64().unwrap() / idx.len() as f64)
                    }
                })
                .collect(),
        ))
    }

    fn agg_min(&self, groups: &Vec<(usize, Vec<usize>)>) -> Option<Series> {
        Some(
            groups
                .par_iter()
                .map(|(_first, idx)| {
                    if let Ok(slice) = self.cont_slice() {
                        let mut min = None;
                        for i in idx {
                            let v = slice[*i];

                            min = match min {
                                Some(min) => {
                                    if min < v {
                                        Some(min)
                                    } else {
                                        Some(v)
                                    }
                                }
                                None => Some(v),
                            };
                        }
                        min
                    } else {
                        let take = unsafe {
                            self.take_unchecked(idx.into_iter().copied(), Some(self.len()))
                        };
                        take.min()
                    }
                })
                .collect::<ChunkedArray<T>>()
                .into_series(),
        )
    }

    fn agg_max(&self, groups: &Vec<(usize, Vec<usize>)>) -> Option<Series> {
        Some(
            groups
                .par_iter()
                .map(|(_first, idx)| {
                    if let Ok(slice) = self.cont_slice() {
                        let mut max = None;
                        for i in idx {
                            let v = slice[*i];

                            max = match max {
                                Some(max) => {
                                    if max > v {
                                        Some(max)
                                    } else {
                                        Some(v)
                                    }
                                }
                                None => Some(v),
                            };
                        }
                        max
                    } else {
                        let take = unsafe {
                            self.take_unchecked(idx.into_iter().copied(), Some(self.len()))
                        };
                        take.max()
                    }
                })
                .collect::<ChunkedArray<T>>()
                .into_series(),
        )
    }

    fn agg_sum(&self, groups: &Vec<(usize, Vec<usize>)>) -> Option<Series> {
        Some(
            groups
                .par_iter()
                .map(|(_first, idx)| {
                    if let Ok(slice) = self.cont_slice() {
                        let mut sum = Zero::zero();
                        for i in idx {
                            sum = sum + slice[*i]
                        }
                        Some(sum)
                    } else {
                        let take = unsafe {
                            self.take_unchecked(idx.into_iter().copied(), Some(self.len()))
                        };
                        take.sum()
                    }
                })
                .collect::<ChunkedArray<T>>()
                .into_series(),
        )
    }
}

#[enum_dispatch(Series)]
trait AggFirst {
    fn agg_first(&self, _groups: &Vec<(usize, Vec<usize>)>) -> Series;
}

macro_rules! impl_agg_first {
    ($self:ident, $groups:ident, $ca_type:ty) => {{
        $groups
            .iter()
            .map(|(first, _idx)| $self.get(*first))
            .collect::<$ca_type>()
            .into_series()
    }};
}

impl<T> AggFirst for ChunkedArray<T>
where
    T: ArrowPrimitiveType + Send,
{
    fn agg_first(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
        impl_agg_first!(self, groups, ChunkedArray<T>)
    }
}

impl AggFirst for Utf8Chunked {
    fn agg_first(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
        impl_agg_first!(self, groups, Utf8Chunked)
    }
}

impl AggFirst for LargeListChunked {
    fn agg_first(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
        impl_agg_first!(self, groups, LargeListChunked)
    }
}

#[enum_dispatch(Series)]
trait AggLast {
    fn agg_last(&self, _groups: &Vec<(usize, Vec<usize>)>) -> Series;
}

macro_rules! impl_agg_last {
    ($self:ident, $groups:ident, $ca_type:ty) => {{
        $groups
            .iter()
            .map(|(_first, idx)| $self.get(idx[idx.len() - 1]))
            .collect::<$ca_type>()
            .into_series()
    }};
}

impl<T> AggLast for ChunkedArray<T>
where
    T: ArrowPrimitiveType + Send,
{
    fn agg_last(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
        impl_agg_last!(self, groups, ChunkedArray<T>)
    }
}

impl AggLast for Utf8Chunked {
    fn agg_last(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
        impl_agg_last!(self, groups, Utf8Chunked)
    }
}

impl AggLast for LargeListChunked {
    fn agg_last(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
        impl_agg_last!(self, groups, LargeListChunked)
    }
}

#[enum_dispatch(Series)]
trait AggNUnique {
    fn agg_n_unique(&self, _groups: &Vec<(usize, Vec<usize>)>) -> Option<UInt32Chunked> {
        None
    }
}

macro_rules! impl_agg_n_unique {
    ($self:ident, $groups:ident, $ca_type:ty) => {{
        $groups
            .into_par_iter()
            .map(|(_first, idx)| {
                if $self.null_count() == 0 {
                    let mut set = HashSet::with_hasher(FnvBuildHasher::default());
                    for i in idx {
                        let v = unsafe { $self.get_unchecked(*i) };
                        set.insert(v);
                    }
                    set.len() as u32
                } else {
                    let mut set = HashSet::with_hasher(FnvBuildHasher::default());
                    for i in idx {
                        let opt_v = $self.get(*i);
                        set.insert(opt_v);
                    }
                    set.len() as u32
                }
            })
            .collect::<$ca_type>()
            .into_inner()
    }};
}

impl<T> AggNUnique for ChunkedArray<T>
where
    T: PolarsIntegerType + Sync,
    T::Native: Hash + Eq,
{
    fn agg_n_unique(&self, groups: &Vec<(usize, Vec<usize>)>) -> Option<UInt32Chunked> {
        Some(impl_agg_n_unique!(self, groups, Xob<UInt32Chunked>))
    }
}

impl AggNUnique for Float32Chunked {}
impl AggNUnique for Float64Chunked {}
impl AggNUnique for LargeListChunked {}

// TODO: could be faster as it can only be null, true, or false
impl AggNUnique for BooleanChunked {
    fn agg_n_unique(&self, groups: &Vec<(usize, Vec<usize>)>) -> Option<UInt32Chunked> {
        Some(impl_agg_n_unique!(self, groups, Xob<UInt32Chunked>))
    }
}

impl AggNUnique for Utf8Chunked {
    fn agg_n_unique(&self, groups: &Vec<(usize, Vec<usize>)>) -> Option<UInt32Chunked> {
        Some(impl_agg_n_unique!(self, groups, Xob<UInt32Chunked>))
    }
}

#[enum_dispatch(Series)]
trait AggQuantile {
    fn agg_quantile(&self, _groups: &Vec<(usize, Vec<usize>)>, _quantile: f64) -> Option<Series> {
        None
    }
}

impl<T> AggQuantile for ChunkedArray<T>
where
    T: PolarsNumericType + Sync,
    T::Native: PartialEq,
{
    fn agg_quantile(&self, groups: &Vec<(usize, Vec<usize>)>, quantile: f64) -> Option<Series> {
        Some(
            groups
                .into_par_iter()
                .map(|(_first, idx)| {
                    let group_vals =
                        unsafe { self.take_unchecked(idx.iter().copied(), Some(idx.len())) };
                    let sorted_idx = group_vals.argsort(false);
                    let quant_idx = (quantile * (sorted_idx.len() - 1) as f64) as usize;
                    let value_idx = sorted_idx[quant_idx];
                    group_vals.get(value_idx)
                })
                .collect::<ChunkedArray<T>>()
                .into_series(),
        )
    }
}

impl AggQuantile for Utf8Chunked {}
impl AggQuantile for BooleanChunked {}
impl AggQuantile for LargeListChunked {}

impl<'df, 'selection_str> GroupBy<'df, 'selection_str> {
    /// Select the column by which the determine the groups.
    /// You can select a single column or a slice of columns.
    pub fn select<S, J>(mut self, selection: S) -> Self
    where
        S: Selection<'selection_str, J>,
    {
        self.selected_agg = Some(selection.to_selection_vec());
        self
    }

    fn keys(&self) -> Vec<Series> {
        // Keys will later be appended with the aggregation columns, so we already allocate extra space
        let size;
        if let Some(sel) = &self.selected_agg {
            size = sel.len() + self.selected_keys.len();
        } else {
            size = self.selected_keys.len();
        }
        let mut keys = Vec::with_capacity(size);
        unsafe {
            self.selected_keys.iter().for_each(|s| {
                let key = s.take_iter_unchecked(
                    self.groups.iter().map(|(idx, _)| *idx),
                    Some(self.groups.len()),
                );
                keys.push(key)
            });
        }
        keys
    }

    fn prepare_agg(&self) -> Result<(Vec<Series>, Vec<Series>)> {
        let selection = match &self.selected_agg {
            Some(selection) => selection.clone(),
            None => {
                let by: Vec<_> = self.selected_keys.iter().map(|s| s.name()).collect();
                self.df
                    .columns()
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
    /// # use polars::prelude::*;
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
    /// | date32     | f64       | f64       |
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
            let new_name = format!["{}_mean", agg_col.name()];
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
    /// # Example                agg.rename(&new_name);
    ///
    /// ```rust
    /// # use polars::prelude::*;
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
    /// | date32     | i32      |
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
            let new_name = format!["{}_sum", agg_col.name()];
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
    /// # use polars::prelude::*;
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
    /// | date32     | i32      |
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
            let new_name = format!["{}_min", agg_col.name()];
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
    /// # use polars::prelude::*;
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
    /// | date32     | i32      |
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
            let new_name = format!["{}_max", agg_col.name()];
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
    /// # use polars::prelude::*;
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
    /// | date32     | i32        |
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
            let new_name = format!["{}_first", agg_col.name()];
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
    /// # use polars::prelude::*;
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
    /// | date32     | i32        |
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
            let new_name = format!["{}_last", agg_col.name()];
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
    /// # use polars::prelude::*;
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
    /// | date32     | u32           |
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
            let new_name = format!["{}_n_unique", agg_col.name()];
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
    /// # use polars::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby("date")?.select("temp").quantile(0.2)
    /// }
    /// ```
    pub fn quantile(&self, quantile: f64) -> Result<DataFrame> {
        if quantile < 0.0 || quantile > 1.0 {
            return Err(PolarsError::Other(
                "quantile should be within 0.0 and 1.0".into(),
            ));
        }
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = format!["{}_quantile_{:.2}", agg_col.name(), quantile];
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
    /// # use polars::prelude::*;
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby("date")?.select("temp").median()
    /// }
    /// ```
    pub fn median(&self) -> Result<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = format!["{}_median", agg_col.name()];
            let opt_agg = agg_col.agg_quantile(&self.groups, 0.5);
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
    /// # use polars::prelude::*;
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
    /// | date32     | u32        |
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
            let new_name = format!["{}_count", agg_col.name()];
            let mut builder = PrimitiveChunkedBuilder::new(&new_name, self.groups.len());
            for (_first, idx) in &self.groups {
                builder.append_value(idx.len() as u32);
            }
            let ca = builder.finish();
            let agg = Series::UInt32(ca);
            cols.push(agg);
        }
        DataFrame::new(cols)
    }

    /// Aggregate the groups of the groupby operation into lists.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars::prelude::*;
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
    /// | date32     | list [i32]             |
    /// +============+========================+
    /// | 2020-08-23 | "[Some(9)]"            |
    /// +------------+------------------------+
    /// | 2020-08-22 | "[Some(7), Some(1)]"   |
    /// +------------+------------------------+
    /// | 2020-08-21 | "[Some(20), Some(10)]" |
    /// +------------+------------------------+
    /// ```
    pub fn agg_list(&self) -> Result<DataFrame> {
        macro_rules! impl_gb {
            ($type:ty, $agg_col:expr) => {{
                let values_builder = PrimitiveBuilder::<$type>::new(self.groups.len());
                let mut builder =
                    LargeListPrimitiveChunkedBuilder::new("", values_builder, self.groups.len());
                for (_first, idx) in &self.groups {
                    let s = unsafe {
                        $agg_col.take_iter_unchecked(idx.into_iter().copied(), Some(idx.len()))
                    };
                    builder.append_opt_series(&Some(s))
                }
                builder.finish().into_series()
            }};
        }

        macro_rules! impl_gb_utf8 {
            ($agg_col:expr) => {{
                let values_builder = StringBuilder::new(self.groups.len());
                let mut builder =
                    LargeListUtf8ChunkedBuilder::new("", values_builder, self.groups.len());
                for (_first, idx) in &self.groups {
                    let s = unsafe {
                        $agg_col.take_iter_unchecked(idx.into_iter().copied(), Some(idx.len()))
                    };
                    builder.append_series(&s)
                }
                builder.finish().into_series()
            }};
        }

        let (mut cols, agg_cols) = self.prepare_agg()?;
        for agg_col in agg_cols {
            let new_name = format!["{}_agg_list", agg_col.name()];
            let mut agg =
                match_arrow_data_type_apply_macro!(agg_col.dtype(), impl_gb, impl_gb_utf8, agg_col);
            agg.rename(&new_name);
            cols.push(agg);
        }
        DataFrame::new(cols)
    }

    /// Pivot a column of the current `DataFrame` and perform one of the following aggregations:
    /// * first
    /// * sum
    /// * min
    /// * max
    /// * mean
    /// * median
    ///
    /// The pivot operation consists of a group by one, or multiple collumns (these will be the new
    /// y-axis), column that will be pivoted (this will be the new x-axis) and an aggregation.
    ///
    /// # Panics
    /// If the values column is not a numerical type, the code will panic.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars::prelude::*;
    /// let s0 = Series::new("foo", ["A", "A", "B", "B", "C"].as_ref());
    /// let s1 = Series::new("N", [1, 2, 2, 4, 2].as_ref());
    /// let s2 = Series::new("bar", ["k", "l", "m", "n", "o"].as_ref());
    /// // create a new DataFrame
    /// let df = DataFrame::new(vec![s0, s1, s2]).unwrap();
    ///
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///     df.groupby("foo")?
    ///     .pivot("bar", "N")
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
    pub fn pivot(
        &mut self,
        pivot_column: &'selection_str str,
        values_column: &'selection_str str,
    ) -> Pivot {
        // same as select method
        self.selected_agg = Some(vec![pivot_column, values_column]);

        let pivot = Pivot {
            gb: self,
            pivot_column,
            values_column,
        };
        pivot
    }
}

/// Intermediate structure when a `pivot` operation is applied.
/// See [the pivot method for more information.](../group_by/struct.GroupBy.html#method.pivot)
pub struct Pivot<'df, 'selection_str> {
    gb: &'df GroupBy<'df, 'selection_str>,
    pivot_column: &'selection_str str,
    values_column: &'selection_str str,
}

#[enum_dispatch(Series)]
trait ChunkPivot {
    fn pivot(
        &self,
        _pivot_series: &Series,
        _keys: Vec<Series>,
        _groups: &Vec<(usize, Vec<usize>)>,
        _agg_type: PivotAgg,
    ) -> Result<DataFrame> {
        unimplemented!()
    }
}

impl<T> ChunkPivot for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Copy + Num + NumCast,
{
    fn pivot(
        &self,
        pivot_series: &Series,
        keys: Vec<Series>,
        groups: &Vec<(usize, Vec<usize>)>,
        agg_type: PivotAgg,
    ) -> Result<DataFrame> {
        // TODO: save an allocation by creating a random access struct for the Groupable utility type.
        let pivot_vec: Vec<_> = pivot_series.as_groupable_iter()?.collect();

        let values_taker = self.take_rand();

        let new_column_map = |size| {
            // create a new hashmap that will be filled with new Vecs that later will be aggegrated
            let mut columns_agg_map =
                HashMap::with_capacity_and_hasher(size, FnvBuildHasher::default());
            for opt_column_name in &pivot_vec {
                if let Some(column_name) = opt_column_name {
                    columns_agg_map
                        .entry(column_name)
                        .or_insert_with(|| Vec::new());
                }
            }

            columns_agg_map
        };

        // create a hash map that will be filled with the results of the aggregation.
        // let mut columns_agg_map_main = new_column_map(groups.len());
        let mut columns_agg_map_main =
            HashMap::with_capacity_and_hasher(pivot_vec.len(), FnvBuildHasher::default());
        for opt_column_name in &pivot_vec {
            if let Some(column_name) = opt_column_name {
                columns_agg_map_main.entry(column_name).or_insert_with(|| {
                    PrimitiveChunkedBuilder::<T>::new(&format!("{:?}", column_name), groups.len())
                });
            }
        }

        // iterate over the groups that need to be aggregated
        // idxes are the indexes of the groups in the keys, pivot, and values columns
        for (_first, idx) in groups {
            // for every group do the aggregation by adding them to the vector belonging by that column
            // the columns are hashed with the pivot values
            let mut columns_agg_map_group = new_column_map(idx.len());
            for &i in idx {
                let opt_pivot_val = unsafe { pivot_vec.get_unchecked(i) };

                if let Some(pivot_val) = opt_pivot_val {
                    let values_val = values_taker.get(i);
                    columns_agg_map_group
                        .get_mut(&pivot_val)
                        .map(|v| v.push(values_val));
                }
            }

            // After the vectors are filled we really do the aggregation and add the result to the main
            // hash map, mapping pivot values as column to aggregate result.
            for (k, v) in &mut columns_agg_map_group {
                let main_builder = columns_agg_map_main.get_mut(k).unwrap();

                match v.len() {
                    0 => main_builder.append_null(),
                    // NOTE: now we take first, but this is the place where all aggregations happen
                    _ => match agg_type {
                        PivotAgg::First => pivot_agg_first(main_builder, v),
                        PivotAgg::Sum => pivot_agg_sum(main_builder, v),
                        PivotAgg::Min => pivot_agg_min(main_builder, v),
                        PivotAgg::Max => pivot_agg_max(main_builder, v),
                        PivotAgg::Mean => pivot_agg_mean(main_builder, v),
                        PivotAgg::Median => pivot_agg_median(main_builder, v),
                    },
                }
            }
        }
        // Finalize the pivot by creating a vec of all the columns and creating a DataFrame
        let mut cols = keys;
        cols.reserve_exact(columns_agg_map_main.len());

        for (_, builder) in columns_agg_map_main {
            let ca = builder.finish();
            cols.push(ca.into_series());
        }

        DataFrame::new(cols)
    }
}

impl ChunkPivot for BooleanChunked {}
impl ChunkPivot for Utf8Chunked {}
impl ChunkPivot for LargeListChunked {}

enum PivotAgg {
    First,
    Sum,
    Min,
    Max,
    Mean,
    Median,
}

fn pivot_agg_first<T>(builder: &mut PrimitiveChunkedBuilder<T>, v: &Vec<Option<T::Native>>)
where
    T: PolarsNumericType,
{
    builder.append_option(v[0]);
}

fn pivot_agg_median<T>(builder: &mut PrimitiveChunkedBuilder<T>, v: &mut Vec<Option<T::Native>>)
where
    T: PolarsNumericType,
    T::Native: PartialOrd,
{
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    builder.append_option(v[v.len() / 2]);
}

fn pivot_agg_sum<T>(builder: &mut PrimitiveChunkedBuilder<T>, v: &Vec<Option<T::Native>>)
where
    T: PolarsNumericType,
    T::Native: Num + Zero,
{
    builder.append_option(v.iter().copied().fold_options(Zero::zero(), Add::add));
}

fn pivot_agg_mean<T>(builder: &mut PrimitiveChunkedBuilder<T>, v: &Vec<Option<T::Native>>)
where
    T: PolarsNumericType,
    T::Native: Num + Zero + NumCast,
{
    builder.append_option(
        v.iter()
            .copied()
            .fold_options::<T::Native, T::Native, _>(Zero::zero(), Add::add)
            .map(|sum_val| sum_val / NumCast::from(v.len()).unwrap()),
    );
}

fn pivot_agg_min<T>(builder: &mut PrimitiveChunkedBuilder<T>, v: &Vec<Option<T::Native>>)
where
    T: PolarsNumericType,
{
    let mut min = None;

    for opt_val in v {
        if let Some(val) = opt_val {
            match min {
                None => min = Some(*val),
                Some(minimum) => {
                    if val < &minimum {
                        min = Some(*val)
                    }
                }
            }
        }
    }

    builder.append_option(min);
}

fn pivot_agg_max<T>(builder: &mut PrimitiveChunkedBuilder<T>, v: &Vec<Option<T::Native>>)
where
    T: PolarsNumericType,
{
    let mut max = None;

    for opt_val in v {
        if let Some(val) = opt_val {
            match max {
                None => max = Some(*val),
                Some(maximum) => {
                    if val > &maximum {
                        max = Some(*val)
                    }
                }
            }
        }
    }

    builder.append_option(max);
}

impl<'df, 'sel_str> Pivot<'df, 'sel_str> {
    /// Aggregate the pivot results by taking the first occurring value.
    pub fn first(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        values_series.pivot(
            pivot_series,
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::First,
        )
    }

    /// Aggregate the pivot results by taking the sum of all duplicates.
    pub fn sum(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        values_series.pivot(pivot_series, self.gb.keys(), &self.gb.groups, PivotAgg::Sum)
    }

    /// Aggregate the pivot results by taking the minimal value of all duplicates.
    pub fn min(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        values_series.pivot(pivot_series, self.gb.keys(), &self.gb.groups, PivotAgg::Min)
    }

    /// Aggregate the pivot results by taking the maximum value of all duplicates.
    pub fn max(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        values_series.pivot(pivot_series, self.gb.keys(), &self.gb.groups, PivotAgg::Max)
    }

    /// Aggregate the pivot results by taking the mean value of all duplicates.
    pub fn mean(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        values_series.pivot(
            pivot_series,
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::Mean,
        )
    }
    /// Aggregate the pivot results by taking the median value of all duplicates.
    pub fn median(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        values_series.pivot(
            pivot_series,
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::Median,
        )
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_group_by() {
        let s0 = Date32Chunked::parse_from_str_slice(
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
        assert_eq!(gb.width(), 2)
    }

    #[test]
    fn test_pivot() {
        let s0 = Series::new("foo", ["A", "A", "B", "B", "C"].as_ref());
        let s1 = Series::new("N", [1, 2, 2, 4, 2].as_ref());
        let s2 = Series::new("bar", ["k", "l", "m", "m", "l"].as_ref());
        let df = DataFrame::new(vec![s0, s1, s2]).unwrap();
        println!("{:?}", df);

        let pvt = df.groupby("foo").unwrap().pivot("bar", "N").sum().unwrap();
        assert_eq!(
            Vec::from(pvt.column("m").unwrap().i32().unwrap()),
            &[None, Some(6), None]
        );
        let pvt = df.groupby("foo").unwrap().pivot("bar", "N").min().unwrap();
        assert_eq!(
            Vec::from(pvt.column("m").unwrap().i32().unwrap()),
            &[None, Some(2), None]
        );
        let pvt = df.groupby("foo").unwrap().pivot("bar", "N").max().unwrap();
        assert_eq!(
            Vec::from(pvt.column("m").unwrap().i32().unwrap()),
            &[None, Some(4), None]
        );
        let pvt = df.groupby("foo").unwrap().pivot("bar", "N").mean().unwrap();
        assert_eq!(
            Vec::from(pvt.column("m").unwrap().i32().unwrap()),
            &[None, Some(3), None]
        );
    }
}
