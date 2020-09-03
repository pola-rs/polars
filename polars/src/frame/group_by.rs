use super::hash_join::prepare_hashed_relation;
use crate::chunked_array::builder::PrimitiveChunkedBuilder;
use crate::frame::select::Selection;
use crate::prelude::*;
use arrow::array::{PrimitiveBuilder, StringBuilder};
use enum_dispatch::enum_dispatch;
use num::{Num, NumCast, ToPrimitive, Zero};
use rayon::prelude::*;
use std::hash::Hash;

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

impl Series {
    fn as_groupable_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<Groupable>> + 'a> {
        macro_rules! as_groupable_iter {
            ($ca:expr, $variant:ident ) => {{
                Box::new(
                    $ca.into_iter()
                        .map(|opt_b| opt_b.map(|b| Groupable::$variant(b))),
                )
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
            _ => unimplemented!(),
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
    pub fn groupby<'g, S: Selection<'g>>(&self, by: S) -> Result<GroupBy> {
        let selected_keys = self.select_series(by)?;

        let groups = match selected_keys.len() {
            1 => selected_keys[0].group_tuples(),
            2 => {
                let iter = selected_keys[0]
                    .as_groupable_iter()
                    .zip(selected_keys[1].as_groupable_iter());
                groupby(iter)
            }
            3 => {
                let iter = selected_keys[0]
                    .as_groupable_iter()
                    .zip(selected_keys[1].as_groupable_iter())
                    .zip(selected_keys[2].as_groupable_iter());
                groupby(iter)
            }
            4 => {
                let iter = selected_keys[0]
                    .as_groupable_iter()
                    .zip(selected_keys[1].as_groupable_iter())
                    .zip(selected_keys[2].as_groupable_iter())
                    .zip(selected_keys[3].as_groupable_iter());
                groupby(iter)
            }
            5 => {
                let iter = selected_keys[0]
                    .as_groupable_iter()
                    .zip(selected_keys[1].as_groupable_iter())
                    .zip(selected_keys[2].as_groupable_iter())
                    .zip(selected_keys[3].as_groupable_iter())
                    .zip(selected_keys[4].as_groupable_iter());
                groupby(iter)
            }
            _ => {
                return Err(PolarsError::Other(
                    "more than 5 combined keys are currently not supported".to_string(),
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
pub struct GroupBy<'a, 'b> {
    df: &'a DataFrame,
    selected_keys: Vec<Series>,
    // [first idx, [other idx]]
    groups: Vec<(usize, Vec<usize>)>,
    // columns selected for aggregation
    selected_agg: Option<Vec<&'b str>>,
}

#[enum_dispatch(Series)]
trait NumericAggSync {
    fn agg_mean(&self, _groups: &Vec<(usize, Vec<usize>)>) -> Series {
        unimplemented!()
    }
    fn agg_min(&self, _groups: &Vec<(usize, Vec<usize>)>) -> Series {
        unimplemented!()
    }
    fn agg_max(&self, _groups: &Vec<(usize, Vec<usize>)>) -> Series {
        unimplemented!()
    }
    fn agg_sum(&self, _groups: &Vec<(usize, Vec<usize>)>) -> Series {
        unimplemented!()
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
    fn agg_mean(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
        Series::Float64(
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
        )
    }

    fn agg_min(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
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
                    let take =
                        unsafe { self.take_unchecked(idx.into_iter().copied(), Some(self.len())) };
                    take.min()
                }
            })
            .collect::<ChunkedArray<T>>()
            .into_series()
    }

    fn agg_max(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
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
                    let take =
                        unsafe { self.take_unchecked(idx.into_iter().copied(), Some(self.len())) };
                    take.max()
                }
            })
            .collect::<ChunkedArray<T>>()
            .into_series()
    }

    fn agg_sum(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
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
                    let take =
                        unsafe { self.take_unchecked(idx.into_iter().copied(), Some(self.len())) };
                    take.sum()
                }
            })
            .collect::<ChunkedArray<T>>()
            .into_series()
    }
}

#[enum_dispatch(Series)]
trait AggFirst {
    fn agg_first(&self, _groups: &Vec<(usize, Vec<usize>)>) -> Series {
        unimplemented!()
    }
}

macro_rules! impl_agg_first {
    ($self:ident, $groups:ident, $ca_type:ty) => {{
        $groups
            .par_iter()
            .map(|(first, _idx)| {
                let taker = $self.take_rand();
                taker.get(*first)
            })
            .collect::<$ca_type>()
            .into_series()
    }};
}

impl<T> AggFirst for ChunkedArray<T>
where
    T: PolarsNumericType + std::marker::Sync,
{
    fn agg_first(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
        impl_agg_first!(self, groups, ChunkedArray<T>)
    }
}

impl AggFirst for BooleanChunked {
    fn agg_first(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
        impl_agg_first!(self, groups, BooleanChunked)
    }
}

impl AggFirst for Utf8Chunked {
    fn agg_first(&self, groups: &Vec<(usize, Vec<usize>)>) -> Series {
        groups
            .par_iter()
            .map(|(first, _idx)| {
                let taker = self.take_rand();
                taker.get(*first).map(|s| s.to_string())
            })
            .collect::<Utf8Chunked>()
            .into_series()
    }
}

impl AggFirst for LargeListChunked {}

impl<'a, 'b> GroupBy<'a, 'b> {
    /// Select the column by which the determine the groups.
    /// You can select a single column or a slice of columns.
    pub fn select<S>(mut self, selection: S) -> Self
    where
        S: Selection<'b>,
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
            Some(selection) => selection,
            None => return Err(PolarsError::NoSelection),
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
            let mut agg = agg_col.agg_mean(&self.groups);
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
            let mut agg = agg_col.agg_sum(&self.groups);
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
            let mut agg = agg_col.agg_min(&self.groups);
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
            let mut agg = agg_col.agg_max(&self.groups);
            agg.rename(&new_name);
            cols.push(agg);
        }
        DataFrame::new(cols)
    }

    /// Aggregate grouped series and find the first value per group.
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
                let s = unsafe {
                    agg_col.take_iter_unchecked(idx.into_iter().copied(), Some(idx.len()))
                };
                builder.append_value(s.len() as u32);
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
    ///  +------------+---------------+
    ///  | date       | temp_agg_list |
    ///  | ---        | ---           |
    ///  | date32     | list [i32]    |
    ///  +============+===============+
    ///  | 2020-08-23 | list [Int32]  |
    ///  +------------+---------------+
    ///  | 2020-08-22 | list [Int32]  |
    ///  +------------+---------------+
    ///  | 2020-08-21 | list [Int32]  |
    ///  +------------+---------------+
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
                    builder.append_opt_series(Some(&s))
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
                    builder.append_opt_series(Some(&s))
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
    }
}
