use super::GroupBy;
use crate::prelude::*;
use hashbrown::HashMap;
use itertools::Itertools;
use num::{Num, NumCast, Zero};
use std::collections::hash_map::RandomState;
use std::fmt::{Debug, Formatter};
use std::ops::Add;

/// Utility enum used for grouping on multiple columns
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub(crate) enum Groupable<'a> {
    Boolean(bool),
    Utf8(&'a str),
    UInt32(u32),
    UInt64(u64),
    Int32(i32),
    Int64(i64),
}

impl<'a> Debug for Groupable<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use Groupable::*;
        match self {
            Boolean(v) => write!(f, "{}", v),
            Utf8(v) => write!(f, "{}", v),
            UInt32(v) => write!(f, "{}", v),
            UInt64(v) => write!(f, "{}", v),
            Int32(v) => write!(f, "{}", v),
            Int64(v) => write!(f, "{}", v),
        }
    }
}

impl Series {
    pub(crate) fn as_groupable_iter<'a>(
        // mutable reference is needed to put an owned cast to back to the callers location.
        // this allows us to return a reference to 'a
        // This still is quite hacky. This should probably be reimplemented.
        &'a mut self,
    ) -> Result<Box<dyn Iterator<Item = Option<Groupable>> + 'a + Send>> {
        macro_rules! as_groupable_iter {
            ($ca:expr, $variant:ident ) => {{
                let bx = Box::new($ca.into_iter().map(|opt_b| opt_b.map(Groupable::$variant)));
                Ok(bx)
            }};
        }

        match self.dtype() {
            DataType::Boolean => as_groupable_iter!(self.bool().unwrap(), Boolean),
            DataType::Int8 | DataType::UInt8 | DataType::Int16 | DataType::UInt16 => {
                let s = self.cast::<Int32Type>()?;
                *self = s;
                self.as_groupable_iter()
            }
            DataType::UInt32 => as_groupable_iter!(self.u32().unwrap(), UInt32),
            DataType::UInt64 => as_groupable_iter!(self.u64().unwrap(), UInt64),
            DataType::Int32 => as_groupable_iter!(self.i32().unwrap(), Int32),
            DataType::Int64 => as_groupable_iter!(self.i64().unwrap(), Int64),
            DataType::Date32 => {
                let s = self.cast::<Int32Type>()?;
                *self = s;
                self.as_groupable_iter()
            }
            DataType::Date64 => {
                let s = self.cast::<Int64Type>()?;
                *self = s;
                self.as_groupable_iter()
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                let s = self.cast::<Int64Type>()?;
                *self = s;
                self.as_groupable_iter()
            }
            DataType::Utf8 => as_groupable_iter!(self.utf8().unwrap(), Utf8),
            DataType::Float32 => {
                let s = self.f32()?.bit_repr_small().into_series();
                *self = s;
                self.as_groupable_iter()
            }
            DataType::Float64 => {
                let s = self.f64()?.bit_repr_small().into_series();
                *self = s;
                self.as_groupable_iter()
            }
            DataType::Categorical => {
                let s = self.cast::<UInt32Type>()?;
                *self = s;
                self.as_groupable_iter()
            }
            dt => Err(PolarsError::Other(
                format!("Column with dtype {:?} is not groupable", dt).into(),
            )),
        }
    }
}

impl<'df, 'selection_str> GroupBy<'df, 'selection_str> {
    /// Pivot a column of the current `DataFrame` and perform one of the following aggregations:
    ///
    /// * first
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
    ///     let df = df!("foo" => &["A", "A", "B", "B", "C"],
    ///         "N" => &[1, 2, 2, 4, 2],
    ///         "bar" => &["k", "l", "m", "n", "0"]
    ///         )?;
    ///
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
    #[cfg_attr(docsrs, doc(cfg(feature = "pivot")))]
    pub fn pivot(
        &mut self,
        pivot_column: &'selection_str str,
        values_column: &'selection_str str,
    ) -> Pivot {
        // same as select method
        self.selected_agg = Some(vec![pivot_column, values_column]);

        Pivot {
            gb: self,
            pivot_column,
            values_column,
        }
    }
}

/// Intermediate structure when a `pivot` operation is applied.
/// See [the pivot method for more information.](../group_by/struct.GroupBy.html#method.pivot)
#[cfg_attr(docsrs, doc(cfg(feature = "pivot")))]
pub struct Pivot<'df, 'selection_str> {
    gb: &'df GroupBy<'df, 'selection_str>,
    pivot_column: &'selection_str str,
    values_column: &'selection_str str,
}

pub(crate) trait ChunkPivot {
    fn pivot<'a>(
        &self,
        _pivot_series: &'a Series,
        _keys: Vec<Series>,
        _groups: &[(u32, Vec<u32>)],
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
        _groups: &[(u32, Vec<u32>)],
    ) -> Result<DataFrame> {
        Err(PolarsError::InvalidOperation(
            "Pivot count operation not implemented for this type".into(),
        ))
    }
}

/// Create a hashmap that maps column/keys names to values. This is not yet the result of the aggregation.
fn create_column_values_map<'a, T>(
    pivot_vec: &'a [Option<Groupable>],
    size: usize,
) -> HashMap<&'a Groupable<'a>, Vec<Option<T>>, RandomState> {
    let mut columns_agg_map = HashMap::with_capacity_and_hasher(size, RandomState::new());

    for column_name in pivot_vec.iter().flatten() {
        columns_agg_map.entry(column_name).or_insert_with(Vec::new);
    }

    columns_agg_map
}

/// Create a hashmap that maps columns/keys to the result of the aggregation.
fn create_new_column_builder_map<'a, T>(
    pivot_vec: &'a [Option<Groupable>],
    groups: &[(u32, Vec<u32>)],
) -> PlHashMap<&'a Groupable<'a>, PrimitiveChunkedBuilder<T>>
where
    T: PolarsNumericType,
{
    // create a hash map that will be filled with the results of the aggregation.
    let mut columns_agg_map_main = PlHashMap::new();
    for column_name in pivot_vec.iter().flatten() {
        columns_agg_map_main.entry(column_name).or_insert_with(|| {
            PrimitiveChunkedBuilder::<T>::new(&format!("{:?}", column_name), groups.len())
        });
    }
    columns_agg_map_main
}

impl<T> ChunkPivot for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Copy + Num + NumCast + PartialOrd,
    ChunkedArray<T>: IntoSeries,
{
    fn pivot<'a>(
        &self,
        pivot_series: &'a Series,
        keys: Vec<Series>,
        groups: &[(u32, Vec<u32>)],
        agg_type: PivotAgg,
    ) -> Result<DataFrame> {
        // TODO: save an allocation by creating a random access struct for the Groupable utility type.

        // Note: we also create pivot_vec with unique values, otherwise we have quadratic behavior
        let mut pivot_series = pivot_series.clone();
        let mut pivot_unique = pivot_series.unique()?;
        let iter = pivot_unique.as_groupable_iter()?;
        let pivot_vec_unique: Vec<_> = iter.collect();
        let iter = pivot_series.as_groupable_iter()?;
        let pivot_vec: Vec<_> = iter.collect();
        let values_taker = self.take_rand();
        // create a hash map that will be filled with the results of the aggregation.
        let mut columns_agg_map_main =
            create_new_column_builder_map::<T>(&pivot_vec_unique, groups);

        // iterate over the groups that need to be aggregated
        // idxes are the indexes of the groups in the keys, pivot, and values columns
        for (_first, idx) in groups {
            // for every group do the aggregation by adding them to the vector belonging by that column
            // the columns are hashed with the pivot values
            let mut columns_agg_map_group =
                create_column_values_map::<T::Native>(&pivot_vec_unique, idx.len());
            for &i in idx {
                let i = i as usize;
                let opt_pivot_val = unsafe { pivot_vec.get_unchecked(i) };

                if let Some(pivot_val) = opt_pivot_val {
                    let values_val = values_taker.get(i);
                    if let Some(v) = columns_agg_map_group.get_mut(&pivot_val) {
                        v.push(values_val)
                    }
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

    fn pivot_count<'a>(
        &self,
        pivot_series: &'a Series,
        keys: Vec<Series>,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<DataFrame> {
        pivot_count_impl(self, pivot_series, keys, groups)
    }
}

fn pivot_count_impl<'a, CA: TakeRandom>(
    ca: &CA,
    pivot_series: &'a Series,
    keys: Vec<Series>,
    groups: &[(u32, Vec<u32>)],
) -> Result<DataFrame> {
    let mut pivot_series = pivot_series.clone();
    let mut pivot_unique = pivot_series.unique()?;
    let iter = pivot_unique.as_groupable_iter()?;
    let pivot_vec_unique: Vec<_> = iter.collect();
    let iter = pivot_series.as_groupable_iter()?;
    let pivot_vec: Vec<_> = iter.collect();
    // create a hash map that will be filled with the results of the aggregation.
    let mut columns_agg_map_main =
        create_new_column_builder_map::<UInt32Type>(&pivot_vec_unique, groups);

    // iterate over the groups that need to be aggregated
    // idxes are the indexes of the groups in the keys, pivot, and values columns
    for (_first, idx) in groups {
        // for every group do the aggregation by adding them to the vector belonging by that column
        // the columns are hashed with the pivot values
        let mut columns_agg_map_group =
            create_column_values_map::<CA::Item>(&pivot_vec_unique, idx.len());
        for &i in idx {
            let i = i as usize;
            let opt_pivot_val = unsafe { pivot_vec.get_unchecked(i) };

            if let Some(pivot_val) = opt_pivot_val {
                let values_val = ca.get(i);
                if let Some(v) = columns_agg_map_group.get_mut(&pivot_val) {
                    v.push(values_val)
                }
            }
        }

        // After the vectors are filled we really do the aggregation and add the result to the main
        // hash map, mapping pivot values as column to aggregate result.
        for (k, v) in &mut columns_agg_map_group {
            let main_builder = columns_agg_map_main.get_mut(k).unwrap();
            main_builder.append_value(v.len() as u32)
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

impl ChunkPivot for BooleanChunked {
    fn pivot_count<'a>(
        &self,
        pivot_series: &'a Series,
        keys: Vec<Series>,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<DataFrame> {
        pivot_count_impl(self, pivot_series, keys, groups)
    }
}
impl ChunkPivot for Utf8Chunked {
    fn pivot_count<'a>(
        &self,
        pivot_series: &'a Series,
        keys: Vec<Series>,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<DataFrame> {
        pivot_count_impl(&self, pivot_series, keys, groups)
    }
}

impl ChunkPivot for CategoricalChunked {
    fn pivot_count<'a>(
        &self,
        pivot_series: &'a Series,
        keys: Vec<Series>,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<DataFrame> {
        self.cast::<UInt32Type>()
            .unwrap()
            .pivot_count(pivot_series, keys, groups)
    }
}

impl ChunkPivot for ListChunked {}
#[cfg(feature = "object")]
impl<T> ChunkPivot for ObjectChunked<T> {}

pub enum PivotAgg {
    First,
    Sum,
    Min,
    Max,
    Mean,
    Median,
}

fn pivot_agg_first<T>(builder: &mut PrimitiveChunkedBuilder<T>, v: &[Option<T::Native>])
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

fn pivot_agg_sum<T>(builder: &mut PrimitiveChunkedBuilder<T>, v: &[Option<T::Native>])
where
    T: PolarsNumericType,
    T::Native: Num + Zero,
{
    builder.append_option(v.iter().copied().fold_options(Zero::zero(), Add::add));
}

fn pivot_agg_mean<T>(builder: &mut PrimitiveChunkedBuilder<T>, v: &[Option<T::Native>])
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

fn pivot_agg_min<T>(builder: &mut PrimitiveChunkedBuilder<T>, v: &[Option<T::Native>])
where
    T: PolarsNumericType,
    T::Native: PartialOrd,
{
    let mut min = None;

    for val in v.iter().flatten() {
        match min {
            None => min = Some(*val),
            Some(minimum) => {
                if val < &minimum {
                    min = Some(*val)
                }
            }
        }
    }

    builder.append_option(min);
}

fn pivot_agg_max<T>(builder: &mut PrimitiveChunkedBuilder<T>, v: &[Option<T::Native>])
where
    T: PolarsNumericType,
    T::Native: PartialOrd,
{
    let mut max = None;

    for val in v.iter().flatten() {
        match max {
            None => max = Some(*val),
            Some(maximum) => {
                if val > &maximum {
                    max = Some(*val)
                }
            }
        }
    }

    builder.append_option(max);
}

impl<'df, 'sel_str> Pivot<'df, 'sel_str> {
    /// Aggregate the pivot results by taking the count the values.
    pub fn count(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        values_series.pivot_count(pivot_series, self.gb.keys(), &self.gb.groups)
    }

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
    use super::*;
    #[test]
    fn test_pivot() {
        let s0 = Series::new("foo", ["A", "A", "B", "B", "C"].as_ref());
        let s1 = Series::new("N", [1, 2, 2, 4, 2].as_ref());
        let s2 = Series::new("bar", ["k", "l", "m", "m", "l"].as_ref());
        let df = DataFrame::new(vec![s0, s1, s2]).unwrap();
        println!("{:?}", df);

        let pvt = df.groupby("foo").unwrap().pivot("bar", "N").sum().unwrap();
        assert_eq!(
            Vec::from(&pvt.column("m").unwrap().i32().unwrap().sort(false)),
            &[None, None, Some(6)]
        );
        let pvt = df.groupby("foo").unwrap().pivot("bar", "N").min().unwrap();
        assert_eq!(
            Vec::from(&pvt.column("m").unwrap().i32().unwrap().sort(false)),
            &[None, None, Some(2)]
        );
        let pvt = df.groupby("foo").unwrap().pivot("bar", "N").max().unwrap();
        assert_eq!(
            Vec::from(&pvt.column("m").unwrap().i32().unwrap().sort(false)),
            &[None, None, Some(4)]
        );
        let pvt = df.groupby("foo").unwrap().pivot("bar", "N").mean().unwrap();
        assert_eq!(
            Vec::from(&pvt.column("m").unwrap().i32().unwrap().sort(false)),
            &[None, None, Some(3)]
        );
        let pvt = df
            .groupby("foo")
            .unwrap()
            .pivot("bar", "N")
            .count()
            .unwrap();
        assert_eq!(
            Vec::from(&pvt.column("m").unwrap().u32().unwrap().sort(false)),
            &[Some(0), Some(0), Some(2)]
        );
    }
}
