use super::GroupBy;
use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use hashbrown::HashMap;
use num::{Num, NumCast, Zero};
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::hash_map::RandomState;
use std::fmt::{Debug, Formatter};
use std::ops::{Add, Deref};

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
        self.pivot_impl(values, index, columns, false, agg_fn)
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
        self.pivot_impl(values, index, columns, true, agg_fn)
    }

    fn pivot_impl(
        &self,
        // these columns will be aggregated in the nested groupby
        mut values: Vec<String>,
        // keys of the first groupby operation
        index: Vec<String>,
        // these columns will be used for a nested groupby
        // the rows of this nested groupby will be pivoted as header column values
        columns: Vec<String>,
        // sort the group tuples
        maintain_order: bool,
        // aggregation function
        agg_fn: PivotAgg,
    ) -> Result<DataFrame> {
        let groups = match maintain_order {
            true => self.groupby_stable(&index)?.groups,
            false => self.groupby(&index)?.groups,
        };
        // broadcast values argument
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
                let cols = vec![values[i].as_str(), columns[i].as_str()];
                // take only the columns we will use in a smaller dataframe
                self.select(cols)
            })
            .collect::<Result<Vec<_>>>()?;

        // make sure that we make smaller dataframes then the take operations are cheaper
        let index_df = self.select(&index)?;

        let mut im_result = POOL.install(|| {
            groups
                .par_iter()
                .map(|g| {
                    // Here we do a nested group by.
                    // Everything we do here produces a single row in the final dataframe

                    // nested group by keys

                    // safety:
                    // group tuples are in bounds
                    // shape (1, len(keys)
                    let sub_index_df = unsafe { index_df.take_unchecked_slice(&g.1[..1]) };

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
                        let sub_vals_and_cols =
                            unsafe { values_and_columns[i].take_unchecked_slice(&g.1) };

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
                            .outer_join(&df_r, columns[i].as_str(), columns[i].as_str())
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
        let mut index = accumulate_dataframes_vertical(indices).unwrap();

        for values in &mut all_values {
            let mut cols = std::mem::take(&mut values.columns);
            sort_cols(&mut cols, 0);
            index = index.hstack(&cols)?
        }
        Ok(index)
    }
}

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
                let s = self.cast(&DataType::Int32)?;
                *self = s;
                self.as_groupable_iter()
            }
            DataType::UInt32 => as_groupable_iter!(self.u32().unwrap(), UInt32),
            DataType::UInt64 => as_groupable_iter!(self.u64().unwrap(), UInt64),
            DataType::Int32 => as_groupable_iter!(self.i32().unwrap(), Int32),
            DataType::Int64 => as_groupable_iter!(self.i64().unwrap(), Int64),
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
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical => {
                let s = self.cast(&DataType::UInt32)?;
                *self = s;
                self.as_groupable_iter()
            }
            dt => Err(PolarsError::ComputeError(
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
    #[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
    #[deprecated(note = "use DataFrame::pivot")]
    #[allow(deprecated)]
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
#[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
#[deprecated(note = "use DataFrame::pivot")]
#[allow(deprecated)]
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

fn sort_cols(cols: &mut [Series], offset: usize) {
    (&mut cols[offset..]).sort_unstable_by(|s1, s2| {
        if s1.name() > s2.name() {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    });
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
                        _ => unimplemented!(),
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
        sort_cols(&mut cols, 1);

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
    sort_cols(&mut cols, 1);

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

#[cfg(feature = "dtype-categorical")]
impl ChunkPivot for CategoricalChunked {
    fn pivot_count<'a>(
        &self,
        pivot_series: &'a Series,
        keys: Vec<Series>,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<DataFrame> {
        self.deref().pivot_count(pivot_series, keys, groups)
    }
}

impl ChunkPivot for ListChunked {
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
        let mut columns_agg_map_main = {
            // create a hash map that will be filled with the results of the aggregation.
            let mut columns_agg_map_main = PlHashMap::new();
            for column_name in pivot_vec.iter().flatten() {
                columns_agg_map_main.entry(column_name).or_insert_with(|| {
                    get_list_builder(
                        &self.inner_dtype(),
                        groups.len(),
                        groups.len(),
                        &format!("{:?}", column_name),
                    )
                });
            }
            columns_agg_map_main
        };

        // iterate over the groups that need to be aggregated
        // idxes are the indexes of the groups in the keys, pivot, and values columns
        for (_first, idx) in groups {
            // for every group do the aggregation by adding them to the vector belonging by that column
            // the columns are hashed with the pivot values
            let mut columns_agg_map_group =
                create_column_values_map::<Series>(&pivot_vec_unique, idx.len());
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
                        PivotAgg::First => {
                            main_builder.append_opt_series(v[0].as_ref());
                        }
                        _ => unimplemented!(),
                    },
                }
            }
        }
        // Finalize the pivot by creating a vec of all the columns and creating a DataFrame
        let mut cols = keys;
        cols.reserve_exact(columns_agg_map_main.len());

        for (_, mut builder) in columns_agg_map_main {
            let ca = builder.finish();
            cols.push(ca.into_series());
        }
        sort_cols(&mut cols, 1);

        DataFrame::new(cols)
    }
}

#[cfg(feature = "object")]
impl<T> ChunkPivot for ObjectChunked<T> {}

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

fn finish_logical_types(mut out: DataFrame, pivot_series: &Series) -> Result<DataFrame> {
    match pivot_series.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical => {
            let piv = pivot_series.categorical().unwrap();
            let rev_map = piv.categorical_map.as_ref().unwrap();
            for s in out.columns[1..].iter_mut() {
                let category = s.name().parse::<u32>().unwrap();
                let name = rev_map.get(category);
                s.rename(name);
            }
            Ok(out)
        }
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(tu, _) => {
            let fun = match tu {
                TimeUnit::Nanoseconds => timestamp_ns_to_datetime,
                TimeUnit::Milliseconds => timestamp_ms_to_datetime,
            };

            for s in out.columns[1..].iter_mut() {
                let ts = s.name().parse::<i64>().unwrap();
                let nd = fun(ts);
                s.rename(&format!("{}", nd));
            }
            Ok(out)
        }
        #[cfg(feature = "dtype-date")]
        DataType::Date => {
            for s in out.columns[1..].iter_mut() {
                let days = s.name().parse::<i32>().unwrap();
                let nd = date32_to_date(days);
                s.rename(&format!("{}", nd));
            }
            Ok(out)
        }
        _ => Ok(out),
    }
}

#[allow(deprecated)]
impl<'df, 'sel_str> Pivot<'df, 'sel_str> {
    /// Aggregate the pivot results by taking the count values.
    #[deprecated(note = "use DataFrame::pivot")]
    pub fn count(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        let out = values_series.pivot_count(
            &pivot_series.to_physical_repr(),
            self.gb.keys(),
            &self.gb.groups,
        )?;
        finish_logical_types(out, pivot_series)
    }

    /// Aggregate the pivot results by taking the first occurring value.
    #[deprecated(note = "use DataFrame::pivot")]
    pub fn first(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        let out = values_series.pivot(
            &pivot_series.to_physical_repr(),
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::First,
        )?;
        finish_logical_types(out, pivot_series)
    }

    /// Aggregate the pivot results by taking the sum of all duplicates.
    #[deprecated(note = "use DataFrame::pivot")]
    pub fn sum(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        let out = values_series.pivot(
            &pivot_series.to_physical_repr(),
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::Sum,
        )?;
        finish_logical_types(out, pivot_series)
    }

    /// Aggregate the pivot results by taking the minimal value of all duplicates.
    #[deprecated(note = "use DataFrame::pivot")]
    pub fn min(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        let out = values_series.pivot(
            &pivot_series.to_physical_repr(),
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::Min,
        )?;
        finish_logical_types(out, pivot_series)
    }

    /// Aggregate the pivot results by taking the maximum value of all duplicates.
    #[deprecated(note = "use DataFrame::pivot")]
    pub fn max(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        let out = values_series.pivot(
            &pivot_series.to_physical_repr(),
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::Max,
        )?;
        finish_logical_types(out, pivot_series)
    }

    /// Aggregate the pivot results by taking the mean value of all duplicates.
    #[deprecated(note = "use DataFrame::pivot")]
    pub fn mean(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        let out = values_series.pivot(
            &pivot_series.to_physical_repr(),
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::Mean,
        )?;
        finish_logical_types(out, pivot_series)
    }
    /// Aggregate the pivot results by taking the median value of all duplicates.
    #[deprecated(note = "use DataFrame::pivot")]
    pub fn median(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        let out = values_series.pivot(
            &pivot_series.to_physical_repr(),
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::Median,
        )?;
        finish_logical_types(out, pivot_series)
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

        let pvt = df.groupby("foo").unwrap().pivot("bar", "N").sum().unwrap();
        assert_eq!(pvt.get_column_names(), &["foo", "k", "l", "m"]);
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

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_pivot_categorical() -> Result<()> {
        let mut df = df![
            "A" => [1, 1, 1, 1, 1, 1, 1, 1],
            "B" => [8, 2, 3, 6, 3, 6, 2, 2],
            "C" => ["a", "b", "c", "a", "b", "c", "a", "b"]
        ]?;
        df.try_apply("C", |s| s.cast(&DataType::Categorical))?;

        let out = df.groupby("B")?.pivot("C", "A").count()?;
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

        let out = df.groupby("B")?.pivot("C", "A").count()?;
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
