use crate::chunked_array::kernels::take_agg::{
    take_agg_no_null_primitive_iter_unchecked, take_agg_primitive_iter_unchecked,
};
use crate::chunked_array::{builder::PrimitiveChunkedBuilder, float::IntegerDecode};
use crate::frame::select::Selection;
use crate::prelude::*;
use crate::utils::{accumulate_dataframes_vertical, split_ca, split_df, NoNull};
use crate::vector_hasher::{
    create_hash_and_keys_threaded_vectorized, df_rows_to_hashes, df_rows_to_hashes_threaded,
    prepare_hashed_relation, IdBuildHasher, IdxHash,
};
use crate::POOL;
use ahash::RandomState;
use hashbrown::{hash_map::RawEntryMut, HashMap};
use itertools::Itertools;
use num::{Bounded, Num, NumCast, ToPrimitive, Zero};
use polars_arrow::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::hash::{BuildHasher, Hash, Hasher};
use std::{
    fmt::{Debug, Formatter},
    ops::Add,
};

pub trait VecHash {
    /// Compute the hase for all values in the array.
    ///
    /// This currently only works with the ahash RandomState hasher builder.
    fn vec_hash(&self, _random_state: RandomState) -> UInt64Chunked {
        unimplemented!()
    }
}

impl<T> VecHash for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Hash,
{
    fn vec_hash(&self, random_state: RandomState) -> UInt64Chunked {
        if self.null_count() == 0 {
            self.apply_cast_numeric(|v| {
                let mut hasher = random_state.build_hasher();
                v.hash(&mut hasher);
                hasher.finish()
            })
        } else {
            self.branch_apply_cast_numeric_no_null(|opt_v| {
                let mut hasher = random_state.build_hasher();
                opt_v.hash(&mut hasher);
                hasher.finish()
            })
        }
    }
}

impl VecHash for Utf8Chunked {
    fn vec_hash(&self, random_state: RandomState) -> UInt64Chunked {
        if self.null_count() == 0 {
            self.apply_cast_numeric(|v| {
                let mut hasher = random_state.build_hasher();
                v.hash(&mut hasher);
                hasher.finish()
            })
        } else {
            self.branch_apply_cast_numeric_no_null(|opt_v| {
                let mut hasher = random_state.build_hasher();
                opt_v.hash(&mut hasher);
                hasher.finish()
            })
        }
    }
}

impl VecHash for BooleanChunked {
    fn vec_hash(&self, random_state: RandomState) -> UInt64Chunked {
        if self.null_count() == 0 {
            self.apply_cast_numeric(|v| {
                let mut hasher = random_state.build_hasher();
                v.hash(&mut hasher);
                hasher.finish()
            })
        } else {
            self.branch_apply_cast_numeric_no_null(|opt_v| {
                let mut hasher = random_state.build_hasher();
                opt_v.hash(&mut hasher);
                hasher.finish()
            })
        }
    }
}

impl VecHash for Float32Chunked {
    fn vec_hash(&self, random_state: RandomState) -> UInt64Chunked {
        if self.null_count() == 0 {
            self.apply_cast_numeric(|v| {
                let v = v.to_bits();
                let mut hasher = random_state.build_hasher();
                v.hash(&mut hasher);
                hasher.finish()
            })
        } else {
            self.branch_apply_cast_numeric_no_null(|opt_v| {
                let opt_v = opt_v.map(|v| v.to_bits());
                let mut hasher = random_state.build_hasher();
                opt_v.hash(&mut hasher);
                hasher.finish()
            })
        }
    }
}
impl VecHash for Float64Chunked {
    fn vec_hash(&self, random_state: RandomState) -> UInt64Chunked {
        if self.null_count() == 0 {
            self.apply_cast_numeric(|v| {
                let v = v.to_bits();
                let mut hasher = random_state.build_hasher();
                v.hash(&mut hasher);
                hasher.finish()
            })
        } else {
            self.branch_apply_cast_numeric_no_null(|opt_v| {
                let opt_v = opt_v.map(|v| v.to_bits());
                let mut hasher = random_state.build_hasher();
                opt_v.hash(&mut hasher);
                hasher.finish()
            })
        }
    }
}

impl VecHash for ListChunked {}

fn groupby<T>(a: impl Iterator<Item = T>) -> Vec<(u32, Vec<u32>)>
where
    T: Hash + Eq,
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

fn groupby_threaded_flat<I, T>(iters: Vec<I>, group_size_hint: usize) -> Vec<(u32, Vec<u32>)>
where
    I: IntoIterator<Item = T> + Send,
    T: Send + Hash + Eq + Sync + Copy,
{
    groupby_threaded(iters, group_size_hint)
        .into_iter()
        .flatten()
        .collect()
}

/// Determine groupby tuples from an iterator. The group_size_hint is used to pre-allocate the group vectors.
/// When the grouping column is a categorical type we already have a good indication of the avg size of the groups.
fn groupby_threaded<I, T>(iters: Vec<I>, group_size_hint: usize) -> Vec<Vec<(u32, Vec<u32>)>>
where
    I: IntoIterator<Item = T> + Send,
    T: Send + Hash + Eq + Sync + Copy,
{
    let n_threads = iters.len();
    let (hashes_and_keys, random_state) = create_hash_and_keys_threaded_vectorized(iters, None);
    let size = hashes_and_keys.iter().fold(0, |acc, v| acc + v.len());

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_threads).into_par_iter().map(|thread_no| {
            let random_state = random_state.clone();
            let hashes_and_keys = &hashes_and_keys;
            let thread_no = thread_no as u64;

            let mut hash_tbl: HashMap<T, (u32, Vec<u32>), RandomState> =
                HashMap::with_capacity_and_hasher(size / n_threads, random_state);

            let n_threads = n_threads as u64;
            let mut offset = 0;
            for hashes_and_keys in hashes_and_keys {
                let len = hashes_and_keys.len() as u32;
                hashes_and_keys
                    .iter()
                    .enumerate()
                    .for_each(|(idx, (h, k))| {
                        let idx = idx as u32;
                        // partition hashes by thread no.
                        // So only a part of the hashes go to this hashmap
                        if (h + thread_no) % n_threads == 0 {
                            let idx = idx + offset;
                            let entry = hash_tbl
                                .raw_entry_mut()
                                // uses the key to check equality to find and entry
                                .from_key_hashed_nocheck(*h, &k);

                            match entry {
                                RawEntryMut::Vacant(entry) => {
                                    let mut tuples = Vec::with_capacity(group_size_hint);
                                    tuples.push(idx);
                                    entry.insert_hashed_nocheck(*h, *k, (idx, tuples));
                                }
                                RawEntryMut::Occupied(mut entry) => {
                                    let (_k, v) = entry.get_key_value_mut();
                                    v.1.push(idx);
                                }
                            }
                        }
                    });

                offset += len;
            }
            hash_tbl.into_iter().map(|(_k, v)| v).collect::<Vec<_>>()
        })
    })
    .collect()
}

/// Utility function used as comparison function in the hashmap.
/// The rationale is that equality is an AND operation and therefore its probability of success
/// declines rapidly with the number of keys. Instead of first copying an entire row from both
/// sides and then do the comparison, we do the comparison value by value catching early failures
/// eagerly.
///
/// # Safety
/// Doesn't check any bounds
unsafe fn compare_fn(keys: &DataFrame, idx_a: u32, idx_b: u32) -> bool {
    for s in keys.get_columns() {
        if !(s.get_unchecked(idx_a as usize) == s.get_unchecked(idx_b as usize)) {
            return false;
        }
    }
    true
}

fn populate_multiple_key_hashmap(
    hash_tbl: &mut HashMap<IdxHash, (u32, Vec<u32>), IdBuildHasher>,
    idx: u32,
    h: u64,
    keys: &DataFrame,
) {
    let entry = hash_tbl
        .raw_entry_mut()
        // uses the idx to probe rows in the original DataFrame with keys
        // to check equality to find an entry
        .from_hash(h, |idx_hash| {
            let key_idx = idx_hash.idx;
            // Safety:
            // indices in a join operation are always in bounds.
            unsafe { compare_fn(keys, key_idx, idx) }
        });
    match entry {
        RawEntryMut::Vacant(entry) => {
            entry.insert_hashed_nocheck(h, IdxHash::new(idx, h), (idx, vec![idx]));
        }
        RawEntryMut::Occupied(mut entry) => {
            let (_k, v) = entry.get_key_value_mut();
            v.1.push(idx);
        }
    }
}

fn groupby_multiple_keys(keys: DataFrame) -> Vec<(u32, Vec<u32>)> {
    let (hashes, _) = df_rows_to_hashes(&keys, None);
    let size = hashes.len();
    // rather over allocate because rehashing is expensive
    let mut hash_tbl: HashMap<IdxHash, (u32, Vec<u32>), IdBuildHasher> =
        HashMap::with_capacity_and_hasher(size, IdBuildHasher::default());

    // hashes has no nulls
    let mut idx = 0;
    for hashes_chunk in hashes.data_views() {
        for &h in hashes_chunk {
            populate_multiple_key_hashmap(&mut hash_tbl, idx, h, &keys);
            idx += 1;
        }
    }
    hash_tbl.into_iter().map(|(_k, v)| v).collect::<Vec<_>>()
}

fn groupby_threaded_multiple_keys_flat(keys: DataFrame, n_threads: usize) -> Vec<(u32, Vec<u32>)> {
    let dfs = split_df(&keys, n_threads).unwrap();
    let (hashes, _random_state) = df_rows_to_hashes_threaded(&dfs, None);
    let size = hashes.len();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.

    // We use a combination of a custom IdentityHasher and a uitility key IdxHash that stores
    // the index of the row and and the hash. The Hash function of this key just returns the hash it stores.
    POOL.install(|| {
        (0..n_threads).into_par_iter().map(|thread_no| {
            let hashes = &hashes;
            let thread_no = thread_no as u64;

            let keys = &keys;

            // rather over allocate because rehashing is expensive
            let mut hash_tbl: HashMap<IdxHash, (u32, Vec<u32>), IdBuildHasher> =
                HashMap::with_capacity_and_hasher(size / n_threads, IdBuildHasher::default());

            let n_threads = n_threads as u64;
            let mut offset = 0;
            for hashes in hashes {
                let len = hashes.len() as u32;

                let mut idx = 0;
                for hashes_chunk in hashes.data_views() {
                    for &h in hashes_chunk {
                        // partition hashes by thread no.
                        // So only a part of the hashes go to this hashmap
                        if (h + thread_no) % n_threads == 0 {
                            let idx = idx + offset;
                            populate_multiple_key_hashmap(&mut hash_tbl, idx, h, &keys);
                        }
                        idx += 1;
                    }
                }

                offset += len;
            }
            hash_tbl.into_iter().map(|(_k, v)| v).collect::<Vec<_>>()
        })
    })
    .flatten()
    .collect()
}

/// Used to create the tuples for a groupby operation.
pub trait IntoGroupTuples {
    /// Create the tuples need for a groupby operation.
    ///     * The first value in te tuple is the first index of the group.
    ///     * The second value in the tuple is are the indexes of the groups including the first value.
    fn group_tuples(&self, _multithreaded: bool) -> Vec<(u32, Vec<u32>)> {
        unimplemented!()
    }
}

fn group_multithreaded<T>(ca: &ChunkedArray<T>) -> bool {
    // TODO! change to something sensible
    ca.len() > 1000
}

macro_rules! group_tuples {
    ($ca: expr, $multithreaded: expr) => {{
        // TODO! choose a splitting len
        if $multithreaded && group_multithreaded($ca) {
            let n_threads = num_cpus::get();
            let splitted = split_ca($ca, n_threads).unwrap();

            if $ca.null_count() == 0 {
                let iters = splitted
                    .iter()
                    .map(|ca| ca.into_no_null_iter())
                    .collect_vec();
                groupby_threaded_flat(iters, 0)
            } else {
                let iters = splitted.iter().map(|ca| ca.into_iter()).collect_vec();
                groupby_threaded_flat(iters, 0)
            }
        } else {
            if $ca.null_count() == 0 {
                groupby($ca.into_no_null_iter())
            } else {
                groupby($ca.into_iter())
            }
        }
    }};
}

impl<T> IntoGroupTuples for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Eq + Hash + Send,
{
    fn group_tuples(&self, multithreaded: bool) -> Vec<(u32, Vec<u32>)> {
        let group_size_hint = if let Some(m) = &self.categorical_map {
            self.len() / m.len()
        } else {
            0
        };
        if multithreaded && group_multithreaded(self) {
            let n_threads = num_cpus::get();
            let splitted = split_ca(self, n_threads).unwrap();

            // use the arrays as iterators
            if self.chunks.len() == 1 {
                if self.null_count() == 0 {
                    let iters = splitted
                        .iter()
                        .map(|ca| ca.downcast_chunks().into_iter().map(|array| array.values()))
                        .flatten()
                        .collect_vec();
                    groupby_threaded_flat(iters, group_size_hint)
                } else {
                    let iters = splitted
                        .iter()
                        .map(|ca| ca.downcast_chunks().into_iter())
                        .flatten()
                        .collect_vec();
                    groupby_threaded_flat(iters, group_size_hint)
                }
                // use the polars-iterators
            } else if self.null_count() == 0 {
                let iters = splitted
                    .iter()
                    .map(|ca| ca.into_no_null_iter())
                    .collect_vec();
                groupby_threaded_flat(iters, group_size_hint)
            } else {
                let iters = splitted.iter().map(|ca| ca.into_iter()).collect_vec();
                groupby_threaded_flat(iters, group_size_hint)
            }
        } else if self.null_count() == 0 {
            groupby(self.into_no_null_iter())
        } else {
            groupby(self.into_iter())
        }
    }
}
impl IntoGroupTuples for BooleanChunked {
    fn group_tuples(&self, multithreaded: bool) -> Vec<(u32, Vec<u32>)> {
        group_tuples!(self, multithreaded)
    }
}

impl IntoGroupTuples for Utf8Chunked {
    fn group_tuples(&self, multithreaded: bool) -> Vec<(u32, Vec<u32>)> {
        group_tuples!(self, multithreaded)
    }
}

impl IntoGroupTuples for CategoricalChunked {
    fn group_tuples(&self, multithreaded: bool) -> Vec<(u32, Vec<u32>)> {
        self.cast::<UInt32Type>()
            .unwrap()
            .group_tuples(multithreaded)
    }
}

macro_rules! impl_into_group_tpls_float {
    ($self: ident, $multithreaded:expr) => {
        if $multithreaded && group_multithreaded($self) {
            let n_threads = num_cpus::get();
            let splitted = split_ca($self, n_threads).unwrap();
            match $self.null_count() {
                0 => {
                    let iters = splitted
                        .iter()
                        .map(|ca| ca.into_no_null_iter().map(|v| v.to_bits()))
                        .collect_vec();
                    groupby_threaded_flat(iters, 0)
                }
                _ => {
                    let iters = splitted
                        .iter()
                        .map(|ca| ca.into_iter().map(|opt_v| opt_v.map(|v| v.to_bits())))
                        .collect_vec();
                    groupby_threaded_flat(iters, 0)
                }
            }
        } else {
            match $self.null_count() {
                0 => groupby($self.into_no_null_iter().map(|v| v.to_bits())),
                _ => groupby($self.into_iter().map(|opt_v| opt_v.map(|v| v.to_bits()))),
            }
        }
    };
}

impl IntoGroupTuples for Float64Chunked {
    fn group_tuples(&self, multithreaded: bool) -> Vec<(u32, Vec<u32>)> {
        impl_into_group_tpls_float!(self, multithreaded)
    }
}
impl IntoGroupTuples for Float32Chunked {
    fn group_tuples(&self, multithreaded: bool) -> Vec<(u32, Vec<u32>)> {
        impl_into_group_tpls_float!(self, multithreaded)
    }
}
impl IntoGroupTuples for ListChunked {}
#[cfg(feature = "object")]
impl<T> IntoGroupTuples for ObjectChunked<T> {}

/// Utility enum used for grouping on multiple columns
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub(crate) enum Groupable<'a> {
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
    // mantissa, exponent, sign.
    Float32(u64, i16, i8),
    Float64(u64, i16, i8),
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
            Float32(m, e, s) => write!(f, "float32 mantissa: {} exponent: {} sign: {}", m, e, s),
            Float64(m, e, s) => write!(f, "float64 mantissa: {} exponent: {} sign: {}", m, e, s),
        }
    }
}

impl From<f64> for Groupable<'_> {
    fn from(v: f64) -> Self {
        let (m, e, s) = v.integer_decode();
        Groupable::Float64(m, e, s)
    }
}
impl From<f32> for Groupable<'_> {
    fn from(v: f32) -> Self {
        let (m, e, s) = v.integer_decode();
        Groupable::Float32(m, e, s)
    }
}

fn float_to_groupable_iter<'a, T>(
    ca: &'a ChunkedArray<T>,
) -> Box<dyn Iterator<Item = Option<Groupable>> + 'a + Send>
where
    T: PolarsNumericType,
    T::Native: Into<Groupable<'a>>,
{
    let iter = ca.into_iter().map(|opt_v| opt_v.map(|v| v.into()));
    Box::new(iter)
}

impl<'b> (dyn SeriesTrait + 'b) {
    pub(crate) fn as_groupable_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = Option<Groupable>> + 'a + Send>> {
        macro_rules! as_groupable_iter {
            ($ca:expr, $variant:ident ) => {{
                let bx = Box::new($ca.into_iter().map(|opt_b| opt_b.map(Groupable::$variant)));
                Ok(bx)
            }};
        }

        match self.dtype() {
            DataType::Boolean => as_groupable_iter!(self.bool().unwrap(), Boolean),
            DataType::UInt8 => as_groupable_iter!(self.u8().unwrap(), UInt8),
            DataType::UInt16 => as_groupable_iter!(self.u16().unwrap(), UInt16),
            DataType::UInt32 => as_groupable_iter!(self.u32().unwrap(), UInt32),
            DataType::UInt64 => as_groupable_iter!(self.u64().unwrap(), UInt64),
            DataType::Int8 => as_groupable_iter!(self.i8().unwrap(), Int8),
            DataType::Int16 => as_groupable_iter!(self.i16().unwrap(), Int16),
            DataType::Int32 => as_groupable_iter!(self.i32().unwrap(), Int32),
            DataType::Int64 => as_groupable_iter!(self.i64().unwrap(), Int64),
            DataType::Date32 => {
                as_groupable_iter!(self.date32().unwrap(), Int32)
            }
            DataType::Date64 => {
                as_groupable_iter!(self.date64().unwrap(), Int64)
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                as_groupable_iter!(self.time64_nanosecond().unwrap(), Int64)
            }
            DataType::Duration(TimeUnit::Nanosecond) => {
                as_groupable_iter!(self.duration_nanosecond().unwrap(), Int64)
            }
            DataType::Duration(TimeUnit::Millisecond) => {
                as_groupable_iter!(self.duration_millisecond().unwrap(), Int64)
            }
            DataType::Utf8 => as_groupable_iter!(self.utf8().unwrap(), Utf8),
            DataType::Float32 => Ok(float_to_groupable_iter(self.f32().unwrap())),
            DataType::Float64 => Ok(float_to_groupable_iter(self.f64().unwrap())),
            DataType::Categorical => as_groupable_iter!(self.categorical().unwrap(), UInt32),
            dt => Err(PolarsError::Other(
                format!("Column with dtype {:?} is not groupable", dt).into(),
            )),
        }
    }
}

impl DataFrame {
    pub fn groupby_with_series(&self, by: Vec<Series>, multithreaded: bool) -> Result<GroupBy> {
        if by.is_empty() || by[0].len() != self.height() {
            return Err(PolarsError::ShapeMisMatch(
                "the Series used as keys should have the same length as the DataFrame".into(),
            ));
        };

        // make sure that categorical is used as uint32 in value type
        let keys_df = DataFrame::new(
            by.iter()
                .map(|s| match s.dtype() {
                    DataType::Categorical => s.cast::<UInt32Type>().unwrap(),
                    _ => s.clone(),
                })
                .collect(),
        )?;

        let groups = match by.len() {
            1 => {
                let series = &by[0];
                series.group_tuples(multithreaded)
            }
            _ => {
                if multithreaded {
                    let n_threads = num_cpus::get();
                    groupby_threaded_multiple_keys_flat(keys_df, n_threads)
                } else {
                    groupby_multiple_keys(keys_df)
                }
            }
        };

        Ok(GroupBy {
            df: self,
            selected_keys: by,
            groups,
            selected_agg: None,
        })
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
        gb.groups.sort();
        Ok(gb)
    }
}

/// Returned by a groupby operation on a DataFrame. This struct supports
/// several aggregations.
///
/// Until described otherwise, the examples in this struct are performed on the following DataFrame:
///
/// ```rust
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
    pub(crate) selected_keys: Vec<Series>,
    // [first idx, [other idx]]
    pub(crate) groups: Vec<(u32, Vec<u32>)>,
    // columns selected for aggregation
    selected_agg: Option<Vec<&'selection_str str>>,
}

pub(crate) trait NumericAggSync {
    fn agg_mean(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
    fn agg_min(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
    fn agg_max(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
    fn agg_sum(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
    fn agg_std(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
    fn agg_var(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
}

impl NumericAggSync for BooleanChunked {}
impl NumericAggSync for Utf8Chunked {}
impl NumericAggSync for ListChunked {}
impl NumericAggSync for CategoricalChunked {}
#[cfg(feature = "object")]
impl<T> NumericAggSync for ObjectChunked<T> {}

impl<T> NumericAggSync for ChunkedArray<T>
where
    T: PolarsNumericType + Sync,
    T::Native: std::ops::Add<Output = T::Native> + Num + NumCast + Bounded,
    ChunkedArray<T>: IntoSeries,
{
    fn agg_mean(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        let ca: Float64Chunked = groups
            .par_iter()
            .map(|(first, idx)| {
                if idx.len() == 1 {
                    self.get(*first as usize).map(|sum| sum.to_f64().unwrap())
                } else {
                    match (self.null_count(), self.chunks.len()) {
                        (0, 1) => unsafe {
                            take_agg_no_null_primitive_iter_unchecked(
                                self.downcast_chunks()[0],
                                idx.iter().map(|i| *i as usize),
                                |a, b| a + b,
                                T::Native::zero(),
                            )
                        }
                        .to_f64()
                        .map(|sum| sum / idx.len() as f64),
                        (_, 1) => unsafe {
                            take_agg_primitive_iter_unchecked(
                                self.downcast_chunks()[0],
                                idx.iter().map(|i| *i as usize),
                                |a, b| a + b,
                                T::Native::zero(),
                            )
                        }
                        .map(|sum| sum.to_f64().map(|sum| sum / idx.len() as f64).unwrap()),
                        _ => {
                            let take = unsafe {
                                self.take_unchecked(idx.iter().map(|i| *i as usize).into())
                            };
                            let opt_sum: Option<T::Native> = take.sum();
                            opt_sum.map(|sum| sum.to_f64().unwrap() / idx.len() as f64)
                        }
                    }
                }
            })
            .collect();
        Some(ca.into_series())
    }

    fn agg_min(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        Some(
            groups
                .par_iter()
                .map(|(first, idx)| {
                    if idx.len() == 1 {
                        self.get(*first as usize)
                    } else {
                        match (self.null_count(), self.chunks.len()) {
                            (0, 1) => Some(unsafe {
                                take_agg_no_null_primitive_iter_unchecked(
                                    self.downcast_chunks()[0],
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| if a < b { a } else { b },
                                    T::Native::max_value(),
                                )
                            }),
                            (_, 1) => unsafe {
                                take_agg_primitive_iter_unchecked(
                                    self.downcast_chunks()[0],
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| if a < b { a } else { b },
                                    T::Native::max_value(),
                                )
                            },
                            _ => {
                                let take = unsafe {
                                    self.take_unchecked(idx.iter().map(|i| *i as usize).into())
                                };
                                take.min()
                            }
                        }
                    }
                })
                .collect::<ChunkedArray<T>>()
                .into_series(),
        )
    }

    fn agg_max(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        Some(
            groups
                .par_iter()
                .map(|(first, idx)| {
                    if idx.len() == 1 {
                        self.get(*first as usize)
                    } else {
                        match (self.null_count(), self.chunks.len()) {
                            (0, 1) => Some(unsafe {
                                take_agg_no_null_primitive_iter_unchecked(
                                    self.downcast_chunks()[0],
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| if a > b { a } else { b },
                                    T::Native::min_value(),
                                )
                            }),
                            (_, 1) => unsafe {
                                take_agg_primitive_iter_unchecked(
                                    self.downcast_chunks()[0],
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| if a > b { a } else { b },
                                    T::Native::min_value(),
                                )
                            },
                            _ => {
                                let take = unsafe {
                                    self.take_unchecked(idx.iter().map(|i| *i as usize).into())
                                };
                                take.max()
                            }
                        }
                    }
                })
                .collect::<ChunkedArray<T>>()
                .into_series(),
        )
    }

    fn agg_sum(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        Some(
            groups
                .par_iter()
                .map(|(first, idx)| {
                    if idx.len() == 1 {
                        self.get(*first as usize)
                    } else {
                        match (self.null_count(), self.chunks.len()) {
                            (0, 1) => Some(unsafe {
                                take_agg_no_null_primitive_iter_unchecked(
                                    self.downcast_chunks()[0],
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| a + b,
                                    T::Native::zero(),
                                )
                            }),
                            (_, 1) => unsafe {
                                take_agg_primitive_iter_unchecked(
                                    self.downcast_chunks()[0],
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| a + b,
                                    T::Native::zero(),
                                )
                            },
                            _ => {
                                let take = unsafe {
                                    self.take_unchecked(idx.iter().map(|i| *i as usize).into())
                                };
                                take.sum()
                            }
                        }
                    }
                })
                .collect::<ChunkedArray<T>>()
                .into_series(),
        )
    }
    fn agg_var(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        Some(
            groups
                .par_iter()
                .map(|(_first, idx)| {
                    let take =
                        unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                    take.into_series()
                        .var_as_series()
                        .unpack::<T>()
                        .unwrap()
                        .get(0)
                })
                .collect::<ChunkedArray<T>>()
                .into_series(),
        )
    }
    fn agg_std(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        Some(
            groups
                .par_iter()
                .map(|(_first, idx)| {
                    let take =
                        unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                    take.into_series()
                        .std_as_series()
                        .unpack::<T>()
                        .unwrap()
                        .get(0)
                })
                .collect::<ChunkedArray<T>>()
                .into_series(),
        )
    }
}

pub(crate) trait AggFirst {
    fn agg_first(&self, _groups: &[(u32, Vec<u32>)]) -> Series;
}

macro_rules! impl_agg_first {
    ($self:ident, $groups:ident, $ca_type:ty) => {{
        $groups
            .iter()
            .map(|(first, _idx)| $self.get(*first as usize))
            .collect::<$ca_type>()
            .into_series()
    }};
}

impl<T> AggFirst for ChunkedArray<T>
where
    T: PolarsPrimitiveType + Send,
    ChunkedArray<T>: IntoSeries,
{
    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_first!(self, groups, ChunkedArray<T>)
    }
}

impl AggFirst for BooleanChunked {
    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_first!(self, groups, BooleanChunked)
    }
}

impl AggFirst for Utf8Chunked {
    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_first!(self, groups, Utf8Chunked)
    }
}

impl AggFirst for ListChunked {
    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_first!(self, groups, ListChunked)
    }
}

impl AggFirst for CategoricalChunked {
    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        self.cast::<UInt32Type>()
            .unwrap()
            .agg_first(groups)
            .cast::<CategoricalType>()
            .unwrap()
    }
}

#[cfg(feature = "object")]
impl<T> AggFirst for ObjectChunked<T> {
    fn agg_first(&self, _groups: &[(u32, Vec<u32>)]) -> Series {
        todo!()
    }
}

pub(crate) trait AggLast {
    fn agg_last(&self, _groups: &[(u32, Vec<u32>)]) -> Series;
}

macro_rules! impl_agg_last {
    ($self:ident, $groups:ident, $ca_type:ty) => {{
        $groups
            .iter()
            .map(|(_first, idx)| $self.get(idx[idx.len() - 1] as usize))
            .collect::<$ca_type>()
            .into_series()
    }};
}

impl<T> AggLast for ChunkedArray<T>
where
    T: PolarsPrimitiveType + Send,
    ChunkedArray<T>: IntoSeries,
{
    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_last!(self, groups, ChunkedArray<T>)
    }
}

impl AggLast for BooleanChunked {
    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_last!(self, groups, BooleanChunked)
    }
}

impl AggLast for Utf8Chunked {
    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_last!(self, groups, Utf8Chunked)
    }
}

impl AggLast for CategoricalChunked {
    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        self.cast::<UInt32Type>()
            .unwrap()
            .agg_last(groups)
            .cast::<CategoricalType>()
            .unwrap()
    }
}

impl AggLast for ListChunked {
    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_last!(self, groups, ListChunked)
    }
}

#[cfg(feature = "object")]
impl<T> AggLast for ObjectChunked<T> {
    fn agg_last(&self, _groups: &[(u32, Vec<u32>)]) -> Series {
        todo!()
    }
}

pub(crate) trait AggNUnique {
    fn agg_n_unique(&self, _groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
        None
    }
}

macro_rules! impl_agg_n_unique {
    ($self:ident, $groups:ident, $ca_type:ty) => {{
        $groups
            .into_par_iter()
            .map(|(_first, idx)| {
                if $self.null_count() == 0 {
                    let mut set = HashSet::with_hasher(RandomState::new());
                    for i in idx {
                        let v = unsafe { $self.get_unchecked(*i as usize) };
                        set.insert(v);
                    }
                    set.len() as u32
                } else {
                    let mut set = HashSet::with_hasher(RandomState::new());
                    for i in idx {
                        let opt_v = $self.get(*i as usize);
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
    fn agg_n_unique(&self, groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
        Some(impl_agg_n_unique!(self, groups, NoNull<UInt32Chunked>))
    }
}

// todo! could use mantissa method here
impl AggNUnique for Float32Chunked {}
impl AggNUnique for Float64Chunked {}
impl AggNUnique for ListChunked {}
impl AggNUnique for CategoricalChunked {
    fn agg_n_unique(&self, groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
        self.cast::<UInt32Type>().unwrap().agg_n_unique(groups)
    }
}
#[cfg(feature = "object")]
impl<T> AggNUnique for ObjectChunked<T> {}

// TODO: could be faster as it can only be null, true, or false
impl AggNUnique for BooleanChunked {
    fn agg_n_unique(&self, groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
        Some(impl_agg_n_unique!(self, groups, NoNull<UInt32Chunked>))
    }
}

impl AggNUnique for Utf8Chunked {
    fn agg_n_unique(&self, groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
        Some(impl_agg_n_unique!(self, groups, NoNull<UInt32Chunked>))
    }
}

pub(crate) trait AggList {
    fn agg_list(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
}
impl<T> AggList for ChunkedArray<T>
where
    T: PolarsDataType,
    ChunkedArray<T>: IntoSeries,
{
    fn agg_list(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        // needed capacity for the list
        let values_cap = groups.iter().fold(0, |acc, g| acc + g.1.len());

        macro_rules! impl_gb {
            ($type:ty, $agg_col:expr) => {{
                let values_builder = PrimitiveArrayBuilder::<$type>::new(values_cap);
                let mut builder =
                    ListPrimitiveChunkedBuilder::new("", values_builder, groups.len());
                for (_first, idx) in groups {
                    let s = unsafe {
                        $agg_col.take_iter_unchecked(&mut idx.into_iter().map(|i| *i as usize))
                    };
                    builder.append_opt_series(Some(&s))
                }
                builder.finish().into_series()
            }};
        }

        macro_rules! impl_gb_utf8 {
            ($agg_col:expr) => {{
                let values_builder = LargeStringBuilder::with_capacity(values_cap * 5, values_cap);
                let mut builder = ListUtf8ChunkedBuilder::new("", values_builder, groups.len());
                for (_first, idx) in groups {
                    let s = unsafe {
                        $agg_col.take_iter_unchecked(&mut idx.into_iter().map(|i| *i as usize))
                    };
                    builder.append_series(&s)
                }
                builder.finish().into_series()
            }};
        }

        macro_rules! impl_gb_bool {
            ($agg_col:expr) => {{
                let values_builder = BooleanArrayBuilder::new(values_cap);
                let mut builder = ListBooleanChunkedBuilder::new("", values_builder, groups.len());
                for (_first, idx) in groups {
                    let s = unsafe {
                        $agg_col.take_iter_unchecked(&mut idx.into_iter().map(|i| *i as usize))
                    };
                    builder.append_series(&s)
                }
                builder.finish().into_series()
            }};
        }

        let s = self.clone().into_series();
        Some(match_arrow_data_type_apply_macro!(
            s.dtype(),
            impl_gb,
            impl_gb_utf8,
            impl_gb_bool,
            s
        ))
    }
}

pub(crate) trait AggQuantile {
    fn agg_quantile(&self, _groups: &[(u32, Vec<u32>)], _quantile: f64) -> Option<Series> {
        None
    }

    fn agg_median(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        self.agg_quantile(groups, 0.5)
    }
}

impl<T> AggQuantile for ChunkedArray<T>
where
    T: PolarsNumericType + Sync,
    T::Native: PartialEq,
    ChunkedArray<T>: IntoSeries,
{
    fn agg_quantile(&self, groups: &[(u32, Vec<u32>)], quantile: f64) -> Option<Series> {
        Some(
            groups
                .into_par_iter()
                .map(|(_first, idx)| {
                    let group_vals =
                        unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                    let sorted_idx_ca = group_vals.argsort(false);
                    let sorted_idx = sorted_idx_ca.downcast_chunks()[0].values();
                    let quant_idx = (quantile * (sorted_idx.len() - 1) as f64) as usize;
                    let value_idx = sorted_idx[quant_idx];
                    group_vals.get(value_idx as usize)
                })
                .collect::<ChunkedArray<T>>()
                .into_series(),
        )
    }
}

impl AggQuantile for Utf8Chunked {}
impl AggQuantile for BooleanChunked {}
impl AggQuantile for ListChunked {}
impl AggQuantile for CategoricalChunked {}
#[cfg(feature = "object")]
impl<T> AggQuantile for ObjectChunked<T> {}

impl<'df, 'selection_str> GroupBy<'df, 'selection_str> {
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
    pub fn get_groups(&self) -> &Vec<(u32, Vec<u32>)> {
        &self.groups
    }

    pub fn keys(&self) -> Vec<Series> {
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
                let key =
                    s.take_iter_unchecked(&mut self.groups.iter().map(|(idx, _)| *idx as usize));
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
            return Err(PolarsError::Other(
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
    /// | date32(days) | list [u32] |
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
        cols.shrink_to_fit();
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
    ///  | date32(days) | u32           | i32      | i32      |
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
        let mut map = HashMap::with_capacity_and_hasher(column_to_agg.len(), RandomState::new());
        column_to_agg.iter().for_each(|(column, aggregations)| {
            map.insert(column.as_ref(), aggregations.as_ref());
        });

        macro_rules! finish_agg_opt {
            ($self:ident, $name_fmt:expr, $agg_fn:ident, $agg_col:ident, $cols:ident) => {{
                let new_name = format![$name_fmt, $agg_col.name()];
                let opt_agg = $agg_col.$agg_fn(&$self.groups);
                if let Some(mut agg) = opt_agg {
                    agg.rename(&new_name);
                    $cols.push(agg.into_series());
                }
            }};
        }
        macro_rules! finish_agg {
            ($self:ident, $name_fmt:expr, $agg_fn:ident, $agg_col:ident, $cols:ident) => {{
                let new_name = format![$name_fmt, $agg_col.name()];
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
                            let new_name = format!["{}_count", agg_col.name()];
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
    /// use polars_core::prelude::*;
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

        Pivot {
            gb: self,
            pivot_column,
            values_column,
        }
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
        Min => format!["{}_min", name],
        Max => format!["{}_max", name],
        Median => format!["{}_median", name],
        Mean => format!["{}_mean", name],
        First => format!["{}_first", name],
        Last => format!["{}_last", name],
        Sum => format!["{}_sum", name],
        Groups => "groups".to_string(),
        NUnique => format!["{}_n_unique", name],
        Count => format!["{}_count", name],
        List => format!["{}_agg_list", name],
        Quantile(quantile) => format!["{}_quantile_{:.2}", name, quantile],
        Std => format!["{}_agg_std", name],
        Var => format!["{}_agg_var", name],
    }
}

/// Intermediate structure when a `pivot` operation is applied.
/// See [the pivot method for more information.](../group_by/struct.GroupBy.html#method.pivot)
pub struct Pivot<'df, 'selection_str> {
    gb: &'df GroupBy<'df, 'selection_str>,
    pivot_column: &'selection_str str,
    values_column: &'selection_str str,
}

pub(crate) trait ChunkPivot {
    fn pivot<'a>(
        &self,
        _pivot_series: &'a (dyn SeriesTrait + 'a),
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
        _pivot_series: &'a (dyn SeriesTrait + 'a),
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
) -> HashMap<&'a Groupable<'a>, PrimitiveChunkedBuilder<T>, RandomState>
where
    T: PolarsNumericType,
{
    // create a hash map that will be filled with the results of the aggregation.
    let mut columns_agg_map_main =
        HashMap::with_capacity_and_hasher(pivot_vec.len(), RandomState::new());
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
    T::Native: Copy + Num + NumCast,
    ChunkedArray<T>: IntoSeries,
{
    fn pivot<'a>(
        &self,
        pivot_series: &'a (dyn SeriesTrait + 'a),
        keys: Vec<Series>,
        groups: &[(u32, Vec<u32>)],
        agg_type: PivotAgg,
    ) -> Result<DataFrame> {
        // TODO: save an allocation by creating a random access struct for the Groupable utility type.
        let pivot_unique = pivot_series.unique()?;
        let pivot_vec_unique: Vec<_> = pivot_unique.as_groupable_iter()?.collect();
        let pivot_vec: Vec<_> = pivot_series.as_groupable_iter()?.collect();
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
        pivot_series: &'a (dyn SeriesTrait + 'a),
        keys: Vec<Series>,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<DataFrame> {
        pivot_count_impl(self, pivot_series, keys, groups)
    }
}

fn pivot_count_impl<'a, CA: TakeRandom>(
    ca: &CA,
    pivot_series: &'a (dyn SeriesTrait + 'a),
    keys: Vec<Series>,
    groups: &[(u32, Vec<u32>)],
) -> Result<DataFrame> {
    let pivot_vec: Vec<_> = pivot_series.as_groupable_iter()?.collect();
    // create a hash map that will be filled with the results of the aggregation.
    let mut columns_agg_map_main = create_new_column_builder_map::<UInt32Type>(&pivot_vec, groups);

    // iterate over the groups that need to be aggregated
    // idxes are the indexes of the groups in the keys, pivot, and values columns
    for (_first, idx) in groups {
        // for every group do the aggregation by adding them to the vector belonging by that column
        // the columns are hashed with the pivot values
        let mut columns_agg_map_group = create_column_values_map::<CA::Item>(&pivot_vec, idx.len());
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
        pivot_series: &'a (dyn SeriesTrait + 'a),
        keys: Vec<Series>,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<DataFrame> {
        pivot_count_impl(self, pivot_series, keys, groups)
    }
}
impl ChunkPivot for Utf8Chunked {
    fn pivot_count<'a>(
        &self,
        pivot_series: &'a (dyn SeriesTrait + 'a),
        keys: Vec<Series>,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<DataFrame> {
        pivot_count_impl(&self, pivot_series, keys, groups)
    }
}

impl ChunkPivot for CategoricalChunked {
    fn pivot_count<'a>(
        &self,
        pivot_series: &'a (dyn SeriesTrait + 'a),
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
        values_series.pivot_count(&**pivot_series, self.gb.keys(), &self.gb.groups)
    }

    /// Aggregate the pivot results by taking the first occurring value.
    pub fn first(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        values_series.pivot(
            &**pivot_series,
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::First,
        )
    }

    /// Aggregate the pivot results by taking the sum of all duplicates.
    pub fn sum(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        values_series.pivot(
            &**pivot_series,
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::Sum,
        )
    }

    /// Aggregate the pivot results by taking the minimal value of all duplicates.
    pub fn min(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        values_series.pivot(
            &**pivot_series,
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::Min,
        )
    }

    /// Aggregate the pivot results by taking the maximum value of all duplicates.
    pub fn max(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        values_series.pivot(
            &**pivot_series,
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::Max,
        )
    }

    /// Aggregate the pivot results by taking the mean value of all duplicates.
    pub fn mean(&self) -> Result<DataFrame> {
        let pivot_series = self.gb.df.column(self.pivot_column)?;
        let values_series = self.gb.df.column(self.values_column)?;
        values_series.pivot(
            &**pivot_series,
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
            &**pivot_series,
            self.gb.keys(),
            &self.gb.groups,
            PivotAgg::Median,
        )
    }
}

#[cfg(test)]
mod test {
    use crate::frame::group_by::{groupby, groupby_threaded_flat};
    use crate::prelude::*;
    use crate::utils::split_ca;
    use itertools::Itertools;

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

    #[test]
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
    fn test_groupby_threaded() {
        for slice in &[
            vec![1, 2, 3, 4, 4, 4, 2, 1],
            vec![1, 2, 3, 4, 4, 4, 2, 1, 1],
            vec![1, 2, 3, 4, 4, 4],
        ] {
            let ca = UInt8Chunked::new_from_slice("", &slice);
            let splitted = split_ca(&ca, 4).unwrap();

            let a = groupby(ca.into_iter()).into_iter().sorted().collect_vec();
            let b = groupby_threaded_flat(splitted.iter().map(|ca| ca.into_iter()).collect(), 0)
                .into_iter()
                .sorted()
                .collect_vec();

            assert_eq!(a, b);
        }
    }
}
