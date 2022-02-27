pub(crate) mod multiple_keys;

use polars_arrow::utils::CustomIterTools;

use crate::frame::hash_join::multiple_keys::{
    inner_join_multiple_keys, left_join_multiple_keys, outer_join_multiple_keys,
};
use crate::prelude::*;
use crate::utils::{set_partition_size, split_ca};
use crate::vector_hasher::{
    create_hash_and_keys_threaded_vectorized, prepare_hashed_relation_threaded, this_partition,
    AsU64, StrHash,
};
use crate::{datatypes::PlHashMap, POOL};
use ahash::RandomState;
use hashbrown::hash_map::{Entry, RawEntryMut};
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash, Hasher};

#[cfg(feature = "private")]
pub use self::multiple_keys::private_left_join_multiple_keys;
use crate::frame::groupby::hashing::HASHMAP_INIT_SIZE;
use crate::utils::series::to_physical_and_bit_repr;

/// If Categorical types are created without a global string cache or under
/// a different global string cache the mapping will be incorrect.
#[cfg(feature = "dtype-categorical")]
pub(crate) fn check_categorical_src(l: &DataType, r: &DataType) -> Result<()> {
    match (l, r) {
        (DataType::Categorical(Some(l)), DataType::Categorical(Some(r))) => {
            if !l.same_src(&*r) {
                return Err(PolarsError::ValueError("joins/or comparisons on categorical dtypes can only happen if they are created under the same global string cache".into()));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

macro_rules! det_hash_prone_order {
    ($self:expr, $other:expr) => {{
        // The shortest relation will be used to create a hash table.
        let left_first = $self.len() > $other.len();
        let a;
        let b;
        if left_first {
            a = $self;
            b = $other;
        } else {
            b = $self;
            a = $other;
        }

        (a, b, !left_first)
    }};
}

#[derive(Clone, Debug, PartialEq)]
pub enum JoinType {
    Left,
    Inner,
    Outer,
    #[cfg(feature = "asof_join")]
    AsOf(AsOfOptions),
    Cross,
}

pub(crate) unsafe fn get_hash_tbl_threaded_join_partitioned<T, H>(
    h: u64,
    hash_tables: &[HashMap<T, Vec<IdxSize>, H>],
    len: u64,
) -> &HashMap<T, Vec<IdxSize>, H> {
    let mut idx = 0;
    for i in 0..len {
        // can only be done for powers of two.
        // n % 2^i = n & (2^i - 1)
        if (h + i) & (len - 1) == 0 {
            idx = i as usize;
        }
    }
    hash_tables.get_unchecked(idx)
}

#[allow(clippy::type_complexity)]
unsafe fn get_hash_tbl_threaded_join_mut_partitioned<T, H>(
    h: u64,
    hash_tables: &mut [HashMap<T, (bool, Vec<IdxSize>), H>],
    len: u64,
) -> &mut HashMap<T, (bool, Vec<IdxSize>), H> {
    let mut idx = 0;
    for i in 0..len {
        // can only be done for powers of two.
        // n % 2^i = n & (2^i - 1)
        if (h + i) & (len - 1) == 0 {
            idx = i as usize;
        }
    }
    hash_tables.get_unchecked_mut(idx)
}

/// Probe the build table and add tuples to the results (inner join)
fn probe_inner<T, F>(
    probe: &[T],
    hash_tbls: &[PlHashMap<T, Vec<IdxSize>>],
    results: &mut Vec<(IdxSize, IdxSize)>,
    local_offset: usize,
    n_tables: u64,
    swap_fn: F,
) where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    F: Fn(IdxSize, IdxSize) -> (IdxSize, IdxSize),
{
    assert!(hash_tbls.len().is_power_of_two());
    probe.iter().enumerate().for_each(|(idx_a, k)| {
        let idx_a = (idx_a + local_offset) as IdxSize;
        // probe table that contains the hashed value
        let current_probe_table =
            unsafe { get_hash_tbl_threaded_join_partitioned(k.as_u64(), hash_tbls, n_tables) };

        let value = current_probe_table.get(k);

        if let Some(indexes_b) = value {
            let tuples = indexes_b.iter().map(|&idx_b| swap_fn(idx_a, idx_b));
            results.extend(tuples);
        }
    });
}

pub(crate) fn create_probe_table<T, IntoSlice>(
    keys: Vec<IntoSlice>,
) -> Vec<PlHashMap<T, Vec<IdxSize>>>
where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    IntoSlice: AsRef<[T]> + Send + Sync,
{
    let n_partitions = set_partition_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions).into_par_iter().map(|partition_no| {
            let partition_no = partition_no as u64;

            let mut hash_tbl: PlHashMap<T, Vec<IdxSize>> =
                PlHashMap::with_capacity(HASHMAP_INIT_SIZE);

            let n_partitions = n_partitions as u64;
            let mut offset = 0;
            for keys in &keys {
                let keys = keys.as_ref();
                let len = keys.len() as IdxSize;

                let mut cnt = 0;
                keys.iter().for_each(|k| {
                    let idx = cnt + offset;
                    cnt += 1;

                    if this_partition(k.as_u64(), partition_no, n_partitions) {
                        let entry = hash_tbl.entry(*k);

                        match entry {
                            Entry::Vacant(entry) => {
                                entry.insert(vec![idx]);
                            }
                            Entry::Occupied(mut entry) => {
                                let v = entry.get_mut();
                                v.push(idx);
                            }
                        }
                    }
                });
                offset += len;
            }
            hash_tbl
        })
    })
    .collect()
}

fn hash_join_tuples_inner<T, IntoSlice>(
    probe: Vec<IntoSlice>,
    build: Vec<IntoSlice>,
    // Because b should be the shorter relation we could need to swap to keep left left and right right.
    swap: bool,
) -> Vec<(IdxSize, IdxSize)>
where
    IntoSlice: AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    // NOTE: see the left join for more elaborate comments

    // first we hash one relation
    let hash_tbls = create_probe_table(build);

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());
    let offsets = probe
        .iter()
        .map(|ph| ph.as_ref().len())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect::<Vec<_>>();
    // next we probe the other relation
    // code duplication is because we want to only do the swap check once
    POOL.install(|| {
        probe
            .into_par_iter()
            .zip(offsets)
            .map(|(probe, offset)| {
                let probe = probe.as_ref();
                // local reference
                let hash_tbls = &hash_tbls;
                let mut results = Vec::with_capacity(probe.len());
                let local_offset = offset;

                // branch is to hoist swap out of the inner loop.
                if swap {
                    probe_inner(
                        probe,
                        hash_tbls,
                        &mut results,
                        local_offset,
                        n_tables,
                        |idx_a, idx_b| (idx_b, idx_a),
                    )
                } else {
                    probe_inner(
                        probe,
                        hash_tbls,
                        &mut results,
                        local_offset,
                        n_tables,
                        |idx_a, idx_b| (idx_a, idx_b),
                    )
                }

                results
            })
            .flatten()
            .collect()
    })
}

fn hash_join_tuples_left<T, IntoSlice>(
    probe: Vec<IntoSlice>,
    build: Vec<IntoSlice>,
) -> Vec<(IdxSize, Option<IdxSize>)>
where
    IntoSlice: AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    // first we hash one relation
    let hash_tbls = create_probe_table(build);

    // we determine the offset so that we later know which index to store in the join tuples
    let offsets = probe
        .iter()
        .map(|ph| ph.as_ref().len())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect::<Vec<_>>();

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());

    // next we probe the other relation
    POOL.install(|| {
        probe
            .into_par_iter()
            .zip(offsets)
            // probes_hashes: Vec<u64> processed by this thread
            // offset: offset index
            .map(|(probe, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;
                let probe = probe.as_ref();

                // assume the result tuples equal lenght of the no. of hashes processed by this thread.
                let mut results = Vec::with_capacity(probe.len());

                probe.iter().enumerate().for_each(|(idx_a, k)| {
                    let idx_a = (idx_a + offset) as IdxSize;
                    // probe table that contains the hashed value
                    let current_probe_table = unsafe {
                        get_hash_tbl_threaded_join_partitioned(k.as_u64(), hash_tbls, n_tables)
                    };

                    // we already hashed, so we don't have to hash again.
                    let value = current_probe_table.get(k);

                    match value {
                        // left and right matches
                        Some(indexes_b) => {
                            results.extend(indexes_b.iter().map(|&idx_b| (idx_a, Some(idx_b))))
                        }
                        // only left values, right = null
                        None => results.push((idx_a, None)),
                    }
                });
                results
            })
            .flatten()
            .collect()
    })
}

/// Probe the build table and add tuples to the results (inner join)
fn probe_outer<T, F, G, H>(
    probe_hashes: &[Vec<(u64, T)>],
    hash_tbls: &mut [PlHashMap<T, (bool, Vec<IdxSize>)>],
    results: &mut Vec<(Option<IdxSize>, Option<IdxSize>)>,
    n_tables: u64,
    // Function that get index_a, index_b when there is a match and pushes to result
    swap_fn_match: F,
    // Function that get index_a when there is no match and pushes to result
    swap_fn_no_match: G,
    // Function that get index_b from the build table that did not match any in A and pushes to result
    swap_fn_drain: H,
) where
    T: Send + Hash + Eq + Sync + Copy,
    // idx_a, idx_b -> ...
    F: Fn(IdxSize, IdxSize) -> (Option<IdxSize>, Option<IdxSize>),
    // idx_a -> ...
    G: Fn(IdxSize) -> (Option<IdxSize>, Option<IdxSize>),
    // idx_b -> ...
    H: Fn(IdxSize) -> (Option<IdxSize>, Option<IdxSize>),
{
    // needed for the partition shift instead of modulo to make sense
    assert!(n_tables.is_power_of_two());
    let mut idx_a = 0;
    for probe_hashes in probe_hashes {
        for (h, key) in probe_hashes {
            let h = *h;
            // probe table that contains the hashed value
            let current_probe_table =
                unsafe { get_hash_tbl_threaded_join_mut_partitioned(h, hash_tbls, n_tables) };

            let entry = current_probe_table
                .raw_entry_mut()
                .from_key_hashed_nocheck(h, key);

            match entry {
                // match and remove
                RawEntryMut::Occupied(mut occupied) => {
                    let (tracker, indexes_b) = occupied.get_mut();
                    *tracker = true;
                    results.extend(indexes_b.iter().map(|&idx_b| swap_fn_match(idx_a, idx_b)))
                }
                // no match
                RawEntryMut::Vacant(_) => results.push(swap_fn_no_match(idx_a)),
            }
            idx_a += 1;
        }
    }

    for hash_tbl in hash_tbls {
        hash_tbl.iter().for_each(|(_k, (tracker, indexes_b))| {
            // remaining joined values from the right table
            if !*tracker {
                results.extend(indexes_b.iter().map(|&idx_b| swap_fn_drain(idx_b)))
            }
        });
    }
}

/// Hash join outer. Both left and right can have no match so Options
fn hash_join_tuples_outer<T, I, J>(
    a: Vec<I>,
    b: Vec<J>,
    swap: bool,
) -> Vec<(Option<IdxSize>, Option<IdxSize>)>
where
    I: Iterator<Item = T> + Send + TrustedLen,
    J: Iterator<Item = T> + Send + TrustedLen,
    T: Hash + Eq + Copy + Sync + Send,
{
    // This function is partially multi-threaded.
    // Parts that are done in parallel:
    //  - creation of the probe tables
    //  - creation of the hashes

    // during the probe phase values are removed from the tables, that's done single threaded to
    // keep it lock free.

    let size = a.iter().map(|a| a.size_hint().0).sum::<usize>()
        + b.iter().map(|b| b.size_hint().0).sum::<usize>();
    let mut results = Vec::with_capacity(size);

    // prepare hash table
    let mut hash_tbls = prepare_hashed_relation_threaded(b);
    let random_state = hash_tbls[0].hasher().clone();

    // we pre hash the probing values
    let (probe_hashes, _) = create_hash_and_keys_threaded_vectorized(a, Some(random_state));

    let n_tables = hash_tbls.len() as u64;

    // probe the hash table.
    // Note: indexes from b that are not matched will be None, Some(idx_b)
    // Therefore we remove the matches and the remaining will be joined from the right

    // branch is because we want to only do the swap check once
    if swap {
        probe_outer(
            &probe_hashes,
            &mut hash_tbls,
            &mut results,
            n_tables,
            |idx_a, idx_b| (Some(idx_b), Some(idx_a)),
            |idx_a| (None, Some(idx_a)),
            |idx_b| (Some(idx_b), None),
        )
    } else {
        probe_outer(
            &probe_hashes,
            &mut hash_tbls,
            &mut results,
            n_tables,
            |idx_a, idx_b| (Some(idx_a), Some(idx_b)),
            |idx_a| (Some(idx_a), None),
            |idx_b| (None, Some(idx_b)),
        )
    }
    results
}

pub(crate) trait HashJoin<T> {
    fn hash_join_inner(&self, _other: &ChunkedArray<T>) -> Vec<(IdxSize, IdxSize)> {
        unimplemented!()
    }
    fn hash_join_left(&self, _other: &ChunkedArray<T>) -> Vec<(IdxSize, Option<IdxSize>)> {
        unimplemented!()
    }
    fn hash_join_outer(&self, _other: &ChunkedArray<T>) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        unimplemented!()
    }
}

impl HashJoin<Float32Type> for Float32Chunked {
    fn hash_join_inner(&self, other: &Float32Chunked) -> Vec<(IdxSize, IdxSize)> {
        let ca = self.bit_repr_small();
        let other = other.bit_repr_small();
        ca.hash_join_inner(&other)
    }
    fn hash_join_left(&self, other: &Float32Chunked) -> Vec<(IdxSize, Option<IdxSize>)> {
        let ca = self.bit_repr_small();
        let other = other.bit_repr_small();
        ca.hash_join_left(&other)
    }
    fn hash_join_outer(&self, other: &Float32Chunked) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        let ca = self.bit_repr_small();
        let other = other.bit_repr_small();
        ca.hash_join_outer(&other)
    }
}

impl HashJoin<Float64Type> for Float64Chunked {
    fn hash_join_inner(&self, other: &Float64Chunked) -> Vec<(IdxSize, IdxSize)> {
        let ca = self.bit_repr_large();
        let other = other.bit_repr_large();
        ca.hash_join_inner(&other)
    }
    fn hash_join_left(&self, other: &Float64Chunked) -> Vec<(IdxSize, Option<IdxSize>)> {
        let ca = self.bit_repr_large();
        let other = other.bit_repr_large();
        ca.hash_join_left(&other)
    }
    fn hash_join_outer(&self, other: &Float64Chunked) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        let ca = self.bit_repr_large();
        let other = other.bit_repr_large();
        ca.hash_join_outer(&other)
    }
}

fn num_group_join_inner<T>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
) -> Vec<(IdxSize, IdxSize)>
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq + Send + AsU64 + Copy,
    Option<T::Native>: AsU64,
{
    let n_threads = POOL.current_num_threads();
    let (a, b, swap) = det_hash_prone_order!(left, right);
    let splitted_a = split_ca(a, n_threads).unwrap();
    let splitted_b = split_ca(b, n_threads).unwrap();
    match (
        left.null_count() == 0,
        right.null_count() == 0,
        left.chunks.len(),
        right.chunks.len(),
    ) {
        (true, true, 1, 1) => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| ca.cont_slice().unwrap())
                .collect::<Vec<_>>();
            let keys_b = splitted_b
                .iter()
                .map(|ca| ca.cont_slice().unwrap())
                .collect::<Vec<_>>();
            hash_join_tuples_inner(keys_a, keys_b, swap)
        }
        (true, true, _, _) => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| ca.into_no_null_iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let keys_b = splitted_b
                .iter()
                .map(|ca| ca.into_no_null_iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();
            hash_join_tuples_inner(keys_a, keys_b, swap)
        }
        (_, _, 1, 1) => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| {
                    ca.downcast_iter()
                        .flat_map(|v| v.into_iter().map(|v| v.copied().as_u64()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let keys_b = splitted_b
                .iter()
                .map(|ca| {
                    ca.downcast_iter()
                        .flat_map(|v| v.into_iter().map(|v| v.copied().as_u64()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            hash_join_tuples_inner(keys_a, keys_b, swap)
        }
        _ => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| ca.into_iter().map(|v| v.as_u64()).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let keys_b = splitted_b
                .iter()
                .map(|ca| ca.into_iter().map(|v| v.as_u64()).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            hash_join_tuples_inner(keys_a, keys_b, swap)
        }
    }
}

fn num_group_join_left<T>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
) -> Vec<(IdxSize, Option<IdxSize>)>
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq + Send + AsU64,
    Option<T::Native>: AsU64,
{
    let n_threads = POOL.current_num_threads();
    let splitted_a = split_ca(left, n_threads).unwrap();
    let splitted_b = split_ca(right, n_threads).unwrap();
    match (
        left.null_count(),
        right.null_count(),
        left.chunks.len(),
        right.chunks.len(),
    ) {
        (0, 0, 1, 1) => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| ca.cont_slice().unwrap())
                .collect::<Vec<_>>();
            let keys_b = splitted_b
                .iter()
                .map(|ca| ca.cont_slice().unwrap())
                .collect::<Vec<_>>();
            hash_join_tuples_left(keys_a, keys_b)
        }
        (0, 0, _, _) => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| ca.into_no_null_iter().collect_trusted::<Vec<_>>())
                .collect::<Vec<_>>();
            let keys_b = splitted_b
                .iter()
                .map(|ca| ca.into_no_null_iter().collect_trusted::<Vec<_>>())
                .collect::<Vec<_>>();
            hash_join_tuples_left(keys_a, keys_b)
        }
        (_, _, 1, 1) => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| {
                    // we know that we only iterate over length == self.len()
                    unsafe {
                        ca.downcast_iter()
                            .flat_map(|v| v.into_iter().map(|v| v.copied().as_u64()))
                            .trust_my_length(ca.len())
                            .collect_trusted::<Vec<_>>()
                    }
                })
                .collect::<Vec<_>>();

            let keys_b = splitted_b
                .iter()
                .map(|ca| {
                    // we know that we only iterate over length == self.len()
                    unsafe {
                        ca.downcast_iter()
                            .flat_map(|v| v.into_iter().map(|v| v.copied().as_u64()))
                            .trust_my_length(ca.len())
                            .collect_trusted::<Vec<_>>()
                    }
                })
                .collect::<Vec<_>>();
            hash_join_tuples_left(keys_a, keys_b)
        }
        _ => {
            let keys_a = splitted_a
                .iter()
                .map(|ca| {
                    ca.into_iter()
                        .map(|v| v.as_u64())
                        .collect_trusted::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let keys_b = splitted_b
                .iter()
                .map(|ca| {
                    ca.into_iter()
                        .map(|v| v.as_u64())
                        .collect_trusted::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            hash_join_tuples_left(keys_a, keys_b)
        }
    }
}

impl<T> HashJoin<T> for ChunkedArray<T>
where
    T: PolarsIntegerType + Sync,
    T::Native: Eq + Hash + num::NumCast,
{
    fn hash_join_inner(&self, other: &ChunkedArray<T>) -> Vec<(IdxSize, IdxSize)> {
        match self.dtype() {
            DataType::UInt64 => {
                // convince the compiler that we are this type.
                let ca: &UInt64Chunked = unsafe {
                    &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt64Type>)
                };
                let other: &UInt64Chunked = unsafe {
                    &*(other as *const ChunkedArray<T> as *const ChunkedArray<UInt64Type>)
                };
                num_group_join_inner(ca, other)
            }
            DataType::UInt32 => {
                // convince the compiler that we are this type.
                let ca: &UInt32Chunked = unsafe {
                    &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt32Type>)
                };
                let other: &UInt32Chunked = unsafe {
                    &*(other as *const ChunkedArray<T> as *const ChunkedArray<UInt32Type>)
                };
                num_group_join_inner(ca, other)
            }
            DataType::Int64 | DataType::Float64 => {
                let ca = self.bit_repr_large();
                let other = other.bit_repr_large();
                num_group_join_inner(&ca, &other)
            }
            DataType::Int32 | DataType::Float32 => {
                let ca = self.bit_repr_small();
                let other = other.bit_repr_small();
                num_group_join_inner(&ca, &other)
            }
            _ => {
                let ca = self.cast(&DataType::UInt32).unwrap();
                let ca = ca.u32().unwrap();
                let other = other.cast(&DataType::UInt32).unwrap();
                let other = other.u32().unwrap();
                num_group_join_inner(ca, other)
            }
        }
    }

    fn hash_join_left(&self, other: &ChunkedArray<T>) -> Vec<(IdxSize, Option<IdxSize>)> {
        match self.dtype() {
            DataType::UInt64 => {
                // convince the compiler that we are this type.
                let ca: &UInt64Chunked = unsafe {
                    &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt64Type>)
                };
                let other: &UInt64Chunked = unsafe {
                    &*(other as *const ChunkedArray<T> as *const ChunkedArray<UInt64Type>)
                };
                num_group_join_left(ca, other)
            }
            DataType::UInt32 => {
                // convince the compiler that we are this type.
                let ca: &UInt32Chunked = unsafe {
                    &*(self as *const ChunkedArray<T> as *const ChunkedArray<UInt32Type>)
                };
                let other: &UInt32Chunked = unsafe {
                    &*(other as *const ChunkedArray<T> as *const ChunkedArray<UInt32Type>)
                };
                num_group_join_left(ca, other)
            }
            DataType::Int64 | DataType::Float64 => {
                let ca = self.bit_repr_large();
                let other = other.bit_repr_large();
                num_group_join_left(&ca, &other)
            }
            DataType::Int32 | DataType::Float32 => {
                let ca = self.bit_repr_small();
                let other = other.bit_repr_small();
                num_group_join_left(&ca, &other)
            }
            _ => {
                let ca = self.cast(&DataType::UInt32).unwrap();
                let ca = ca.u32().unwrap();
                let other = other.cast(&DataType::UInt32).unwrap();
                let other = other.u32().unwrap();
                num_group_join_left(ca, other)
            }
        }
    }

    fn hash_join_outer(&self, other: &ChunkedArray<T>) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        let (a, b, swap) = det_hash_prone_order!(self, other);

        let n_partitions = set_partition_size();
        let splitted_a = split_ca(a, n_partitions).unwrap();
        let splitted_b = split_ca(b, n_partitions).unwrap();

        match (a.null_count(), b.null_count()) {
            (0, 0) => {
                let iters_a = splitted_a
                    .iter()
                    .map(|ca| ca.into_no_null_iter())
                    .collect::<Vec<_>>();
                let iters_b = splitted_b
                    .iter()
                    .map(|ca| ca.into_no_null_iter())
                    .collect::<Vec<_>>();
                hash_join_tuples_outer(iters_a, iters_b, swap)
            }
            _ => {
                let iters_a = splitted_a
                    .iter()
                    .map(|ca| ca.into_iter())
                    .collect::<Vec<_>>();
                let iters_b = splitted_b
                    .iter()
                    .map(|ca| ca.into_iter())
                    .collect::<Vec<_>>();
                hash_join_tuples_outer(iters_a, iters_b, swap)
            }
        }
    }
}

impl HashJoin<BooleanType> for BooleanChunked {
    fn hash_join_inner(&self, other: &BooleanChunked) -> Vec<(IdxSize, IdxSize)> {
        let ca = self.cast(&DataType::UInt32).unwrap();
        let ca = ca.u32().unwrap();
        let other = other.cast(&DataType::UInt32).unwrap();
        let other = other.u32().unwrap();
        ca.hash_join_inner(other)
    }

    fn hash_join_left(&self, other: &BooleanChunked) -> Vec<(IdxSize, Option<IdxSize>)> {
        let ca = self.cast(&DataType::UInt32).unwrap();
        let ca = ca.u32().unwrap();
        let other = other.cast(&DataType::UInt32).unwrap();
        let other = other.u32().unwrap();
        ca.hash_join_left(other)
    }

    fn hash_join_outer(&self, other: &BooleanChunked) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        let (a, b, swap) = det_hash_prone_order!(self, other);

        let n_partitions = set_partition_size();
        let splitted_a = split_ca(a, n_partitions).unwrap();
        let splitted_b = split_ca(b, n_partitions).unwrap();

        match (a.null_count(), b.null_count()) {
            (0, 0) => {
                let iters_a = splitted_a
                    .iter()
                    .map(|ca| ca.into_no_null_iter())
                    .collect::<Vec<_>>();
                let iters_b = splitted_b
                    .iter()
                    .map(|ca| ca.into_no_null_iter())
                    .collect::<Vec<_>>();
                hash_join_tuples_outer(iters_a, iters_b, swap)
            }
            _ => {
                let iters_a = splitted_a
                    .iter()
                    .map(|ca| ca.into_iter())
                    .collect::<Vec<_>>();
                let iters_b = splitted_b
                    .iter()
                    .map(|ca| ca.into_iter())
                    .collect::<Vec<_>>();
                hash_join_tuples_outer(iters_a, iters_b, swap)
            }
        }
    }
}

pub(crate) fn prepare_strs<'a>(
    been_split: &'a [Utf8Chunked],
    hb: &RandomState,
) -> Vec<Vec<StrHash<'a>>> {
    POOL.install(|| {
        been_split
            .par_iter()
            .map(|ca| {
                ca.into_iter()
                    .map(|opt_s| {
                        let mut state = hb.build_hasher();
                        opt_s.hash(&mut state);
                        let hash = state.finish();
                        StrHash::new(opt_s, hash)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    })
}

impl HashJoin<Utf8Type> for Utf8Chunked {
    fn hash_join_inner(&self, other: &Utf8Chunked) -> Vec<(IdxSize, IdxSize)> {
        let n_threads = POOL.current_num_threads();

        let (a, b, swap) = det_hash_prone_order!(self, other);

        let hb = RandomState::default();
        let splitted_a = split_ca(a, n_threads).unwrap();
        let splitted_b = split_ca(b, n_threads).unwrap();

        let str_hashes_a = prepare_strs(&splitted_a, &hb);
        let str_hashes_b = prepare_strs(&splitted_b, &hb);
        hash_join_tuples_inner(str_hashes_a, str_hashes_b, swap)
    }

    fn hash_join_left(&self, other: &Utf8Chunked) -> Vec<(IdxSize, Option<IdxSize>)> {
        let n_threads = POOL.current_num_threads();

        let hb = RandomState::default();
        let splitted_a = split_ca(self, n_threads).unwrap();
        let splitted_b = split_ca(other, n_threads).unwrap();

        let str_hashes_a = prepare_strs(&splitted_a, &hb);
        let str_hashes_b = prepare_strs(&splitted_b, &hb);
        hash_join_tuples_left(str_hashes_a, str_hashes_b)
    }

    fn hash_join_outer(&self, other: &Utf8Chunked) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        let (a, b, swap) = det_hash_prone_order!(self, other);

        let n_partitions = set_partition_size();
        let splitted_a = split_ca(a, n_partitions).unwrap();
        let splitted_b = split_ca(b, n_partitions).unwrap();

        match (a.has_validity(), b.has_validity()) {
            (false, false) => {
                let iters_a = splitted_a
                    .iter()
                    .map(|ca| ca.into_no_null_iter())
                    .collect::<Vec<_>>();
                let iters_b = splitted_b
                    .iter()
                    .map(|ca| ca.into_no_null_iter())
                    .collect::<Vec<_>>();
                hash_join_tuples_outer(iters_a, iters_b, swap)
            }
            _ => {
                let iters_a = splitted_a
                    .iter()
                    .map(|ca| ca.into_iter())
                    .collect::<Vec<_>>();
                let iters_b = splitted_b
                    .iter()
                    .map(|ca| ca.into_iter())
                    .collect::<Vec<_>>();
                hash_join_tuples_outer(iters_a, iters_b, swap)
            }
        }
    }
}

pub trait ZipOuterJoinColumn {
    fn zip_outer_join_column(
        &self,
        _right_column: &Series,
        _opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
    ) -> Series {
        unimplemented!()
    }
}

impl<T> ZipOuterJoinColumn for ChunkedArray<T>
where
    T: PolarsIntegerType,
    ChunkedArray<T>: IntoSeries,
{
    fn zip_outer_join_column(
        &self,
        right_column: &Series,
        opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
    ) -> Series {
        let right_ca = self.unpack_series_matching_type(right_column).unwrap();

        let left_rand_access = self.take_rand();
        let right_rand_access = right_ca.take_rand();

        opt_join_tuples
            .iter()
            .map(|(opt_left_idx, opt_right_idx)| {
                if let Some(left_idx) = opt_left_idx {
                    unsafe { left_rand_access.get_unchecked(*left_idx as usize) }
                } else {
                    unsafe {
                        let right_idx = opt_right_idx.unwrap_unchecked();
                        right_rand_access.get_unchecked(right_idx as usize)
                    }
                }
            })
            .collect_trusted::<ChunkedArray<T>>()
            .into_series()
    }
}

macro_rules! impl_zip_outer_join {
    ($chunkedtype:ident) => {
        impl ZipOuterJoinColumn for $chunkedtype {
            fn zip_outer_join_column(
                &self,
                right_column: &Series,
                opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
            ) -> Series {
                let right_ca = self.unpack_series_matching_type(right_column).unwrap();

                let left_rand_access = self.take_rand();
                let right_rand_access = right_ca.take_rand();

                opt_join_tuples
                    .iter()
                    .map(|(opt_left_idx, opt_right_idx)| {
                        if let Some(left_idx) = opt_left_idx {
                            unsafe { left_rand_access.get_unchecked(*left_idx as usize) }
                        } else {
                            unsafe {
                                let right_idx = opt_right_idx.unwrap_unchecked();
                                right_rand_access.get_unchecked(right_idx as usize)
                            }
                        }
                    })
                    .collect::<$chunkedtype>()
                    .into_series()
            }
        }
    };
}
impl_zip_outer_join!(BooleanChunked);
impl_zip_outer_join!(Utf8Chunked);

impl ZipOuterJoinColumn for Float32Chunked {
    fn zip_outer_join_column(
        &self,
        right_column: &Series,
        opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
    ) -> Series {
        self.apply_as_ints(|s| {
            s.zip_outer_join_column(
                &right_column.bit_repr_small().into_series(),
                opt_join_tuples,
            )
        })
    }
}

impl ZipOuterJoinColumn for Float64Chunked {
    fn zip_outer_join_column(
        &self,
        right_column: &Series,
        opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
    ) -> Series {
        self.apply_as_ints(|s| {
            s.zip_outer_join_column(
                &right_column.bit_repr_large().into_series(),
                opt_join_tuples,
            )
        })
    }
}

impl DataFrame {
    /// Utility method to finish a join.
    pub(crate) fn finish_join(
        &self,
        mut df_left: DataFrame,
        mut df_right: DataFrame,
        suffix: Option<String>,
    ) -> Result<DataFrame> {
        let mut left_names = HashSet::with_capacity_and_hasher(df_left.width(), RandomState::new());

        df_left.columns.iter().for_each(|series| {
            left_names.insert(series.name());
        });

        let mut rename_strs = Vec::with_capacity(df_right.width());

        df_right.columns.iter().for_each(|series| {
            if left_names.contains(series.name()) {
                rename_strs.push(series.name().to_owned())
            }
        });
        let suffix = suffix.as_deref().unwrap_or("_right");

        for name in rename_strs {
            df_right.rename(&name, &format!("{}{}", name, suffix))?;
        }

        df_left.hstack_mut(&df_right.columns)?;
        Ok(df_left)
    }

    fn create_left_df<B: Sync>(&self, join_tuples: &[(IdxSize, B)], left_join: bool) -> DataFrame {
        if left_join && join_tuples.len() == self.height() {
            self.clone()
        } else {
            unsafe {
                self.take_iter_unchecked(join_tuples.iter().map(|(left, _right)| *left as usize))
            }
        }
    }

    fn join_impl(
        &self,
        other: &DataFrame,
        selected_left: Vec<Series>,
        selected_right: Vec<Series>,
        how: JoinType,
        suffix: Option<String>,
    ) -> Result<DataFrame> {
        if selected_right.len() != selected_left.len() {
            return Err(PolarsError::ValueError(
                "the number of columns given as join key should be equal".into(),
            ));
        }
        if selected_left
            .iter()
            .zip(&selected_right)
            .any(|(l, r)| l.dtype() != r.dtype())
        {
            return Err(PolarsError::ValueError("the dtype of the join keys don't match. first cast your columns to the correct dtype".into()));
        }

        #[cfg(feature = "dtype-categorical")]
        for (l, r) in selected_left.iter().zip(&selected_right) {
            check_categorical_src(l.dtype(), r.dtype())?
        }

        // Single keys
        if selected_left.len() == 1 {
            let s_left = self.column(selected_left[0].name())?;
            let s_right = other.column(selected_right[0].name())?;
            return match how {
                JoinType::Inner => self.inner_join_from_series(other, s_left, s_right, suffix),
                JoinType::Left => self.left_join_from_series(other, s_left, s_right, suffix),
                JoinType::Outer => self.outer_join_from_series(other, s_left, s_right, suffix),
                #[cfg(feature = "asof_join")]
                JoinType::AsOf(options) => {
                    let left_on = selected_left[0].name();
                    let right_on = selected_right[0].name();

                    match (options.left_by, options.right_by) {
                        (Some(left_by), Some(right_by)) => self.join_asof_by(
                            other,
                            left_on,
                            right_on,
                            left_by,
                            right_by,
                            options.strategy,
                        ),
                        (None, None) => self.join_asof(
                            other,
                            left_on,
                            right_on,
                            options.strategy,
                            options.tolerance,
                            suffix,
                        ),
                        _ => {
                            panic!("expected by arguments on both sides")
                        }
                    }
                }
                JoinType::Cross => {
                    unreachable!()
                }
            };
        }

        fn remove_selected(df: &DataFrame, selected: &[Series]) -> DataFrame {
            let mut new = None;
            for s in selected {
                new = match new {
                    None => Some(df.drop(s.name()).unwrap()),
                    Some(new) => Some(new.drop(s.name()).unwrap()),
                }
            }
            new.unwrap()
        }

        // hack for a macro
        impl DataFrame {
            fn len(&self) -> usize {
                self.height()
            }
        }
        // make sure that we don't have logical types.
        // we don't overwrite the original selected as that might be used to create a column in the new df
        let selected_left_physical = to_physical_and_bit_repr(&selected_left);
        let selected_right_physical = to_physical_and_bit_repr(&selected_right);

        // multiple keys
        match how {
            JoinType::Inner => {
                let left = DataFrame::new_no_checks(selected_left_physical);
                let right = DataFrame::new_no_checks(selected_right_physical);
                let (left, right, swap) = det_hash_prone_order!(left, right);
                let join_tuples = inner_join_multiple_keys(&left, &right, swap);

                let (df_left, df_right) = POOL.join(
                    || self.create_left_df(&join_tuples, false),
                    || unsafe {
                        // remove join columns
                        remove_selected(other, &selected_right).take_iter_unchecked(
                            join_tuples.iter().map(|(_left, right)| *right as usize),
                        )
                    },
                );
                self.finish_join(df_left, df_right, suffix)
            }
            JoinType::Left => {
                let left = DataFrame::new_no_checks(selected_left_physical);
                let right = DataFrame::new_no_checks(selected_right_physical);
                let join_tuples = left_join_multiple_keys(&left, &right);

                let (df_left, df_right) = POOL.join(
                    || self.create_left_df(&join_tuples, true),
                    || unsafe {
                        // remove join columns
                        remove_selected(other, &selected_right).take_opt_iter_unchecked(
                            join_tuples
                                .iter()
                                .map(|(_left, right)| right.map(|i| i as usize)),
                        )
                    },
                );
                self.finish_join(df_left, df_right, suffix)
            }
            JoinType::Outer => {
                let left = DataFrame::new_no_checks(selected_left_physical);
                let right = DataFrame::new_no_checks(selected_right_physical);

                let (left, right, swap) = det_hash_prone_order!(left, right);
                let opt_join_tuples = outer_join_multiple_keys(&left, &right, swap);

                // Take the left and right dataframes by join tuples
                let (mut df_left, df_right) = POOL.join(
                    || unsafe {
                        remove_selected(self, &selected_left).take_opt_iter_unchecked(
                            opt_join_tuples
                                .iter()
                                .map(|(left, _right)| left.map(|i| i as usize)),
                        )
                    },
                    || unsafe {
                        remove_selected(other, &selected_right).take_opt_iter_unchecked(
                            opt_join_tuples
                                .iter()
                                .map(|(_left, right)| right.map(|i| i as usize)),
                        )
                    },
                );
                for (s_left, s_right) in selected_left.iter().zip(&selected_right) {
                    let mut s = s_left.zip_outer_join_column(s_right, &opt_join_tuples);
                    s.rename(s_left.name());
                    df_left.with_column(s)?;
                }
                self.finish_join(df_left, df_right, suffix)
            }
            #[cfg(feature = "asof_join")]
            JoinType::AsOf(_) => Err(PolarsError::ValueError(
                "asof join not supported for join on multiple keys".into(),
            )),
            JoinType::Cross => {
                unreachable!()
            }
        }
    }

    /// Generic join method. Can be used to join on multiple columns.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Fruit" => &["Apple", "Banana", "Pear"],
    ///                          "Phosphorus (mg/100g)" => &[11, 22, 12])?;
    /// let df2: DataFrame = df!("Name" => &["Apple", "Banana", "Pear"],
    ///                          "Potassium (mg/100g)" => &[107, 358, 115])?;
    ///
    /// let df3: DataFrame = df1.join(&df2, ["Fruit"], ["Name"], JoinType::Inner, None)?;
    /// assert_eq!(df3.shape(), (3, 3));
    /// println!("{}", df3);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (3, 3)
    /// +--------+----------------------+---------------------+
    /// | Fruit  | Phosphorus (mg/100g) | Potassium (mg/100g) |
    /// | ---    | ---                  | ---                 |
    /// | str    | i32                  | i32                 |
    /// +========+======================+=====================+
    /// | Apple  | 11                   | 107                 |
    /// +--------+----------------------+---------------------+
    /// | Banana | 22                   | 358                 |
    /// +--------+----------------------+---------------------+
    /// | Pear   | 12                   | 115                 |
    /// +--------+----------------------+---------------------+
    /// ```
    pub fn join<I, S>(
        &self,
        other: &DataFrame,
        left_on: I,
        right_on: I,
        how: JoinType,
        suffix: Option<String>,
    ) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        #[cfg(feature = "cross_join")]
        if let JoinType::Cross = how {
            return self.cross_join(other);
        }

        #[allow(unused_mut)]
        let mut selected_left = self.select_series(left_on)?;
        #[allow(unused_mut)]
        let mut selected_right = other.select_series(right_on)?;
        self.join_impl(other, selected_left, selected_right, how, suffix)
    }

    /// Perform an inner join on two DataFrames.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn join_dfs(left: &DataFrame, right: &DataFrame) -> Result<DataFrame> {
    ///     left.inner_join(right, ["join_column_left"], ["join_column_right"])
    /// }
    /// ```
    pub fn inner_join<I, S>(&self, other: &DataFrame, left_on: I, right_on: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.join(other, left_on, right_on, JoinType::Inner, None)
    }

    pub(crate) fn inner_join_from_series(
        &self,
        other: &DataFrame,
        s_left: &Series,
        s_right: &Series,
        suffix: Option<String>,
    ) -> Result<DataFrame> {
        #[cfg(feature = "dtype-categorical")]
        check_categorical_src(s_left.dtype(), s_right.dtype())?;
        let join_tuples = s_left.hash_join_inner(s_right);

        let (df_left, df_right) = POOL.join(
            || self.create_left_df(&join_tuples, false),
            || unsafe {
                other
                    .drop(s_right.name())
                    .unwrap()
                    .take_iter_unchecked(join_tuples.iter().map(|(_left, right)| *right as usize))
            },
        );
        self.finish_join(df_left, df_right, suffix)
    }

    /// Perform a left join on two DataFrames
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Wavelength (nm)" => &[480.0, 650.0, 577.0, 1201.0, 100.0])?;
    /// let df2: DataFrame = df!("Color" => &["Blue", "Yellow", "Red"],
    ///                          "Wavelength nm" => &[480.0, 577.0, 650.0])?;
    ///
    /// let df3: DataFrame = df1.left_join(&df2, ["Wavelength (nm)"], ["Wavelength nm"])?;
    /// println!("{:?}", df3);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (5, 2)
    /// +-----------------+--------+
    /// | Wavelength (nm) | Color  |
    /// | ---             | ---    |
    /// | f64             | str    |
    /// +=================+========+
    /// | 480             | Blue   |
    /// +-----------------+--------+
    /// | 650             | Red    |
    /// +-----------------+--------+
    /// | 577             | Yellow |
    /// +-----------------+--------+
    /// | 1201            | null   |
    /// +-----------------+--------+
    /// | 100             | null   |
    /// +-----------------+--------+
    /// ```
    pub fn left_join<I, S>(&self, other: &DataFrame, left_on: I, right_on: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.join(other, left_on, right_on, JoinType::Left, None)
    }

    pub(crate) fn left_join_from_series(
        &self,
        other: &DataFrame,
        s_left: &Series,
        s_right: &Series,
        suffix: Option<String>,
    ) -> Result<DataFrame> {
        #[cfg(feature = "dtype-categorical")]
        check_categorical_src(s_left.dtype(), s_right.dtype())?;
        let opt_join_tuples = s_left.hash_join_left(s_right);

        let (df_left, df_right) = POOL.join(
            || self.create_left_df(&opt_join_tuples, true),
            || unsafe {
                other.drop(s_right.name()).unwrap().take_opt_iter_unchecked(
                    opt_join_tuples
                        .iter()
                        .map(|(_left, right)| right.map(|i| i as usize)),
                )
            },
        );
        self.finish_join(df_left, df_right, suffix)
    }

    /// Perform an outer join on two DataFrames
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn join_dfs(left: &DataFrame, right: &DataFrame) -> Result<DataFrame> {
    ///     left.outer_join(right, ["join_column_left"], ["join_column_right"])
    /// }
    /// ```
    pub fn outer_join<I, S>(&self, other: &DataFrame, left_on: I, right_on: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.join(other, left_on, right_on, JoinType::Outer, None)
    }
    pub(crate) fn outer_join_from_series(
        &self,
        other: &DataFrame,
        s_left: &Series,
        s_right: &Series,
        suffix: Option<String>,
    ) -> Result<DataFrame> {
        #[cfg(feature = "dtype-categorical")]
        check_categorical_src(s_left.dtype(), s_right.dtype())?;

        // store this so that we can keep original column order.
        let join_column_index = self.iter().position(|s| s.name() == s_left.name()).unwrap();

        // Get the indexes of the joined relations
        let opt_join_tuples = s_left.hash_join_outer(s_right);

        // Take the left and right dataframes by join tuples
        let (mut df_left, df_right) = POOL.join(
            || unsafe {
                self.drop(s_left.name()).unwrap().take_opt_iter_unchecked(
                    opt_join_tuples
                        .iter()
                        .map(|(left, _right)| left.map(|i| i as usize)),
                )
            },
            || unsafe {
                other.drop(s_right.name()).unwrap().take_opt_iter_unchecked(
                    opt_join_tuples
                        .iter()
                        .map(|(_left, right)| right.map(|i| i as usize)),
                )
            },
        );

        let mut s = s_left
            .to_physical_repr()
            .zip_outer_join_column(&s_right.to_physical_repr(), &opt_join_tuples);
        s.rename(s_left.name());
        let s = match s_left.dtype() {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                let ca_left = s_left.categorical().unwrap();
                let new_rev_map = ca_left.merge_categorical_map(s_right.categorical().unwrap());
                let logical = s.u32().unwrap().clone();
                CategoricalChunked::from_cats_and_rev_map(logical, new_rev_map).into_series()
            }
            dt @ DataType::Datetime(_, _)
            | dt @ DataType::Time
            | dt @ DataType::Date
            | dt @ DataType::Duration(_) => s.cast(dt).unwrap(),
            _ => s,
        };

        df_left.get_columns_mut().insert(join_column_index, s);
        self.finish_join(df_left, df_right, suffix)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::{df, POOL};

    fn create_frames() -> (DataFrame, DataFrame) {
        let s0 = Series::new("days", &[0, 1, 2]);
        let s1 = Series::new("temp", &[22.1, 19.9, 7.]);
        let s2 = Series::new("rain", &[0.2, 0.1, 0.3]);
        let temp = DataFrame::new(vec![s0, s1, s2]).unwrap();

        let s0 = Series::new("days", &[1, 2, 3, 1]);
        let s1 = Series::new("rain", &[0.1, 0.2, 0.3, 0.4]);
        let rain = DataFrame::new(vec![s0, s1]).unwrap();
        (temp, rain)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_inner_join() {
        let (temp, rain) = create_frames();

        for i in 1..8 {
            std::env::set_var("POLARS_MAX_THREADS", format!("{}", i));
            let joined = temp.inner_join(&rain, ["days"], ["days"]).unwrap();

            let join_col_days = Series::new("days", &[1, 2, 1]);
            let join_col_temp = Series::new("temp", &[19.9, 7., 19.9]);
            let join_col_rain = Series::new("rain", &[0.1, 0.3, 0.1]);
            let join_col_rain_right = Series::new("rain_right", [0.1, 0.2, 0.4].as_ref());
            let true_df = DataFrame::new(vec![
                join_col_days,
                join_col_temp,
                join_col_rain,
                join_col_rain_right,
            ])
            .unwrap();

            println!("{}", joined);
            assert!(joined.frame_equal(&true_df));
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    #[cfg_attr(miri, ignore)]
    fn test_left_join() {
        for i in 1..8 {
            std::env::set_var("POLARS_MAX_THREADS", format!("{}", i));
            let s0 = Series::new("days", &[0, 1, 2, 3, 4]);
            let s1 = Series::new("temp", &[22.1, 19.9, 7., 2., 3.]);
            let temp = DataFrame::new(vec![s0, s1]).unwrap();

            let s0 = Series::new("days", &[1, 2]);
            let s1 = Series::new("rain", &[0.1, 0.2]);
            let rain = DataFrame::new(vec![s0, s1]).unwrap();
            let joined = temp.left_join(&rain, ["days"], ["days"]).unwrap();
            println!("{}", &joined);
            assert_eq!(
                (joined.column("rain").unwrap().sum::<f32>().unwrap() * 10.).round(),
                3.
            );
            assert_eq!(joined.column("rain").unwrap().null_count(), 3);

            // test join on utf8
            let s0 = Series::new("days", &["mo", "tue", "wed", "thu", "fri"]);
            let s1 = Series::new("temp", &[22.1, 19.9, 7., 2., 3.]);
            let temp = DataFrame::new(vec![s0, s1]).unwrap();

            let s0 = Series::new("days", &["tue", "wed"]);
            let s1 = Series::new("rain", &[0.1, 0.2]);
            let rain = DataFrame::new(vec![s0, s1]).unwrap();
            let joined = temp.left_join(&rain, ["days"], ["days"]).unwrap();
            println!("{}", &joined);
            assert_eq!(
                (joined.column("rain").unwrap().sum::<f32>().unwrap() * 10.).round(),
                3.
            );
            assert_eq!(joined.column("rain").unwrap().null_count(), 3);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_outer_join() -> Result<()> {
        let (temp, rain) = create_frames();
        let joined = temp.outer_join(&rain, ["days"], ["days"])?;
        println!("{:?}", &joined);
        assert_eq!(joined.height(), 5);
        assert_eq!(joined.column("days")?.sum::<i32>(), Some(7));

        let df_left = df!(
                "a"=> ["a", "b", "a", "z"],
                "b"=>[1, 2, 3, 4],
                "c"=>[6, 5, 4, 3]
        )?;
        let df_right = df!(
                "a"=> ["b", "c", "b", "a"],
                "k"=> [0, 3, 9, 6],
                "c"=> [1, 0, 2, 1]
        )?;

        let out = df_left.outer_join(&df_right, ["a"], ["a"])?;
        assert_eq!(out.column("c_right")?.null_count(), 1);

        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_join_with_nulls() {
        let dts = &[20, 21, 22, 23, 24, 25, 27, 28];
        let vals = &[1.2, 2.4, 4.67, 5.8, 4.4, 3.6, 7.6, 6.5];
        let df = DataFrame::new(vec![Series::new("date", dts), Series::new("val", vals)]).unwrap();

        let vals2 = &[Some(1.1), None, Some(3.3), None, None];
        let df2 = DataFrame::new(vec![
            Series::new("date", &dts[3..]),
            Series::new("val2", vals2),
        ])
        .unwrap();

        let joined = df.left_join(&df2, ["date"], ["date"]).unwrap();
        assert_eq!(
            joined
                .column("val2")
                .unwrap()
                .f64()
                .unwrap()
                .get(joined.height() - 1),
            None
        );
    }

    fn get_dfs() -> (DataFrame, DataFrame) {
        let df_a = df! {
            "a" => &[1, 2, 1, 1],
            "b" => &["a", "b", "c", "c"],
            "c" => &[0, 1, 2, 3]
        }
        .unwrap();

        let df_b = df! {
            "foo" => &[1, 1, 1],
            "bar" => &["a", "c", "c"],
            "ham" => &["let", "var", "const"]
        }
        .unwrap();
        (df_a, df_b)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_join_multiple_columns() {
        let (mut df_a, mut df_b) = get_dfs();

        // First do a hack with concatenated string dummy column
        let mut s = df_a
            .column("a")
            .unwrap()
            .cast(&DataType::Utf8)
            .unwrap()
            .utf8()
            .unwrap()
            + df_a.column("b").unwrap().utf8().unwrap();
        s.rename("dummy");

        df_a.with_column(s).unwrap();
        let mut s = df_b
            .column("foo")
            .unwrap()
            .cast(&DataType::Utf8)
            .unwrap()
            .utf8()
            .unwrap()
            + df_b.column("bar").unwrap().utf8().unwrap();
        s.rename("dummy");
        df_b.with_column(s).unwrap();

        let joined = df_a.left_join(&df_b, ["dummy"], ["dummy"]).unwrap();
        let ham_col = joined.column("ham").unwrap();
        let ca = ham_col.utf8().unwrap();

        let correct_ham = &[
            Some("let"),
            None,
            Some("var"),
            Some("const"),
            Some("var"),
            Some("const"),
        ];

        assert_eq!(Vec::from(ca), correct_ham);

        // now check the join with multiple columns
        let joined = df_a
            .join(&df_b, ["a", "b"], ["foo", "bar"], JoinType::Left, None)
            .unwrap();
        let ca = joined.column("ham").unwrap().utf8().unwrap();
        dbg!(&df_a, &df_b);
        assert_eq!(Vec::from(ca), correct_ham);
        let joined_inner_hack = df_a.inner_join(&df_b, ["dummy"], ["dummy"]).unwrap();
        let joined_inner = df_a
            .join(&df_b, ["a", "b"], ["foo", "bar"], JoinType::Inner, None)
            .unwrap();

        dbg!(&joined_inner_hack, &joined_inner);
        assert!(joined_inner_hack
            .column("ham")
            .unwrap()
            .series_equal_missing(joined_inner.column("ham").unwrap()));

        let joined_outer_hack = df_a.outer_join(&df_b, ["dummy"], ["dummy"]).unwrap();
        let joined_outer = df_a
            .join(&df_b, ["a", "b"], ["foo", "bar"], JoinType::Outer, None)
            .unwrap();
        assert!(joined_outer_hack
            .column("ham")
            .unwrap()
            .series_equal_missing(joined_outer.column("ham").unwrap()));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[cfg(feature = "dtype-categorical")]
    fn test_join_categorical() {
        use crate::toggle_string_cache;
        let _lock = crate::SINGLE_LOCK.lock();
        toggle_string_cache(true);

        let (mut df_a, mut df_b) = get_dfs();

        df_a.try_apply("b", |s| s.cast(&DataType::Categorical(None)))
            .unwrap();
        df_b.try_apply("bar", |s| s.cast(&DataType::Categorical(None)))
            .unwrap();

        let out = df_a
            .join(&df_b, ["b"], ["bar"], JoinType::Left, None)
            .unwrap();
        assert_eq!(out.shape(), (6, 5));
        let correct_ham = &[
            Some("let"),
            None,
            Some("var"),
            Some("const"),
            Some("var"),
            Some("const"),
        ];
        let ham_col = out.column("ham").unwrap();
        let ca = ham_col.utf8().unwrap();

        assert_eq!(Vec::from(ca), correct_ham);

        // test dispatch
        for jt in [JoinType::Left, JoinType::Inner, JoinType::Outer] {
            let out = df_a.join(&df_b, ["b"], ["bar"], jt, None).unwrap();
            let out = out.column("b").unwrap();
            assert_eq!(out.dtype(), &DataType::Categorical(None));
        }

        // Test error when joining on different string cache
        let (mut df_a, mut df_b) = get_dfs();
        df_a.try_apply("b", |s| s.cast(&DataType::Categorical(None)))
            .unwrap();
        // create a new cache
        toggle_string_cache(false);
        toggle_string_cache(true);

        df_b.try_apply("bar", |s| s.cast(&DataType::Categorical(None)))
            .unwrap();
        let out = df_a.join(&df_b, ["b"], ["bar"], JoinType::Left, None);
        assert!(out.is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn empty_df_join() -> Result<()> {
        let empty: Vec<String> = vec![];
        let empty_df = DataFrame::new(vec![
            Series::new("key", &empty),
            Series::new("eval", &empty),
        ])
        .unwrap();

        let df = DataFrame::new(vec![
            Series::new("key", &["foo"]),
            Series::new("aval", &[4]),
        ])
        .unwrap();

        let out = empty_df.inner_join(&df, ["key"], ["key"]).unwrap();
        assert_eq!(out.height(), 0);
        let out = empty_df.left_join(&df, ["key"], ["key"]).unwrap();
        assert_eq!(out.height(), 0);
        let out = empty_df.outer_join(&df, ["key"], ["key"]).unwrap();
        assert_eq!(out.height(), 1);
        df.left_join(&empty_df, ["key"], ["key"])?;
        df.inner_join(&empty_df, ["key"], ["key"])?;
        df.outer_join(&empty_df, ["key"], ["key"])?;

        let empty: Vec<String> = vec![];
        let _empty_df = DataFrame::new(vec![
            Series::new("key", &empty),
            Series::new("eval", &empty),
        ])
        .unwrap();

        let df = df![
            "key" => [1i32, 2],
            "vals" => [1, 2],
        ]?;

        // https://github.com/pola-rs/polars/issues/1824
        let empty: Vec<i32> = vec![];
        let empty_df = DataFrame::new(vec![
            Series::new("key", &empty),
            Series::new("1val", &empty),
            Series::new("2val", &empty),
        ])?;

        let out = df.left_join(&empty_df, ["key"], ["key"])?;
        assert_eq!(out.shape(), (2, 4));

        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn unit_df_join() -> Result<()> {
        let df1 = df![
            "a" => [1],
            "b" => [2]
        ]?;

        let df2 = df![
            "a" => [1, 2, 3, 4],
            "b" => [Some(1), None, Some(3), Some(4)]
        ]?;

        let out = df1.left_join(&df2, ["a"], ["a"])?;
        let expected = df![
            "a" => [1],
            "b" => [2],
            "b_right" => [1]
        ]?;
        assert!(out.frame_equal(&expected));
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_join_err() -> Result<()> {
        let df1 = df![
            "a" => [1, 2],
            "b" => ["foo", "bar"]
        ]?;

        let df2 = df![
            "a" => [1, 2, 3, 4],
            "b" => [true, true, true, false]
        ]?;

        // dtypes don't match, error
        assert!(df1
            .join(&df2, vec!["a", "b"], vec!["a", "b"], JoinType::Left, None)
            .is_err());
        // length of join keys don't match error
        assert!(df1
            .join(&df2, vec!["a"], vec!["a", "b"], JoinType::Left, None)
            .is_err());
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_joins_with_duplicates() -> Result<()> {
        // test joins with duplicates in both dataframes

        let df_left = df![
            "col1" => [1, 1, 2],
            "int_col" => [1, 2, 3]
        ]
        .unwrap();

        let df_right = df![
            "join_col1" => [1, 1, 1, 1, 1, 3],
            "dbl_col" => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        ]
        .unwrap();

        let df_inner_join = df_left
            .inner_join(&df_right, ["col1"], ["join_col1"])
            .unwrap();

        assert_eq!(df_inner_join.height(), 10);
        assert_eq!(df_inner_join.column("col1")?.null_count(), 0);
        assert_eq!(df_inner_join.column("int_col")?.null_count(), 0);
        assert_eq!(df_inner_join.column("dbl_col")?.null_count(), 0);

        let df_left_join = df_left
            .left_join(&df_right, ["col1"], ["join_col1"])
            .unwrap();

        assert_eq!(df_left_join.height(), 11);
        assert_eq!(df_left_join.column("col1")?.null_count(), 0);
        assert_eq!(df_left_join.column("int_col")?.null_count(), 0);
        assert_eq!(df_left_join.column("dbl_col")?.null_count(), 1);

        let df_outer_join = df_left
            .outer_join(&df_right, ["col1"], ["join_col1"])
            .unwrap();

        // ensure the column names don't get swapped by the drop we do
        assert_eq!(
            df_outer_join.get_column_names(),
            &["col1", "int_col", "dbl_col"]
        );
        assert_eq!(df_outer_join.height(), 12);
        assert_eq!(df_outer_join.column("col1")?.null_count(), 0);
        assert_eq!(df_outer_join.column("int_col")?.null_count(), 1);
        assert_eq!(df_outer_join.column("dbl_col")?.null_count(), 1);

        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_multi_joins_with_duplicates() -> Result<()> {
        // test joins with multiple join columns and duplicates in both
        // dataframes

        let df_left = df![
            "col1" => [1, 1, 1],
            "join_col2" => ["a", "a", "b"],
            "int_col" => [1, 2, 3]
        ]
        .unwrap();

        let df_right = df![
            "join_col1" => [1, 1, 1, 1, 1, 2],
            "col2" => ["a", "a", "a", "a", "a", "c"],
            "dbl_col" => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        ]
        .unwrap();

        let df_inner_join = df_left
            .join(
                &df_right,
                &["col1", "join_col2"],
                &["join_col1", "col2"],
                JoinType::Inner,
                None,
            )
            .unwrap();

        assert_eq!(df_inner_join.height(), 10);
        assert_eq!(df_inner_join.column("col1")?.null_count(), 0);
        assert_eq!(df_inner_join.column("join_col2")?.null_count(), 0);
        assert_eq!(df_inner_join.column("int_col")?.null_count(), 0);
        assert_eq!(df_inner_join.column("dbl_col")?.null_count(), 0);

        let df_left_join = df_left
            .join(
                &df_right,
                &["col1", "join_col2"],
                &["join_col1", "col2"],
                JoinType::Left,
                None,
            )
            .unwrap();

        assert_eq!(df_left_join.height(), 11);
        assert_eq!(df_left_join.column("col1")?.null_count(), 0);
        assert_eq!(df_left_join.column("join_col2")?.null_count(), 0);
        assert_eq!(df_left_join.column("int_col")?.null_count(), 0);
        assert_eq!(df_left_join.column("dbl_col")?.null_count(), 1);

        let df_outer_join = df_left
            .join(
                &df_right,
                &["col1", "join_col2"],
                &["join_col1", "col2"],
                JoinType::Outer,
                None,
            )
            .unwrap();

        assert_eq!(df_outer_join.height(), 12);
        assert_eq!(df_outer_join.column("col1")?.null_count(), 0);
        assert_eq!(df_outer_join.column("join_col2")?.null_count(), 0);
        assert_eq!(df_outer_join.column("int_col")?.null_count(), 1);
        assert_eq!(df_outer_join.column("dbl_col")?.null_count(), 1);

        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_join_floats() -> Result<()> {
        let df_a = df! {
            "a" => &[1.0, 2.0, 1.0, 1.0],
            "b" => &["a", "b", "c", "c"],
            "c" => &[0.0, 1.0, 2.0, 3.0]
        }?;

        let df_b = df! {
            "foo" => &[1.0, 2.0, 1.0],
            "bar" => &[1.0, 1.0, 1.0],
            "ham" => &["let", "var", "const"]
        }?;

        let out = df_a.join(
            &df_b,
            vec!["a", "c"],
            vec!["foo", "bar"],
            JoinType::Left,
            None,
        )?;
        assert_eq!(
            Vec::from(out.column("ham")?.utf8()?),
            &[None, Some("var"), None, None]
        );

        let out = df_a.join(
            &df_b,
            vec!["a", "c"],
            vec!["foo", "bar"],
            JoinType::Outer,
            None,
        )?;
        assert_eq!(
            out.dtypes(),
            &[
                DataType::Utf8,
                DataType::Float64,
                DataType::Float64,
                DataType::Utf8
            ]
        );
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_join_nulls() -> Result<()> {
        let a = df![
            "a" => [Some(1), None, None]
        ]?;
        let b = df![
            "a" => [Some(1), None, None, None, None]
        ]?;

        let out = a.inner_join(&b, ["a"], ["a"])?;

        assert_eq!(out.shape(), (9, 1));
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_4_threads_bit_offset() -> Result<()> {
        // run this locally with a thread pool size of 4
        // this was an obscure bug caused by not taking the offset of a bit into account.
        let n = 8i64;
        let mut left_a = (0..n).map(Some).collect::<Int64Chunked>();
        let mut left_b = (0..n)
            .map(|i| if i % 2 == 0 { None } else { Some(0) })
            .collect::<Int64Chunked>();
        left_a.rename("a");
        left_b.rename("b");
        let left_df = DataFrame::new(vec![left_a.into_series(), left_b.into_series()])?;

        let i = 1;
        let len = 8;
        let range = i..i + len;
        let mut right_a = range.clone().map(Some).collect::<Int64Chunked>();
        let mut right_b = range
            .map(|i| if i % 3 == 0 { None } else { Some(1) })
            .collect::<Int64Chunked>();
        right_a.rename("a");
        right_b.rename("b");

        let right_df = DataFrame::new(vec![right_a.into_series(), right_b.into_series()])?;
        let out = left_df.join(&right_df, ["a", "b"], ["a", "b"], JoinType::Inner, None)?;
        assert_eq!(out.shape(), (1, 2));
        Ok(())
    }
}
