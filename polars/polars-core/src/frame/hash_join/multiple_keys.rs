use hashbrown::hash_map::RawEntryMut;
use hashbrown::HashMap;
use rayon::prelude::*;

use super::*;
use crate::frame::groupby::hashing::{populate_multiple_key_hashmap, HASHMAP_INIT_SIZE};
use crate::frame::hash_join::{
    get_hash_tbl_threaded_join_mut_partitioned, get_hash_tbl_threaded_join_partitioned,
};
use crate::prelude::*;
use crate::utils::series::_to_physical_and_bit_repr;
use crate::utils::{_set_partition_size, split_df};
use crate::vector_hasher::{df_rows_to_hashes_threaded, this_partition, IdBuildHasher, IdxHash};
use crate::POOL;

/// Compare the rows of two DataFrames
pub(crate) unsafe fn compare_df_rows2(
    left: &DataFrame,
    right: &DataFrame,
    left_idx: usize,
    right_idx: usize,
) -> bool {
    for (l, r) in left.get_columns().iter().zip(right.get_columns()) {
        if !(l.get_unchecked(left_idx) == r.get_unchecked(right_idx)) {
            return false;
        }
    }
    true
}

pub(crate) fn create_probe_table(
    hashes: &[UInt64Chunked],
    keys: &DataFrame,
) -> Vec<HashMap<IdxHash, Vec<IdxSize>, IdBuildHasher>> {
    let n_partitions = _set_partition_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions)
            .into_par_iter()
            .map(|part_no| {
                let part_no = part_no as u64;
                let mut hash_tbl: HashMap<IdxHash, Vec<IdxSize>, IdBuildHasher> =
                    HashMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());

                let n_partitions = n_partitions as u64;
                let mut offset = 0;
                for hashes in hashes {
                    for hashes in hashes.data_views() {
                        let len = hashes.len();
                        let mut idx = 0;
                        hashes.iter().for_each(|h| {
                            // partition hashes by thread no.
                            // So only a part of the hashes go to this hashmap
                            if this_partition(*h, part_no, n_partitions) {
                                let idx = idx + offset;
                                populate_multiple_key_hashmap(
                                    &mut hash_tbl,
                                    idx,
                                    *h,
                                    keys,
                                    || vec![idx],
                                    |v| v.push(idx),
                                )
                            }
                            idx += 1;
                        });

                        offset += len as IdxSize;
                    }
                }
                hash_tbl
            })
            .collect()
    })
}

fn create_build_table_outer(
    hashes: &[UInt64Chunked],
    keys: &DataFrame,
) -> Vec<HashMap<IdxHash, (bool, Vec<IdxSize>), IdBuildHasher>> {
    // Outer join equivalent of create_build_table() adds a bool in the hashmap values for tracking
    // whether a value in the hash table has already been matched to a value in the probe hashes.
    let n_partitions = _set_partition_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions).into_par_iter().map(|part_no| {
            let part_no = part_no as u64;
            let mut hash_tbl: HashMap<IdxHash, (bool, Vec<IdxSize>), IdBuildHasher> =
                HashMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());

            let n_partitions = n_partitions as u64;
            let mut offset = 0;
            for hashes in hashes {
                for hashes in hashes.data_views() {
                    let len = hashes.len();
                    let mut idx = 0;
                    hashes.iter().for_each(|h| {
                        // partition hashes by thread no.
                        // So only a part of the hashes go to this hashmap
                        if this_partition(*h, part_no, n_partitions) {
                            let idx = idx + offset;
                            populate_multiple_key_hashmap(
                                &mut hash_tbl,
                                idx,
                                *h,
                                keys,
                                || (false, vec![idx]),
                                |v| v.1.push(idx),
                            )
                        }
                        idx += 1;
                    });

                    offset += len as IdxSize;
                }
            }
            hash_tbl
        })
    })
    .collect()
}

/// Probe the build table and add tuples to the results (inner join)
#[allow(clippy::too_many_arguments)]
fn probe_inner<F>(
    probe_hashes: &UInt64Chunked,
    hash_tbls: &[HashMap<IdxHash, Vec<IdxSize>, IdBuildHasher>],
    results: &mut Vec<(IdxSize, IdxSize)>,
    local_offset: usize,
    n_tables: u64,
    a: &DataFrame,
    b: &DataFrame,
    swap_fn: F,
) where
    F: Fn(IdxSize, IdxSize) -> (IdxSize, IdxSize),
{
    let mut idx_a = local_offset as IdxSize;
    for probe_hashes in probe_hashes.data_views() {
        for &h in probe_hashes {
            // probe table that contains the hashed value
            let current_probe_table =
                unsafe { get_hash_tbl_threaded_join_partitioned(h, hash_tbls, n_tables) };

            let entry = current_probe_table.raw_entry().from_hash(h, |idx_hash| {
                let idx_b = idx_hash.idx;
                // Safety:
                // indices in a join operation are always in bounds.
                unsafe { compare_df_rows2(a, b, idx_a as usize, idx_b as usize) }
            });

            if let Some((_, indexes_b)) = entry {
                let tuples = indexes_b.iter().map(|&idx_b| swap_fn(idx_a, idx_b));
                results.extend(tuples);
            }
            idx_a += 1;
        }
    }
}

pub(crate) fn get_offsets(probe_hashes: &[UInt64Chunked]) -> Vec<usize> {
    probe_hashes
        .iter()
        .map(|ph| ph.len())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect()
}

pub fn _inner_join_multiple_keys(
    a: &mut DataFrame,
    b: &mut DataFrame,
    swap: bool,
) -> (Vec<IdxSize>, Vec<IdxSize>) {
    // we assume that the b DataFrame is the shorter relation.
    // b will be used for the build phase.

    let n_threads = POOL.current_num_threads();
    let dfs_a = split_df(a, n_threads).unwrap();
    let dfs_b = split_df(b, n_threads).unwrap();

    let (build_hashes, random_state) = df_rows_to_hashes_threaded(&dfs_b, None).unwrap();
    let (probe_hashes, _) = df_rows_to_hashes_threaded(&dfs_a, Some(random_state)).unwrap();

    let hash_tbls = create_probe_table(&build_hashes, b);
    // early drop to reduce memory pressure
    drop(build_hashes);

    let n_tables = hash_tbls.len() as u64;
    let offsets = get_offsets(&probe_hashes);
    // next we probe the other relation
    // code duplication is because we want to only do the swap check once
    POOL.install(|| {
        probe_hashes
            .into_par_iter()
            .zip(offsets)
            .flat_map(|(probe_hashes, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;
                let mut results =
                    Vec::with_capacity(probe_hashes.len() / POOL.current_num_threads());
                let local_offset = offset;
                // code duplication is to hoist swap out of the inner loop.
                if swap {
                    probe_inner(
                        &probe_hashes,
                        hash_tbls,
                        &mut results,
                        local_offset,
                        n_tables,
                        a,
                        b,
                        |idx_a, idx_b| (idx_b, idx_a),
                    )
                } else {
                    probe_inner(
                        &probe_hashes,
                        hash_tbls,
                        &mut results,
                        local_offset,
                        n_tables,
                        a,
                        b,
                        |idx_a, idx_b| (idx_a, idx_b),
                    )
                }

                results
            })
            .unzip()
    })
}

#[cfg(feature = "private")]
pub fn private_left_join_multiple_keys(
    a: &DataFrame,
    b: &DataFrame,
    // map the global indices to [chunk_idx, array_idx]
    // only needed if we have non contiguous memory
    chunk_mapping_left: Option<&[ChunkId]>,
    chunk_mapping_right: Option<&[ChunkId]>,
) -> LeftJoinIds {
    let mut a = DataFrame::new_no_checks(_to_physical_and_bit_repr(a.get_columns()));
    let mut b = DataFrame::new_no_checks(_to_physical_and_bit_repr(b.get_columns()));
    _left_join_multiple_keys(&mut a, &mut b, chunk_mapping_left, chunk_mapping_right)
}

pub fn _left_join_multiple_keys(
    a: &mut DataFrame,
    b: &mut DataFrame,
    // map the global indices to [chunk_idx, array_idx]
    // only needed if we have non contiguous memory
    chunk_mapping_left: Option<&[ChunkId]>,
    chunk_mapping_right: Option<&[ChunkId]>,
) -> LeftJoinIds {
    // we should not join on logical types
    debug_assert!(!a.iter().any(|s| s.dtype().is_logical()));
    debug_assert!(!b.iter().any(|s| s.dtype().is_logical()));

    let n_threads = POOL.current_num_threads();
    let dfs_a = split_df(a, n_threads).unwrap();
    let dfs_b = split_df(b, n_threads).unwrap();

    let (build_hashes, random_state) = df_rows_to_hashes_threaded(&dfs_b, None).unwrap();
    let (probe_hashes, _) = df_rows_to_hashes_threaded(&dfs_a, Some(random_state)).unwrap();

    let hash_tbls = create_probe_table(&build_hashes, b);
    // early drop to reduce memory pressure
    drop(build_hashes);

    let n_tables = hash_tbls.len() as u64;
    let offsets = get_offsets(&probe_hashes);

    // next we probe the other relation
    // code duplication is because we want to only do the swap check once
    let results = POOL.install(move || {
        probe_hashes
            .into_par_iter()
            .zip(offsets)
            .map(move |(probe_hashes, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;

                let len = probe_hashes.len() / POOL.current_num_threads();
                let mut result_idx_left = Vec::with_capacity(len);
                let mut result_idx_right = Vec::with_capacity(len);
                let local_offset = offset;

                let mut idx_a = local_offset as IdxSize;
                for probe_hashes in probe_hashes.data_views() {
                    for &h in probe_hashes {
                        // probe table that contains the hashed value
                        let current_probe_table = unsafe {
                            get_hash_tbl_threaded_join_partitioned(h, hash_tbls, n_tables)
                        };

                        let entry = current_probe_table.raw_entry().from_hash(h, |idx_hash| {
                            let idx_b = idx_hash.idx;
                            // Safety:
                            // indices in a join operation are always in bounds.
                            unsafe { compare_df_rows2(a, b, idx_a as usize, idx_b as usize) }
                        });

                        match entry {
                            // left and right matches
                            Some((_, indexes_b)) => {
                                result_idx_left
                                    .extend(std::iter::repeat(idx_a).take(indexes_b.len()));
                                result_idx_right.extend(indexes_b.iter().copied().map(Some))
                            }
                            // only left values, right = null
                            None => {
                                result_idx_left.push(idx_a);
                                result_idx_right.push(None);
                            }
                        }
                        idx_a += 1;
                    }
                }

                finish_left_join_mappings(
                    result_idx_left,
                    result_idx_right,
                    chunk_mapping_left,
                    chunk_mapping_right,
                )
            })
            .collect::<Vec<_>>()
    });
    flatten_left_join_ids(results)
}

#[cfg(feature = "semi_anti_join")]
pub(crate) fn create_build_table_semi_anti(
    hashes: &[UInt64Chunked],
    keys: &DataFrame,
) -> Vec<HashMap<IdxHash, (), IdBuildHasher>> {
    let n_partitions = _set_partition_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions).into_par_iter().map(|part_no| {
            let part_no = part_no as u64;
            let mut hash_tbl: HashMap<IdxHash, (), IdBuildHasher> =
                HashMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());

            let n_partitions = n_partitions as u64;
            let mut offset = 0;
            for hashes in hashes {
                for hashes in hashes.data_views() {
                    let len = hashes.len();
                    let mut idx = 0;
                    hashes.iter().for_each(|h| {
                        // partition hashes by thread no.
                        // So only a part of the hashes go to this hashmap
                        if this_partition(*h, part_no, n_partitions) {
                            let idx = idx + offset;
                            populate_multiple_key_hashmap(
                                &mut hash_tbl,
                                idx,
                                *h,
                                keys,
                                || (),
                                |_| (),
                            )
                        }
                        idx += 1;
                    });

                    offset += len as IdxSize;
                }
            }
            hash_tbl
        })
    })
    .collect()
}

#[cfg(feature = "semi_anti_join")]
pub(crate) fn semi_anti_join_multiple_keys_impl<'a>(
    a: &'a mut DataFrame,
    b: &'a mut DataFrame,
) -> impl ParallelIterator<Item = (IdxSize, bool)> + 'a {
    // we should not join on logical types
    debug_assert!(!a.iter().any(|s| s.dtype().is_logical()));
    debug_assert!(!b.iter().any(|s| s.dtype().is_logical()));

    let n_threads = POOL.current_num_threads();
    let dfs_a = split_df(a, n_threads).unwrap();
    let dfs_b = split_df(b, n_threads).unwrap();

    let (build_hashes, random_state) = df_rows_to_hashes_threaded(&dfs_b, None).unwrap();
    let (probe_hashes, _) = df_rows_to_hashes_threaded(&dfs_a, Some(random_state)).unwrap();

    let hash_tbls = create_build_table_semi_anti(&build_hashes, b);
    // early drop to reduce memory pressure
    drop(build_hashes);

    let n_tables = hash_tbls.len() as u64;
    let offsets = get_offsets(&probe_hashes);

    // next we probe the other relation
    // code duplication is because we want to only do the swap check once
    POOL.install(move || {
        probe_hashes
            .into_par_iter()
            .zip(offsets)
            .flat_map(move |(probe_hashes, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;
                let mut results =
                    Vec::with_capacity(probe_hashes.len() / POOL.current_num_threads());
                let local_offset = offset;

                let mut idx_a = local_offset as IdxSize;
                for probe_hashes in probe_hashes.data_views() {
                    for &h in probe_hashes {
                        // probe table that contains the hashed value
                        let current_probe_table = unsafe {
                            get_hash_tbl_threaded_join_partitioned(h, hash_tbls, n_tables)
                        };

                        let entry = current_probe_table.raw_entry().from_hash(h, |idx_hash| {
                            let idx_b = idx_hash.idx;
                            // Safety:
                            // indices in a join operation are always in bounds.
                            unsafe { compare_df_rows2(a, b, idx_a as usize, idx_b as usize) }
                        });

                        match entry {
                            // left and right matches
                            Some((_, _)) => results.push((idx_a, true)),
                            // only left values, right = null
                            None => results.push((idx_a, false)),
                        }
                        idx_a += 1;
                    }
                }

                results
            })
    })
}

#[cfg(feature = "semi_anti_join")]
pub fn _left_anti_multiple_keys(a: &mut DataFrame, b: &mut DataFrame) -> Vec<IdxSize> {
    semi_anti_join_multiple_keys_impl(a, b)
        .filter(|tpls| !tpls.1)
        .map(|tpls| tpls.0)
        .collect()
}

#[cfg(feature = "semi_anti_join")]
pub fn _left_semi_multiple_keys(a: &mut DataFrame, b: &mut DataFrame) -> Vec<IdxSize> {
    semi_anti_join_multiple_keys_impl(a, b)
        .filter(|tpls| tpls.1)
        .map(|tpls| tpls.0)
        .collect()
}

/// Probe the build table and add tuples to the results (inner join)
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn probe_outer<F, G, H>(
    probe_hashes: &[UInt64Chunked],
    hash_tbls: &mut [HashMap<IdxHash, (bool, Vec<IdxSize>), IdBuildHasher>],
    results: &mut Vec<(Option<IdxSize>, Option<IdxSize>)>,
    n_tables: u64,
    a: &DataFrame,
    b: &DataFrame,
    // Function that get index_a, index_b when there is a match and pushes to result
    swap_fn_match: F,
    // Function that get index_a when there is no match and pushes to result
    swap_fn_no_match: G,
    // Function that get index_b from the build table that did not match any in A and pushes to result
    swap_fn_drain: H,
) where
    // idx_a, idx_b -> ...
    F: Fn(IdxSize, IdxSize) -> (Option<IdxSize>, Option<IdxSize>),
    // idx_a -> ...
    G: Fn(IdxSize) -> (Option<IdxSize>, Option<IdxSize>),
    // idx_b -> ...
    H: Fn(IdxSize) -> (Option<IdxSize>, Option<IdxSize>),
{
    let mut idx_a = 0;

    // vec<ca>
    for probe_hashes in probe_hashes {
        // ca
        for probe_hashes in probe_hashes.data_views() {
            // chunk slices
            for &h in probe_hashes {
                // probe table that contains the hashed value
                let current_probe_table =
                    unsafe { get_hash_tbl_threaded_join_mut_partitioned(h, hash_tbls, n_tables) };

                let entry = current_probe_table
                    .raw_entry_mut()
                    .from_hash(h, |idx_hash| {
                        let idx_b = idx_hash.idx;
                        // Safety:
                        // indices in a join operation are always in bounds.
                        unsafe { compare_df_rows2(a, b, idx_a as usize, idx_b as usize) }
                    });

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
    }

    for hash_tbl in hash_tbls {
        hash_tbl.iter().for_each(|(_k, (tracker, indexes_b))| {
            // remaining unmatched joined values from the right table
            if !*tracker {
                results.extend(indexes_b.iter().map(|&idx_b| swap_fn_drain(idx_b)))
            }
        });
    }
}

pub fn _outer_join_multiple_keys(
    a: &mut DataFrame,
    b: &mut DataFrame,
    swap: bool,
) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
    // we assume that the b DataFrame is the shorter relation.
    // b will be used for the build phase.

    let size = a.height() + b.height();
    let mut results = Vec::with_capacity(size);

    let n_threads = POOL.current_num_threads();
    let dfs_a = split_df(a, n_threads).unwrap();
    let dfs_b = split_df(b, n_threads).unwrap();

    let (build_hashes, random_state) = df_rows_to_hashes_threaded(&dfs_b, None).unwrap();
    let (probe_hashes, _) = df_rows_to_hashes_threaded(&dfs_a, Some(random_state)).unwrap();

    let mut hash_tbls = create_build_table_outer(&build_hashes, b);
    // early drop to reduce memory pressure
    drop(build_hashes);

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
            a,
            b,
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
            a,
            b,
            |idx_a, idx_b| (Some(idx_a), Some(idx_b)),
            |idx_a| (Some(idx_a), None),
            |idx_b| (None, Some(idx_b)),
        )
    }
    results
}
