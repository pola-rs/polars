use crate::frame::groupby::hashing::{populate_multiple_key_hashmap, HASHMAP_INIT_SIZE};
use crate::frame::hash_join::{
    get_hash_tbl_threaded_join_mut_partitioned, get_hash_tbl_threaded_join_partitioned,
};
use crate::prelude::*;
use crate::utils::{set_partition_size, split_df};
use crate::vector_hasher::{df_rows_to_hashes_threaded, this_partition, IdBuildHasher, IdxHash};
use crate::POOL;
use hashbrown::hash_map::RawEntryMut;
use hashbrown::HashMap;
use rayon::prelude::*;

/// Compare the rows of two DataFrames
unsafe fn compare_df_rows2(
    left: &DataFrame,
    right: &DataFrame,
    left_idx: usize,
    right_idx: usize,
) -> bool {
    for (l, r) in left.get_columns().iter().zip(right.get_columns()) {
        // get: there could be nulls.
        if !(l.get(left_idx) == r.get(right_idx)) {
            return false;
        }
    }
    true
}

fn create_build_table(
    hashes: &[UInt64Chunked],
    keys: &DataFrame,
) -> Vec<HashMap<IdxHash, Vec<u32>, IdBuildHasher>> {
    let n_partitions = set_partition_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions).into_par_iter().map(|part_no| {
            let part_no = part_no as u64;
            let mut hash_tbl: HashMap<IdxHash, Vec<u32>, IdBuildHasher> =
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

                    offset += len as u32;
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
    hash_tbls: &[HashMap<IdxHash, Vec<u32>, IdBuildHasher>],
    results: &mut Vec<(u32, u32)>,
    local_offset: usize,
    n_tables: u64,
    a: &DataFrame,
    b: &DataFrame,
    swap_fn: F,
) where
    F: Fn(u32, u32) -> (u32, u32),
{
    let mut idx_a = local_offset as u32;
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

fn get_offsets(probe_hashes: &[UInt64Chunked]) -> Vec<usize> {
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

pub(crate) fn inner_join_multiple_keys(
    a: &DataFrame,
    b: &DataFrame,
    swap: bool,
) -> Vec<(u32, u32)> {
    // we assume that the b DataFrame is the shorter relation.
    // b will be used for the build phase.

    let n_threads = POOL.current_num_threads();
    let dfs_a = split_df(a, n_threads).unwrap();
    let dfs_b = split_df(b, n_threads).unwrap();

    let (build_hashes, random_state) = df_rows_to_hashes_threaded(&dfs_b, None);
    let (probe_hashes, _) = df_rows_to_hashes_threaded(&dfs_a, Some(random_state));

    let hash_tbls = create_build_table(&build_hashes, b);
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
            .map(|(probe_hashes, offset)| {
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
            .flatten()
            .collect()
    })
}

#[cfg(feature = "private")]
pub fn private_left_join_multiple_keys(a: &DataFrame, b: &DataFrame) -> Vec<(u32, Option<u32>)> {
    left_join_multiple_keys(a, b)
}

pub(crate) fn left_join_multiple_keys(a: &DataFrame, b: &DataFrame) -> Vec<(u32, Option<u32>)> {
    // we assume that the b DataFrame is the shorter relation.
    // b will be used for the build phase.

    let n_threads = POOL.current_num_threads();
    let dfs_a = split_df(a, n_threads).unwrap();
    let dfs_b = split_df(b, n_threads).unwrap();

    let (build_hashes, random_state) = df_rows_to_hashes_threaded(&dfs_b, None);
    let (probe_hashes, _) = df_rows_to_hashes_threaded(&dfs_a, Some(random_state));

    let hash_tbls = create_build_table(&build_hashes, b);
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
            .map(|(probe_hashes, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;
                let mut results =
                    Vec::with_capacity(probe_hashes.len() / POOL.current_num_threads());
                let local_offset = offset;

                let mut idx_a = local_offset as u32;
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
                                results.extend(indexes_b.iter().map(|&idx_b| (idx_a, Some(idx_b))))
                            }
                            // only left values, right = null
                            None => results.push((idx_a, None)),
                        }
                        idx_a += 1;
                    }
                }

                results
            })
            .flatten()
            .collect()
    })
}

/// Probe the build table and add tuples to the results (inner join)
#[allow(clippy::too_many_arguments)]
fn probe_outer<F, G, H>(
    probe_hashes: &[UInt64Chunked],
    hash_tbls: &mut [HashMap<IdxHash, Vec<u32>, IdBuildHasher>],
    results: &mut Vec<(Option<u32>, Option<u32>)>,
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
    F: Fn(u32, u32) -> (Option<u32>, Option<u32>),
    // idx_a -> ...
    G: Fn(u32) -> (Option<u32>, Option<u32>),
    // idx_b -> ...
    H: Fn(u32) -> (Option<u32>, Option<u32>),
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
                    RawEntryMut::Occupied(occupied) => {
                        let indexes_b = occupied.remove();
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
        hash_tbl.iter().for_each(|(_k, indexes_b)| {
            // remaining joined values from the right table
            results.extend(indexes_b.iter().map(|&idx_b| swap_fn_drain(idx_b)))
        });
    }
}

pub(crate) fn outer_join_multiple_keys(
    a: &DataFrame,
    b: &DataFrame,
    swap: bool,
) -> Vec<(Option<u32>, Option<u32>)> {
    // we assume that the b DataFrame is the shorter relation.
    // b will be used for the build phase.

    let size = a.height() + b.height();
    let mut results = Vec::with_capacity(size);

    let n_threads = POOL.current_num_threads();
    let dfs_a = split_df(a, n_threads).unwrap();
    let dfs_b = split_df(b, n_threads).unwrap();

    let (build_hashes, random_state) = df_rows_to_hashes_threaded(&dfs_b, None);
    let (probe_hashes, _) = df_rows_to_hashes_threaded(&dfs_a, Some(random_state));

    let mut hash_tbls = create_build_table(&build_hashes, b);
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
