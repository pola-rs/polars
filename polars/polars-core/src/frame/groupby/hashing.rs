use crate::prelude::*;
use crate::utils::split_df;
use crate::vector_hasher::{
    create_hash_and_keys_threaded_vectorized, df_rows_to_hashes, df_rows_to_hashes_threaded,
    prepare_hashed_relation, this_thread, IdBuildHasher, IdxHash,
};
use crate::POOL;
use ahash::RandomState;
use hashbrown::hash_map::Entry;
use hashbrown::{hash_map::RawEntryMut, HashMap};
use rayon::prelude::*;
use std::hash::{BuildHasher, Hash, Hasher};
use super::GroupTuples;
use crate::vector_hasher::AsU64;

pub(crate) fn groupby<T>(
    a: impl Iterator<Item = T>,
    b: impl Iterator<Item = T>,
    preallocate: bool,
) -> GroupTuples
    where
        T: Hash + Eq,
{
    let hash_tbl = prepare_hashed_relation(a, b, preallocate);

    hash_tbl
        .into_iter()
        .map(|(_, indexes)| {
            let first = unsafe { *indexes.get_unchecked(0) };
            (first, indexes)
        })
        .collect()
}

pub(crate) fn groupby_threaded_num<T>(keys: Vec<&[T]>, group_size_hint: usize) -> GroupTuples
    where
        T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    let n_threads = keys.len();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_threads).into_par_iter().map(|thread_no| {
            let thread_no = thread_no as u64;

            let mut hash_tbl: HashMap<T, (u32, Vec<u32>), RandomState> = HashMap::default();

            let n_threads = n_threads as u64;
            let mut offset = 0;
            for keys in &keys {
                let len = keys.len() as u32;

                let mut cnt = 0;
                keys.iter().for_each(|k| {
                    let idx = cnt + offset;
                    cnt += 1;

                    if this_thread(k.as_u64(), thread_no, n_threads) {
                        let entry = hash_tbl.entry(*k);

                        match entry {
                            Entry::Vacant(entry) => {
                                let mut tuples = Vec::with_capacity(group_size_hint);
                                tuples.push(idx);
                                entry.insert((idx, tuples));
                            }
                            Entry::Occupied(mut entry) => {
                                let v = entry.get_mut();
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
        .flatten()
        .collect()
}

/// Determine groupby tuples from an iterator. The group_size_hint is used to pre-allocate the group vectors.
/// When the grouping column is a categorical type we already have a good indication of the avg size of the groups.
pub(crate) fn groupby_threaded_flat<I, T>(
    iters: Vec<I>,
    group_size_hint: usize,
    preallocate: bool,
) -> GroupTuples
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

            let mut hash_tbl: HashMap<T, (u32, Vec<u32>), RandomState> = if preallocate {
                HashMap::with_capacity_and_hasher(size / n_threads, random_state)
            } else {
                HashMap::with_hasher(random_state)
            };

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
        .flatten()
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
pub(crate) unsafe fn compare_df_rows(keys: &DataFrame, idx_a: usize, idx_b: usize) -> bool {
    for s in keys.get_columns() {
        if !s.equal_element(idx_a, idx_b, s) {
            return false;
        }
    }
    true
}

/// Populate a multiple key hashmap with row indexes.
/// Instead of the keys (which could be very large), the row indexes are stored.
/// To check if a row is equal the original DataFrame is also passed as ref.
/// When a hash collision occurs the indexes are ptrs to the rows and the rows are compared
/// on equality.
pub(crate) fn populate_multiple_key_hashmap<V, H, F, G>(
    hash_tbl: &mut HashMap<IdxHash, V, H>,
    // row index
    idx: u32,
    // hash
    original_h: u64,
    // keys of the hash table (will not be inserted, the indexes will be used)
    // the keys are needed for the equality check
    keys: &DataFrame,
    // value to insert
    vacant_fn: G,
    // function that gets a mutable ref to the occupied value in the hash table
    occupied_fn: F,
) where
    G: Fn() -> V,
    F: Fn(&mut V),
    H: BuildHasher,
{
    let hb = hash_tbl.hasher();
    let mut state = hb.build_hasher();
    original_h.hash(&mut state);
    let h = state.finish();
    let entry = hash_tbl
        .raw_entry_mut()
        // uses the idx to probe rows in the original DataFrame with keys
        // to check equality to find an entry
        .from_hash(h, |idx_hash| {
            let key_idx = idx_hash.idx;
            // Safety:
            // indices in a groupby operation are always in bounds.
            unsafe { compare_df_rows(keys, key_idx as usize, idx as usize) }
        });
    match entry {
        RawEntryMut::Vacant(entry) => {
            entry.insert_hashed_nocheck(h, IdxHash::new(idx, original_h), vacant_fn());
        }
        RawEntryMut::Occupied(mut entry) => {
            let (_k, v) = entry.get_key_value_mut();
            occupied_fn(v);
        }
    }
}

pub(crate) fn groupby_multiple_keys(keys: DataFrame) -> GroupTuples {
    let (hashes, _) = df_rows_to_hashes(&keys, None);
    let size = hashes.len();
    // rather over allocate because rehashing is expensive
    // its a complicated trade off, because often rehashing is cheaper than
    // overallocation because of cache coherence.
    let mut hash_tbl: HashMap<IdxHash, (u32, Vec<u32>), IdBuildHasher> =
        HashMap::with_capacity_and_hasher(size, IdBuildHasher::default());

    // hashes has no nulls
    let mut idx = 0;
    for hashes_chunk in hashes.data_views() {
        for &h in hashes_chunk {
            populate_multiple_key_hashmap(
                &mut hash_tbl,
                idx,
                h,
                &keys,
                || (idx, vec![idx]),
                |v| v.1.push(idx),
            );
            idx += 1;
        }
    }
    hash_tbl.into_iter().map(|(_k, v)| v).collect::<Vec<_>>()
}

pub(crate) fn groupby_threaded_multiple_keys_flat(keys: DataFrame, n_threads: usize) -> GroupTuples {
    let dfs = split_df(&keys, n_threads).unwrap();
    let (hashes, _random_state) = df_rows_to_hashes_threaded(&dfs, None);

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_threads).into_par_iter().map(|thread_no| {
            let hashes = &hashes;
            let thread_no = thread_no as u64;

            let keys = &keys;

            let mut hash_tbl: HashMap<IdxHash, (u32, Vec<u32>), RandomState> = HashMap::default();

            let n_threads = n_threads as u64;
            let mut offset = 0;
            for hashes in hashes {
                let len = hashes.len() as u32;

                let mut idx = 0;
                for hashes_chunk in hashes.data_views() {
                    for &h in hashes_chunk {
                        // partition hashes by thread no.
                        // So only a part of the hashes go to this hashmap
                        if this_thread(h, thread_no, n_threads) {
                            let idx = idx + offset;
                            populate_multiple_key_hashmap(
                                &mut hash_tbl,
                                idx,
                                h,
                                &keys,
                                || (idx, vec![idx]),
                                |v| v.1.push(idx),
                            );
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
