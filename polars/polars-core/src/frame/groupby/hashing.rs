use super::GroupTuples;
use crate::prelude::compare_inner::PartialEqInner;
use crate::prelude::*;
use crate::vector_hasher::{df_rows_to_hashes_threaded, IdBuildHasher, IdxHash};
use crate::vector_hasher::{this_partition, AsU64};
use crate::POOL;
use crate::{datatypes::PlHashMap, utils::split_df};
use hashbrown::hash_map::Entry;
use hashbrown::{hash_map::RawEntryMut, HashMap};
use rayon::prelude::*;
use std::hash::{BuildHasher, Hash};

// We must strike a balance between cache coherence and resizing costs.
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
pub(crate) const HASHMAP_INIT_SIZE: usize = 512;

pub(crate) fn groupby<T>(a: impl Iterator<Item = T>) -> GroupTuples
where
    T: Hash + Eq,
{
    let mut hash_tbl: PlHashMap<T, (u32, Vec<u32>)> = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);
    let mut cnt = 0;
    a.for_each(|k| {
        let idx = cnt;
        cnt += 1;
        let entry = hash_tbl.entry(k);

        match entry {
            Entry::Vacant(entry) => {
                entry.insert((idx, vec![idx]));
            }
            Entry::Occupied(mut entry) => {
                let v = entry.get_mut();
                v.1.push(idx);
            }
        }
    });

    hash_tbl.into_iter().map(|(_k, tpl)| tpl).collect()
}

/// Determine group tuples over different threads. The hash of the key is used to determine the partitions.
/// Note that rehashing of the keys should be cheap and the keys small to allow efficient rehashing and improved cache locality.
///
/// Besides numeric values, we can also use this for pre hashed strings. The keys are simply a ptr to the str + precomputed hash.
/// The hash will be used to rehash, and the str will be used for equality.
pub(crate) fn groupby_threaded_num<T, IntoSlice>(
    keys: Vec<IntoSlice>,
    group_size_hint: usize,
    n_partitions: u64,
) -> GroupTuples
where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    IntoSlice: AsRef<[T]> + Send + Sync,
{
    assert!(n_partitions.is_power_of_two());

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions).into_par_iter().map(|thread_no| {
            let thread_no = thread_no as u64;

            let mut hash_tbl: PlHashMap<T, (u32, Vec<u32>)> =
                PlHashMap::with_capacity(HASHMAP_INIT_SIZE);

            let mut offset = 0;
            for keys in &keys {
                let keys = keys.as_ref();
                let len = keys.len() as u32;

                let mut cnt = 0;
                keys.iter().for_each(|k| {
                    let idx = cnt + offset;
                    cnt += 1;

                    if this_partition(k.as_u64(), thread_no, n_partitions) {
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

/// Utility function used as comparison function in the hashmap.
/// The rationale is that equality is an AND operation and therefore its probability of success
/// declines rapidly with the number of keys. Instead of first copying an entire row from both
/// sides and then do the comparison, we do the comparison value by value catching early failures
/// eagerly.
///
/// # Safety
/// Doesn't check any bounds
#[inline]
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
    let entry = hash_tbl
        .raw_entry_mut()
        // uses the idx to probe rows in the original DataFrame with keys
        // to check equality to find an entry
        // this does not invalidate the hashmap as this equality function is not used
        // during rehashing/resize (then the keys are already known to be unique).
        // Only during insertion and probing an equality function is needed
        .from_hash(original_h, |idx_hash| {
            let key_idx = idx_hash.idx;
            // Safety:
            // indices in a groupby operation are always in bounds.
            unsafe { compare_df_rows(keys, key_idx as usize, idx as usize) }
        });
    match entry {
        RawEntryMut::Vacant(entry) => {
            entry.insert_hashed_nocheck(original_h, IdxHash::new(idx, original_h), vacant_fn());
        }
        RawEntryMut::Occupied(mut entry) => {
            let (_k, v) = entry.get_key_value_mut();
            occupied_fn(v);
        }
    }
}

#[inline]
pub(crate) unsafe fn compare_keys<'a>(
    keys_cmp: &'a [Box<dyn PartialEqInner + 'a>],
    idx_a: usize,
    idx_b: usize,
) -> bool {
    for cmp in keys_cmp {
        if !cmp.eq_element_unchecked(idx_a, idx_b) {
            return false;
        }
    }
    true
}

// Differs in the because this one uses the PartialEqInner trait objects
// is faster when multiple chunks. Not yet used in join.
pub(crate) fn populate_multiple_key_hashmap2<'a, V, H, F, G>(
    hash_tbl: &mut HashMap<IdxHash, V, H>,
    // row index
    idx: u32,
    // hash
    original_h: u64,
    // keys of the hash table (will not be inserted, the indexes will be used)
    // the keys are needed for the equality check
    keys_cmp: &'a [Box<dyn PartialEqInner + 'a>],
    // value to insert
    vacant_fn: G,
    // function that gets a mutable ref to the occupied value in the hash table
    occupied_fn: F,
) where
    G: Fn() -> V,
    F: Fn(&mut V),
    H: BuildHasher,
{
    let entry = hash_tbl
        .raw_entry_mut()
        // uses the idx to probe rows in the original DataFrame with keys
        // to check equality to find an entry
        // this does not invalidate the hashmap as this equality function is not used
        // during rehashing/resize (then the keys are already known to be unique).
        // Only during insertion and probing an equality function is needed
        .from_hash(original_h, |idx_hash| {
            let key_idx = idx_hash.idx;
            // Safety:
            // indices in a groupby operation are always in bounds.
            unsafe { compare_keys(keys_cmp, key_idx as usize, idx as usize) }
        });
    match entry {
        RawEntryMut::Vacant(entry) => {
            entry.insert_hashed_nocheck(original_h, IdxHash::new(idx, original_h), vacant_fn());
        }
        RawEntryMut::Occupied(mut entry) => {
            let (_k, v) = entry.get_key_value_mut();
            occupied_fn(v);
        }
    }
}

pub(crate) fn groupby_threaded_multiple_keys_flat(
    keys: DataFrame,
    n_partitions: usize,
) -> GroupTuples {
    let dfs = split_df(&keys, n_partitions).unwrap();
    let (hashes, _random_state) = df_rows_to_hashes_threaded(&dfs, None);
    let n_partitions = n_partitions as u64;

    // trait object to compare inner types.
    let keys_cmp = keys
        .iter()
        .map(|s| s.into_partial_eq_inner())
        .collect::<Vec<_>>();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions).into_par_iter().map(|thread_no| {
            let hashes = &hashes;
            let thread_no = thread_no as u64;

            let mut hash_tbl: HashMap<IdxHash, (u32, Vec<u32>), IdBuildHasher> =
                HashMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());

            let mut offset = 0;
            for hashes in hashes {
                let len = hashes.len() as u32;

                let mut idx = 0;
                for hashes_chunk in hashes.data_views() {
                    for &h in hashes_chunk {
                        // partition hashes by thread no.
                        // So only a part of the hashes go to this hashmap
                        if this_partition(h, thread_no, n_partitions) {
                            let idx = idx + offset;
                            populate_multiple_key_hashmap2(
                                &mut hash_tbl,
                                idx,
                                h,
                                &keys_cmp,
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
