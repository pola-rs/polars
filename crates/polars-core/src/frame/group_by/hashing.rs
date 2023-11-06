use std::hash::{BuildHasher, Hash};

use hashbrown::hash_map::{Entry, RawEntryMut};
use hashbrown::HashMap;
use polars_utils::iter::EnumerateIdxTrait;
use polars_utils::sync::SyncPtr;
use rayon::prelude::*;

use super::GroupsProxy;
use crate::datatypes::PlHashMap;
use crate::frame::group_by::{GroupsIdx, IdxItem};
use crate::hashing::{
    _df_rows_to_hashes_threaded_vertical, series_to_hashes, this_partition, AsU64, IdBuildHasher,
    IdxHash, *,
};
use crate::prelude::compare_inner::PartialEqInner;
use crate::prelude::*;
use crate::utils::{flatten, split_df, CustomIterTools};
use crate::POOL;

fn get_init_size() -> usize {
    // we check if this is executed from the main thread
    // we don't want to pre-allocate this much if executed
    // group_tuples in a parallel iterator as that explodes allocation
    if POOL.current_thread_index().is_none() {
        _HASHMAP_INIT_SIZE
    } else {
        0
    }
}

fn finish_group_order(mut out: Vec<Vec<IdxItem>>, sorted: bool) -> GroupsProxy {
    if sorted {
        // we can just take the first value, no need to flatten
        let mut out = if out.len() == 1 {
            out.pop().unwrap()
        } else {
            let (cap, offsets) = flatten::cap_and_offsets(&out);
            // we write (first, all) tuple because of sorting
            let mut items = Vec::with_capacity(cap);
            let items_ptr = unsafe { SyncPtr::new(items.as_mut_ptr()) };

            POOL.install(|| {
                out.into_par_iter()
                    .zip(offsets)
                    .for_each(|(mut g, offset)| {
                        // pre-sort every array
                        // this will make the final single threaded sort much faster
                        g.sort_unstable_by_key(|g| g.0);

                        unsafe {
                            let mut items_ptr: *mut (IdxSize, Vec<IdxSize>) = items_ptr.get();
                            items_ptr = items_ptr.add(offset);

                            for (i, g) in g.into_iter().enumerate() {
                                std::ptr::write(items_ptr.add(i), g)
                            }
                        }
                    });
            });
            unsafe {
                items.set_len(cap);
            }
            items
        };
        out.sort_unstable_by_key(|g| g.0);
        let mut idx = GroupsIdx::from_iter(out);
        idx.sorted = true;
        GroupsProxy::Idx(idx)
    } else {
        // we can just take the first value, no need to flatten
        if out.len() == 1 {
            GroupsProxy::Idx(GroupsIdx::from(out.pop().unwrap()))
        } else {
            // flattens
            GroupsProxy::Idx(GroupsIdx::from(out))
        }
    }
}

// The inner vecs should be sorted by [`IdxSize`]
// the group_by multiple keys variants suffice
// this requirements as they use an [`IdxMap`] strategy
fn finish_group_order_vecs(
    mut vecs: Vec<(Vec<IdxSize>, Vec<Vec<IdxSize>>)>,
    sorted: bool,
) -> GroupsProxy {
    if sorted {
        if vecs.len() == 1 {
            let (first, all) = vecs.pop().unwrap();
            return GroupsProxy::Idx(GroupsIdx::new(first, all, true));
        }

        let cap = vecs.iter().map(|v| v.0.len()).sum::<usize>();
        let offsets = vecs
            .iter()
            .scan(0_usize, |acc, v| {
                let out = *acc;
                *acc += v.0.len();
                Some(out)
            })
            .collect::<Vec<_>>();

        // we write (first, all) tuple because of sorting
        let mut items = Vec::with_capacity(cap);
        let items_ptr = unsafe { SyncPtr::new(items.as_mut_ptr()) };

        POOL.install(|| {
            vecs.into_par_iter()
                .zip(offsets)
                .for_each(|((first, all), offset)| {
                    // pre-sort every array not needed as items are already sorted
                    // this is due to using an index hashmap

                    unsafe {
                        let mut items_ptr: *mut (IdxSize, Vec<IdxSize>) = items_ptr.get();
                        items_ptr = items_ptr.add(offset);

                        // give the compiler some info
                        // maybe it may elide some loop counters
                        assert_eq!(first.len(), all.len());
                        for (i, (first, all)) in first.into_iter().zip(all).enumerate() {
                            std::ptr::write(items_ptr.add(i), (first, all))
                        }
                    }
                });
        });
        unsafe {
            items.set_len(cap);
        }
        // sort again
        items.sort_unstable_by_key(|g| g.0);

        let mut idx = GroupsIdx::from_iter(items);
        idx.sorted = true;
        GroupsProxy::Idx(idx)
    } else {
        // this materialization is parallel in the from impl.
        GroupsProxy::Idx(GroupsIdx::from(vecs))
    }
}

pub(crate) fn group_by<T>(a: impl Iterator<Item = T>, sorted: bool) -> GroupsProxy
where
    T: Hash + Eq,
{
    let init_size = get_init_size();
    let mut hash_tbl: PlHashMap<T, (IdxSize, Vec<IdxSize>)> = PlHashMap::with_capacity(init_size);
    let mut cnt = 0;
    a.for_each(|k| {
        let idx = cnt;
        cnt += 1;
        let entry = hash_tbl.entry(k);

        match entry {
            Entry::Vacant(entry) => {
                entry.insert((idx, vec![idx]));
            },
            Entry::Occupied(mut entry) => {
                let v = entry.get_mut();
                v.1.push(idx);
            },
        }
    });
    if sorted {
        let mut groups = hash_tbl
            .into_iter()
            .map(|(_k, v)| v)
            .collect_trusted::<Vec<_>>();
        groups.sort_unstable_by_key(|g| g.0);
        let mut idx: GroupsIdx = groups.into_iter().collect();
        idx.sorted = true;
        GroupsProxy::Idx(idx)
    } else {
        GroupsProxy::Idx(hash_tbl.into_values().collect())
    }
}

// giving the slice info to the compiler is much
// faster than the using an iterator, that's why we
// have the code duplication
pub(crate) fn group_by_threaded_slice<T, IntoSlice>(
    keys: Vec<IntoSlice>,
    n_partitions: u64,
    sorted: bool,
) -> GroupsProxy
where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    IntoSlice: AsRef<[T]> + Send + Sync,
{
    let init_size = get_init_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    let out = POOL.install(|| {
        (0..n_partitions)
            .into_par_iter()
            .map(|thread_no| {
                let mut hash_tbl: PlHashMap<T, (IdxSize, Vec<IdxSize>)> =
                    PlHashMap::with_capacity(init_size);

                let mut offset = 0;
                for keys in &keys {
                    let keys = keys.as_ref();
                    let len = keys.len() as IdxSize;
                    let hasher = hash_tbl.hasher().clone();

                    let mut cnt = 0;
                    keys.iter().for_each(|k| {
                        let idx = cnt + offset;
                        cnt += 1;

                        if this_partition(k.as_u64(), thread_no, n_partitions) {
                            let hash = hasher.hash_one(k);
                            let entry = hash_tbl.raw_entry_mut().from_key_hashed_nocheck(hash, k);

                            match entry {
                                RawEntryMut::Vacant(entry) => {
                                    let tuples = vec![idx];
                                    entry.insert_with_hasher(hash, *k, (idx, tuples), |k| {
                                        hasher.hash_one(k)
                                    });
                                },
                                RawEntryMut::Occupied(mut entry) => {
                                    let v = entry.get_mut();
                                    v.1.push(idx);
                                },
                            }
                        }
                    });
                    offset += len;
                }
                hash_tbl
                    .into_iter()
                    .map(|(_k, v)| v)
                    .collect_trusted::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    finish_group_order(out, sorted)
}

pub(crate) fn group_by_threaded_iter<T, I>(
    keys: &[I],
    n_partitions: u64,
    sorted: bool,
) -> GroupsProxy
where
    I: IntoIterator<Item = T> + Send + Sync + Copy,
    I::IntoIter: ExactSizeIterator,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    let init_size = get_init_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    let out = POOL.install(|| {
        (0..n_partitions)
            .into_par_iter()
            .map(|thread_no| {
                let mut hash_tbl: PlHashMap<T, (IdxSize, Vec<IdxSize>)> =
                    PlHashMap::with_capacity(init_size);

                let mut offset = 0;
                for keys in keys {
                    let keys = keys.into_iter();
                    let len = keys.len() as IdxSize;
                    let hasher = hash_tbl.hasher().clone();

                    let mut cnt = 0;
                    keys.for_each(|k| {
                        let idx = cnt + offset;
                        cnt += 1;

                        if this_partition(k.as_u64(), thread_no, n_partitions) {
                            let hash = hasher.hash_one(k);
                            let entry = hash_tbl.raw_entry_mut().from_key_hashed_nocheck(hash, &k);

                            match entry {
                                RawEntryMut::Vacant(entry) => {
                                    let tuples = vec![idx];
                                    entry.insert_with_hasher(hash, k, (idx, tuples), |k| {
                                        hasher.hash_one(k)
                                    });
                                },
                                RawEntryMut::Occupied(mut entry) => {
                                    let v = entry.get_mut();
                                    v.1.push(idx);
                                },
                            }
                        }
                    });
                    offset += len;
                }
                // iterating the hash tables locally
                // was faster than iterating in the materialization phase directly
                // the proper end vec. I believe this is because the hash-table
                // currently is local to the thread so in hot cache
                // So we first collect into a tight vec and then do a second
                // materialization run
                // this is also faster than the index-map approach where we
                // directly locally store to a vec at the cost of an extra
                // indirection
                hash_tbl
                    .into_iter()
                    .map(|(_k, v)| v)
                    .collect_trusted::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    finish_group_order(out, sorted)
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
    idx: IdxSize,
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
            // first check the hash values before we incur
            // cache misses
            original_h == idx_hash.hash && {
                let key_idx = idx_hash.idx;
                // Safety:
                // indices in a group_by operation are always in bounds.
                unsafe { compare_keys(keys_cmp, key_idx as usize, idx as usize) }
            }
        });
    match entry {
        RawEntryMut::Vacant(entry) => {
            entry.insert_hashed_nocheck(original_h, IdxHash::new(idx, original_h), vacant_fn());
        },
        RawEntryMut::Occupied(mut entry) => {
            let (_k, v) = entry.get_key_value_mut();
            occupied_fn(v);
        },
    }
}

pub(crate) fn group_by_threaded_multiple_keys_flat(
    mut keys: DataFrame,
    n_partitions: usize,
    sorted: bool,
) -> PolarsResult<GroupsProxy> {
    let dfs = split_df(&mut keys, n_partitions).unwrap();
    let (hashes, _random_state) = _df_rows_to_hashes_threaded_vertical(&dfs, None)?;
    let n_partitions = n_partitions as u64;

    let init_size = get_init_size();

    // trait object to compare inner types.
    let keys_cmp = keys
        .iter()
        .map(|s| s.into_partial_eq_inner())
        .collect::<Vec<_>>();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    let v = POOL.install(|| {
        (0..n_partitions)
            .into_par_iter()
            .map(|thread_no| {
                let hashes = &hashes;

                // IndexMap, the indexes are stored in flat vectors
                // this ensures that order remains and iteration is fast
                let mut hash_tbl: HashMap<IdxHash, IdxSize, IdBuildHasher> =
                    HashMap::with_capacity_and_hasher(init_size, Default::default());
                let mut first_vals = Vec::with_capacity(init_size);
                let mut all_vals = Vec::with_capacity(init_size);

                // put the buffers behind a pointer so we can access them from as the bchk doesn't allow
                // 2 mutable borrows (this is safe as we don't alias)
                // even if the vecs reallocate, we have a pointer to the stack vec, and thus always
                // access the proper data.
                let all_buf_ptr =
                    &mut all_vals as *mut Vec<Vec<IdxSize>> as *const Vec<Vec<IdxSize>>;
                let first_buf_ptr = &mut first_vals as *mut Vec<IdxSize> as *const Vec<IdxSize>;

                let mut offset = 0;
                for hashes in hashes {
                    let len = hashes.len() as IdxSize;

                    let mut idx = 0;
                    for hashes_chunk in hashes.data_views() {
                        for &h in hashes_chunk {
                            // partition hashes by thread no.
                            // So only a part of the hashes go to this hashmap
                            if this_partition(h, thread_no, n_partitions) {
                                let row_idx = idx + offset;
                                populate_multiple_key_hashmap2(
                                    &mut hash_tbl,
                                    row_idx,
                                    h,
                                    &keys_cmp,
                                    || unsafe {
                                        let first_vals = &mut *(first_buf_ptr as *mut Vec<IdxSize>);
                                        let all_vals =
                                            &mut *(all_buf_ptr as *mut Vec<Vec<IdxSize>>);
                                        let offset_idx = first_vals.len() as IdxSize;

                                        let tuples = vec![row_idx];
                                        all_vals.push(tuples);
                                        first_vals.push(row_idx);
                                        offset_idx
                                    },
                                    |v| unsafe {
                                        let all_vals =
                                            &mut *(all_buf_ptr as *mut Vec<Vec<IdxSize>>);
                                        let offset_idx = *v;
                                        let buf = all_vals.get_unchecked_mut(offset_idx as usize);
                                        buf.push(row_idx)
                                    },
                                );
                            }
                            idx += 1;
                        }
                    }

                    offset += len;
                }
                (first_vals, all_vals)
            })
            .collect::<Vec<_>>()
    });
    Ok(finish_group_order_vecs(v, sorted))
}

pub(crate) fn group_by_multiple_keys(keys: DataFrame, sorted: bool) -> PolarsResult<GroupsProxy> {
    let mut hashes = Vec::with_capacity(keys.height());
    let _ = series_to_hashes(keys.get_columns(), None, &mut hashes)?;

    let init_size = get_init_size();

    // trait object to compare inner types.
    let keys_cmp = keys
        .iter()
        .map(|s| s.into_partial_eq_inner())
        .collect::<Vec<_>>();

    // IndexMap, the indexes are stored in flat vectors
    // this ensures that order remains and iteration is fast
    let mut hash_tbl: HashMap<IdxHash, IdxSize, IdBuildHasher> =
        HashMap::with_capacity_and_hasher(init_size, Default::default());
    let mut first_vals = Vec::with_capacity(init_size);
    let mut all_vals = Vec::with_capacity(init_size);

    // put the buffers behind a pointer so we can access them from as the bchk doesn't allow
    // 2 mutable borrows (this is safe as we don't alias)
    // even if the vecs reallocate, we have a pointer to the stack vec, and thus always
    // access the proper data.
    let all_buf_ptr = &mut all_vals as *mut Vec<Vec<IdxSize>> as *const Vec<Vec<IdxSize>>;
    let first_buf_ptr = &mut first_vals as *mut Vec<IdxSize> as *const Vec<IdxSize>;

    for (row_idx, h) in hashes.into_iter().enumerate_idx() {
        populate_multiple_key_hashmap2(
            &mut hash_tbl,
            row_idx,
            h,
            &keys_cmp,
            || unsafe {
                let first_vals = &mut *(first_buf_ptr as *mut Vec<IdxSize>);
                let all_vals = &mut *(all_buf_ptr as *mut Vec<Vec<IdxSize>>);
                let offset_idx = first_vals.len() as IdxSize;

                let tuples = vec![row_idx];
                all_vals.push(tuples);
                first_vals.push(row_idx);
                offset_idx
            },
            |v| unsafe {
                let all_vals = &mut *(all_buf_ptr as *mut Vec<Vec<IdxSize>>);
                let offset_idx = *v;
                let buf = all_vals.get_unchecked_mut(offset_idx as usize);
                buf.push(row_idx)
            },
        );
    }

    let v = vec![(first_vals, all_vals)];
    Ok(finish_group_order_vecs(v, sorted))
}
