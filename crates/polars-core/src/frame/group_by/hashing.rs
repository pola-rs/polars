use std::hash::{BuildHasher, Hash, Hasher};

use hashbrown::hash_map::RawEntryMut;
use polars_utils::hashing::{hash_to_partition, DirtyHash};
use polars_utils::idx_vec::IdxVec;
use polars_utils::sync::SyncPtr;
use polars_utils::total_ord::{ToTotalOrd, TotalHash};
use polars_utils::unitvec;
use rayon::prelude::*;

use crate::hashing::*;
use crate::prelude::*;
use crate::utils::flatten;
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
                            let mut items_ptr: *mut (IdxSize, IdxVec) = items_ptr.get();
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

pub(crate) fn group_by<T>(a: impl Iterator<Item = T>, sorted: bool) -> GroupsProxy
where
    T: TotalHash + TotalEq,
{
    let init_size = get_init_size();
    let mut hash_tbl: PlHashMap<T, (IdxSize, IdxVec)> = PlHashMap::with_capacity(init_size);
    let hasher = hash_tbl.hasher().clone();
    let mut cnt = 0;
    a.for_each(|k| {
        let idx = cnt;
        cnt += 1;

        let mut state = hasher.build_hasher();
        k.tot_hash(&mut state);
        let h = state.finish();
        let entry = hash_tbl.raw_entry_mut().from_hash(h, |k_| k.tot_eq(k_));

        match entry {
            RawEntryMut::Vacant(entry) => {
                let tuples = unitvec![idx];
                entry.insert_with_hasher(h, k, (idx, tuples), |k| {
                    let mut state = hasher.build_hasher();
                    k.tot_hash(&mut state);
                    state.finish()
                });
            },
            RawEntryMut::Occupied(mut entry) => {
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
    n_partitions: usize,
    sorted: bool,
) -> GroupsProxy
where
    T: TotalHash + TotalEq + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Hash + Eq + Sync + Copy + DirtyHash,
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
                let mut hash_tbl: PlHashMap<T::TotalOrdItem, (IdxSize, IdxVec)> =
                    PlHashMap::with_capacity(init_size);

                let mut offset = 0;
                for keys in &keys {
                    let keys = keys.as_ref();
                    let len = keys.len() as IdxSize;
                    let hasher = hash_tbl.hasher().clone();

                    let mut cnt = 0;
                    keys.iter().for_each(|k| {
                        let k = k.to_total_ord();
                        let idx = cnt + offset;
                        cnt += 1;

                        if thread_no == hash_to_partition(k.dirty_hash(), n_partitions) {
                            let hash = hasher.hash_one(k);
                            let entry = hash_tbl.raw_entry_mut().from_key_hashed_nocheck(hash, &k);

                            match entry {
                                RawEntryMut::Vacant(entry) => {
                                    let tuples = unitvec![idx];
                                    entry.insert_with_hasher(hash, k, (idx, tuples), |k| {
                                        hasher.hash_one(*k)
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
    n_partitions: usize,
    sorted: bool,
) -> GroupsProxy
where
    I: IntoIterator<Item = T> + Send + Sync + Clone,
    I::IntoIter: ExactSizeIterator,
    T: TotalHash + TotalEq + DirtyHash + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Hash + Eq + Sync + Copy + DirtyHash,
{
    let init_size = get_init_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    let out = POOL.install(|| {
        (0..n_partitions)
            .into_par_iter()
            .map(|thread_no| {
                let mut hash_tbl: PlHashMap<T::TotalOrdItem, (IdxSize, IdxVec)> =
                    PlHashMap::with_capacity(init_size);

                let mut offset = 0;
                for keys in keys {
                    let keys = keys.clone().into_iter();
                    let len = keys.len() as IdxSize;
                    let hasher = hash_tbl.hasher().clone();

                    let mut cnt = 0;
                    keys.for_each(|k| {
                        let k = k.to_total_ord();
                        let idx = cnt + offset;
                        cnt += 1;

                        if thread_no == hash_to_partition(k.dirty_hash(), n_partitions) {
                            let hash = hasher.hash_one(k);
                            let entry = hash_tbl.raw_entry_mut().from_key_hashed_nocheck(hash, &k);

                            match entry {
                                RawEntryMut::Vacant(entry) => {
                                    let tuples = unitvec![idx];
                                    entry.insert_with_hasher(hash, k, (idx, tuples), |k| {
                                        hasher.hash_one(*k)
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
