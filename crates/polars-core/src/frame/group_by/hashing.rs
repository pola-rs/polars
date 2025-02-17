use hashbrown::hash_map::Entry;
use polars_utils::hashing::{hash_to_partition, DirtyHash};
use polars_utils::idx_vec::IdxVec;
use polars_utils::itertools::Itertools;
use polars_utils::sync::SyncPtr;
use polars_utils::total_ord::{ToTotalOrd, TotalHash, TotalOrdWrap};
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

fn finish_group_order(mut out: Vec<Vec<IdxItem>>, sorted: bool) -> GroupsType {
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
        GroupsType::Idx(idx)
    } else {
        // we can just take the first value, no need to flatten
        if out.len() == 1 {
            GroupsType::Idx(GroupsIdx::from(out.pop().unwrap()))
        } else {
            // flattens
            GroupsType::Idx(GroupsIdx::from(out))
        }
    }
}

pub(crate) fn group_by<K>(keys: impl Iterator<Item = K>, sorted: bool) -> GroupsType
where
    K: TotalHash + TotalEq,
{
    let init_size = get_init_size();
    let (mut first, mut groups);
    if sorted {
        groups = Vec::with_capacity(get_init_size());
        first = Vec::with_capacity(get_init_size());
        let mut hash_tbl = PlHashMap::with_capacity(init_size);
        for (idx, k) in keys.enumerate_idx() {
            match hash_tbl.entry(TotalOrdWrap(k)) {
                Entry::Vacant(entry) => {
                    let group_idx = groups.len() as IdxSize;
                    entry.insert(group_idx);
                    groups.push(unitvec![idx]);
                    first.push(idx);
                },
                Entry::Occupied(entry) => unsafe {
                    groups.get_unchecked_mut(*entry.get() as usize).push(idx)
                },
            }
        }
    } else {
        let mut hash_tbl = PlHashMap::with_capacity(init_size);
        for (idx, k) in keys.enumerate_idx() {
            match hash_tbl.entry(TotalOrdWrap(k)) {
                Entry::Vacant(entry) => {
                    entry.insert((idx, unitvec![idx]));
                },
                Entry::Occupied(mut entry) => entry.get_mut().1.push(idx),
            }
        }
        (first, groups) = hash_tbl.into_values().unzip();
    }
    GroupsType::Idx(GroupsIdx::new(first, groups, sorted))
}

// giving the slice info to the compiler is much
// faster than the using an iterator, that's why we
// have the code duplication
pub(crate) fn group_by_threaded_slice<T, IntoSlice>(
    keys: Vec<IntoSlice>,
    n_partitions: usize,
    sorted: bool,
) -> GroupsType
where
    T: ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Sync + Copy + DirtyHash,
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
                let mut hash_tbl = PlHashMap::with_capacity(init_size);

                let mut offset = 0;
                for keys in &keys {
                    let keys = keys.as_ref();
                    let len = keys.len() as IdxSize;

                    for (key_idx, k) in keys.iter().enumerate_idx() {
                        let k = k.to_total_ord();
                        let idx = key_idx + offset;

                        if thread_no == hash_to_partition(k.dirty_hash(), n_partitions) {
                            match hash_tbl.entry(k) {
                                Entry::Vacant(entry) => {
                                    entry.insert((idx, unitvec![idx]));
                                },
                                Entry::Occupied(mut entry) => {
                                    entry.get_mut().1.push(idx);
                                },
                            }
                        }
                    }
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
) -> GroupsType
where
    I: IntoIterator<Item = T> + Send + Sync + Clone,
    I::IntoIter: ExactSizeIterator,
    T: ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Sync + Copy + DirtyHash,
{
    let init_size = get_init_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    let out = POOL.install(|| {
        (0..n_partitions)
            .into_par_iter()
            .map(|thread_no| {
                let mut hash_tbl: PlHashMap<T::TotalOrdItem, IdxVec> =
                    PlHashMap::with_capacity(init_size);

                let mut offset = 0;
                for keys in keys {
                    let keys = keys.clone().into_iter();
                    let len = keys.len() as IdxSize;

                    for (key_idx, k) in keys.into_iter().enumerate_idx() {
                        let k = k.to_total_ord();
                        let idx = key_idx + offset;

                        if thread_no == hash_to_partition(k.dirty_hash(), n_partitions) {
                            match hash_tbl.entry(k) {
                                Entry::Vacant(entry) => {
                                    entry.insert(unitvec![idx]);
                                },
                                Entry::Occupied(mut entry) => {
                                    entry.get_mut().push(idx);
                                },
                            }
                        }
                    }
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
                    .map(|(_k, v)| (unsafe { *v.first().unwrap_unchecked() }, v))
                    .collect_trusted::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    finish_group_order(out, sorted)
}
