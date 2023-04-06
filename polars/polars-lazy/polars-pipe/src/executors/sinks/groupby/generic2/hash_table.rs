use std::cell::UnsafeCell;

use polars_arrow::trusted_len::PushUnchecked;

use super::*;

struct SpillPartitions {
    // number of different aggregations
    n_aggs: u32,
    // outer vec: partitions (factor of 2)
    // inner vec: number of keys + number of aggregated columns
    partitions: Vec<Vec<AnyValueBufferTrusted<'static>>>,
}

impl SpillPartitions {
    fn insert(&mut self, hash: u64, keys: &[AnyValue<'_>], aggs: &mut [SeriesPhysIter]) {
        let partition = hash_to_partition(hash, self.partitions.len());
        unsafe {
            let partitions = self.partitions.get_unchecked_release_mut(partition);
            debug_assert_eq!(keys.len(), partitions.len());

            // amortize the loop counter
            for i in 0..keys.len() {
                let av = keys.get_unchecked(i);
                let buf = partitions.get_unchecked_mut(i);
                // safety: we can trust the input types to be of consistent dtype
                buf.add_unchecked_owned_physical(av);
            }
            let mut i = keys.len();
            for agg in aggs {
                let av = agg.next().unwrap_unchecked_release();
                let buf = partitions.get_unchecked_mut(i);
                buf.add_unchecked_owned_physical(&av);
                i += 1;
            }
        };
    }
}

const LOCAL_SIZE: usize = 128;

pub(super) struct HashTbl {
    inner_map: PlIdHashMap<Key, IdxSize>,
    keys: Vec<AnyValue<'static>>,
    // the aggregation that are in process
    // the index the hashtable points to the start of the aggregations of that key/group
    running_aggregations: Vec<AggregateFunction>,
    // n aggregation function constructors
    agg_constructors: Vec<AggregateFunction>,
    spill_partitions: SpillPartitions,
    // amortize alloc
    // lifetime is tied to self, so we use static
    // to ensure bck to leave us be
    keys_scratch: UnsafeCell<Vec<AnyValue<'static>>>,
}

impl HashTbl {
    fn new(
        agg_constructors: Vec<AggregateFunction>,
        spill_partitions: SpillPartitions,
        n_keys: usize,
    ) -> Self {
        Self {
            inner_map: Default::default(),
            keys: Default::default(),
            running_aggregations: Default::default(),
            agg_constructors,
            spill_partitions,
            keys_scratch: UnsafeCell::new(Vec::with_capacity(n_keys)),
        }
    }

    fn get_entry(
        &mut self,
        hash: u64,
        tuples: &[AnyValue],
    ) -> RawEntryMut<Key, IdxSize, IdBuildHasher> {
        self.inner_map
            .raw_entry_mut()
            .from_hash(hash, |hash_map_key| {
                hash_map_key.hash == hash && {
                    let idx = hash_map_key.idx as usize;
                    if tuples.len() > 1 {
                        tuples.iter().enumerate().all(|(i, key)| unsafe {
                            self.keys.get_unchecked_release(i + idx) == key
                        })
                    } else {
                        unsafe {
                            self.keys.get_unchecked_release(idx) == tuples.get_unchecked_release(0)
                        }
                    }
                }
            })
    }

    fn insert<'a>(
        &'a mut self,
        hash: u64,
        keys: &mut [SeriesPhysIter],
        agg_iters: &mut [SeriesPhysIter],
        chunk_index: IdxSize,
    ) {
        // safety: no references
        let keys_scratch = unsafe { &mut *self.keys_scratch.get() };
        keys_scratch.clear();
        for key in keys {
            unsafe {
                // safety: this function should never be called if iterator is depleted
                let key = key.next().unwrap_unchecked_release();
                // safety: the static is temporary, we will never access them outside of this function
                let key = std::mem::transmute::<AnyValue<'_>, AnyValue<'static>>(key);
                // safety: we amortized n_keys
                keys_scratch.push_unchecked(key)
            }
        }

        let mut entry = self.get_entry(hash, keys_scratch);

        let agg_idx = match entry {
            RawEntryMut::Occupied(entry) => *entry.get(),
            RawEntryMut::Vacant(entry) => {
                // bchk shenanigans:
                // it does not allow us to hold a `raw entry` and in the meantime
                // have &self acces to get the length of keys
                // so we work with pointers instead
                let borrow = &entry;
                let borrow = borrow as *const RawVacantEntryMut<_, _, _> as usize;
                // ensure the bck forgets this guy
                std::mem::forget(entry);

                if self.inner_map.len() > LOCAL_SIZE {
                    unsafe {
                        // take a hold of the entry again and ensure it gets dropped
                        let borrow =
                            borrow as *const RawVacantEntryMut<'a, Key, IdxSize, IdBuildHasher>;
                        let _entry = std::ptr::read(borrow);
                    }
                    // safety: we can pass the `static` anyvalues
                    // because the spill method will copy the bytes to their own buffers
                    self.spill_partitions.insert(hash, keys_scratch, agg_iters);
                    return;
                }

                let aggregation_idx = self.running_aggregations.len() as IdxSize;
                let key_idx = self.keys.len() as IdxSize;

                let key = Key::new(hash, key_idx);
                unsafe {
                    // take a hold of the entry again and ensure it gets dropped
                    let borrow =
                        borrow as *const RawVacantEntryMut<'a, Key, IdxSize, IdBuildHasher>;
                    let entry = std::ptr::read(borrow);
                    entry.insert(key, aggregation_idx);
                }

                for agg in &self.agg_constructors {
                    self.running_aggregations.push(agg.split2())
                }

                unsafe {
                    self.keys.extend(
                        keys_scratch
                            .iter()
                            .map(|av| av.clone().into_static().unwrap_unchecked_release()),
                    );
                }
                aggregation_idx
            }
        };

        // apply the aggregation
        for (i, agg_iter) in agg_iters.iter_mut().enumerate() {
            let i = agg_idx as usize + i;
            let agg_fn = unsafe { self.running_aggregations.get_unchecked_release_mut(i) };

            agg_fn.pre_agg(chunk_index, agg_iter.as_mut())
        }
    }
}
