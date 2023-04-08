use std::cell::UnsafeCell;

use polars_arrow::trusted_len::PushUnchecked;

use super::*;
use crate::pipeline::PARTITION_SIZE;

const SPILL_SIZE: usize = 128;

#[derive(Clone)]
struct SpillPartitions {
    // outer vec: partitions (factor of 2)
    // inner vec: number of keys + number of aggregated columns
    partitions: Vec<Vec<AnyValueBufferTrusted<'static>>>,
    // outer vec: partitions
    // inner vec: aggregation columns
    finished: Vec<Vec<Series>>,
}

impl SpillPartitions {
    fn new(keys: &[DataType], aggs: &[DataType]) -> Self {
        let n_spills = keys.len() + aggs.len();

        let mut buf = Vec::with_capacity(n_spills);
        for dtype in keys {
            let builder = AnyValueBufferTrusted::new(dtype, 0);
            buf.push(builder);
        }
        for dtype in aggs {
            let builder = AnyValueBufferTrusted::new(dtype, 0);
            buf.push(builder);
        }

        let partitions = (0..PARTITION_SIZE).map(|partition| buf.clone()).collect();

        Self {
            partitions,
            finished: vec![],
        }
    }
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

    fn finish(&mut self) {
        if self.finished.is_empty() {
            let parts = std::mem::take(&mut self.partitions);
            self.finished = parts
                .into_iter()
                .map(|part| part.into_iter().map(|v| v.into_series()).collect())
                .collect();
        }
    }

    fn combine(&mut self, other: &mut Self) {
        self.finish();
        other.finish();

        for (part_self, part_other) in self.finished.iter_mut().zip(other.finished.iter()) {
            for (s, other) in part_self.iter_mut().zip(part_other.iter()) {
                s.append(other).unwrap();
            }
        }
    }
}

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
    output_schema: SchemaRef,
    pub num_keys: usize,
}

impl HashTbl {
    pub(super) fn split(&self) -> Self {
        Self {
            inner_map: Default::default(),
            keys: Default::default(),
            running_aggregations: Default::default(),
            agg_constructors: self.agg_constructors.iter().map(|ac| ac.split()).collect(),
            spill_partitions: self.spill_partitions.clone(),
            // ensure the capacity is set
            // later we push without checking bounds
            keys_scratch: unsafe {
                UnsafeCell::new(Vec::with_capacity((*self.keys_scratch.get()).capacity()))
            },
            num_keys: self.num_keys,
            output_schema: self.output_schema.clone(),
        }
    }

    pub(super) fn new(
        agg_constructors: Vec<AggregateFunction>,
        key_dtypes: &[DataType],
        output_schema: SchemaRef,
    ) -> Self {
        let agg_dtypes = agg_constructors
            .iter()
            .map(|agg| agg.dtype())
            .collect::<Vec<_>>();
        let spill_partitions = SpillPartitions::new(key_dtypes, &agg_dtypes);

        Self {
            inner_map: Default::default(),
            keys: Default::default(),
            running_aggregations: Default::default(),
            agg_constructors,
            spill_partitions,
            keys_scratch: UnsafeCell::new(Vec::with_capacity(key_dtypes.len())),
            num_keys: key_dtypes.len(),
            output_schema,
        }
    }

    unsafe fn get_keys(&self, idx: IdxSize) -> &[AnyValue] {
        let start = idx as usize;
        let end = start + self.num_keys;
        self.keys.get_unchecked(start..end)
    }

    pub(super) fn len(&self) -> usize {
        self.inner_map.len()
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

    /// # Safety
    /// Caller must ensure that `keys` and `agg_iters` are not depleted.
    pub(super) unsafe fn insert<'a>(
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

                if self.inner_map.len() > SPILL_SIZE {
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
                    self.running_aggregations.push(agg.split())
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

    pub(super) fn combine(&mut self, other: &mut Self) {
        for (key_other, agg_idx_other) in other.inner_map.iter() {
            // safety: idx is from the hashmap, so is in bounds
            let keys = unsafe { other.get_keys(key_other.idx) };

            let entry = self.get_entry(key_other.hash, keys);
            let agg_idx_self = match entry {
                RawEntryMut::Occupied(entry) => *entry.get(),
                RawEntryMut::Vacant(entry) => {
                    let borrow = &entry;
                    let borrow = borrow as *const RawVacantEntryMut<_, _, _> as usize;
                    // ensure the bck forgets this guy
                    std::mem::forget(entry);

                    let key_idx = self.keys.len() as IdxSize;
                    let aggregation_idx = self.running_aggregations.len() as IdxSize;

                    let key = Key::new(key_other.hash, key_idx);
                    for agg in &self.agg_constructors {
                        self.running_aggregations.push(agg.split())
                    }

                    // take a hold of the entry again and ensure it gets dropped
                    unsafe {
                        let borrow =
                            borrow as *const RawVacantEntryMut<'_, Key, IdxSize, IdBuildHasher>;
                        let entry = std::ptr::read(borrow);
                        entry.insert(key, aggregation_idx);
                    }
                    // update the keys
                    unsafe {
                        let start = key_other.idx as usize;
                        let end = start + self.num_keys;
                        let keys = other.keys.get_unchecked_release_mut(start..end);

                        for key in keys {
                            let mut owned_key = AnyValue::Null;
                            // this prevents cloning a string
                            std::mem::swap(&mut owned_key, key);
                            self.keys.push(owned_key)
                        }
                    }
                    aggregation_idx
                }
            };
            let start = *agg_idx_other as usize;
            let end = start + self.agg_constructors.len();
            let aggs_other =
                unsafe { other.running_aggregations.get_unchecked_release(start..end) };
            let start = agg_idx_self as usize;
            let end = start + self.agg_constructors.len();
            let aggs_self = unsafe {
                self.running_aggregations
                    .get_unchecked_release_mut(start..end)
            };
            for i in 0..aggs_self.len() {
                unsafe {
                    let agg_self = aggs_self.get_unchecked_release_mut(i);
                    let other = aggs_other.get_unchecked_release(i);
                    // TODO!: try transmutes
                    agg_self.combine(other.as_any())
                }
            }
        }
        self.spill_partitions.combine(&mut other.spill_partitions);
    }

    pub(super) fn finalize(
        &mut self,
        slice: &mut Option<(i64, usize)>,
    ) -> (Option<DataFrame>, Vec<Vec<Series>>) {
        let local_len = self.inner_map.len();
        let (skip_len, take_len) = if let Some((offset, slice_len)) = slice {
            if *offset as usize >= local_len {
                *offset -= local_len as i64;
                return (None, std::mem::take(&mut self.spill_partitions.finished));
            } else {
                let out = (*offset as usize, *slice_len);
                *offset = 0;
                *slice_len = slice_len.saturating_sub(local_len);
                out
            }
        } else {
            (0, local_len)
        };
        let inner_map = std::mem::take(&mut self.inner_map);

        let mut key_builders = self
            .output_schema
            .iter_dtypes()
            .take(self.num_keys)
            .map(|dtype| AnyValueBufferTrusted::new(&dtype.to_physical(), take_len))
            .collect::<Vec<_>>();

        let mut agg_builders = self
            .output_schema
            .iter_dtypes()
            .skip(self.num_keys)
            .map(|dtype| AnyValueBufferTrusted::new(dtype, take_len))
            .collect::<Vec<_>>();
        let num_aggs = self.agg_constructors.len();

        inner_map
            .into_iter()
            .skip(skip_len)
            .take(take_len)
            .for_each(|(k, agg_offset)| {
                let keys_offset = k.idx as usize;
                let keys = unsafe {
                    self.keys
                        .get_unchecked_release(keys_offset..keys_offset + self.num_keys)
                };

                // amortize loop counter
                for i in 0..self.num_keys {
                    unsafe {
                        let key = keys.get_unchecked_release(i);
                        let key_builder = key_builders.get_unchecked_release_mut(i);
                        key_builder.add_unchecked_owned_physical(&key.as_borrowed());
                    }
                }

                let start = agg_offset as usize;
                let end = start + num_aggs;
                for (i, buffer) in (start..end).zip(agg_builders.iter_mut()) {
                    unsafe {
                        let running_agg = self.running_aggregations.get_unchecked_release_mut(i);
                        let av = running_agg.finalize();
                        buffer.add_unchecked_owned_physical(&av);
                    }
                }
            });

        let mut cols = Vec::with_capacity(self.num_keys + self.agg_constructors.len());
        cols.extend(key_builders.into_iter().map(|buf| buf.into_series()));
        cols.extend(agg_builders.into_iter().map(|buf| buf.into_series()));
        physical_agg_to_logical(&mut cols, &self.output_schema);
        (
            Some(DataFrame::new_no_checks(cols)),
            std::mem::take(&mut self.spill_partitions.finished),
        )
    }
}
