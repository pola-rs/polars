use arrow::legacy::trusted_len::TrustedLenPush;
use polars_utils::hashing::hash_to_partition;

use super::*;
use crate::pipeline::PARTITION_SIZE;

pub(super) struct AggHashTable<const FIXED: bool> {
    inner_map: PlIdHashMap<Key, u32>,
    // row data of the keys
    keys: Vec<u8>,
    // the aggregation that are in process
    // the index the hashtable points to the start of the aggregations of that key/group
    running_aggregations: Vec<AggregateFunction>,
    // n aggregation function constructors
    // The are used to create new running aggregators
    agg_constructors: Arc<[AggregateFunction]>,
    output_schema: SchemaRef,
    pub num_keys: usize,
    spill_size: usize,
}

impl<const FIXED: bool> AggHashTable<FIXED> {
    pub(super) fn new(
        agg_constructors: Arc<[AggregateFunction]>,
        key_dtypes: &[DataType],
        output_schema: SchemaRef,
        spill_size: Option<usize>,
    ) -> Self {
        assert_eq!(FIXED, spill_size.is_some());
        Self {
            inner_map: Default::default(),
            keys: Default::default(),
            running_aggregations: Default::default(),
            agg_constructors,
            num_keys: key_dtypes.len(),
            spill_size: spill_size.unwrap_or(usize::MAX),
            output_schema,
        }
    }

    pub(super) fn split(&self) -> Self {
        Self {
            inner_map: Default::default(),
            keys: Default::default(),
            running_aggregations: Default::default(),
            agg_constructors: self.agg_constructors.iter().map(|c| c.split()).collect(),
            num_keys: self.num_keys,
            spill_size: self.spill_size,
            output_schema: self.output_schema.clone(),
        }
    }

    unsafe fn get_keys_row(&self, key: &Key) -> &[u8] {
        let start = key.offset as usize;
        let end = start + key.len as usize;
        self.keys.get_unchecked(start..end)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.inner_map.is_empty()
    }

    fn get_entry(&mut self, hash: u64, row: &[u8]) -> RawEntryMut<Key, u32, IdBuildHasher> {
        let keys = self.keys.as_ptr();

        self.inner_map
            .raw_entry_mut()
            .from_hash(hash, |hash_map_key| {
                // first check the hash as that has no indirection
                hash_map_key.hash == hash && {
                    let offset = hash_map_key.offset as usize;
                    let len = hash_map_key.len as usize;

                    unsafe { std::slice::from_raw_parts(keys.add(offset), len) == row }
                }
            })
    }

    fn insert_key<'a>(&'a mut self, hash: u64, row: &[u8]) -> Option<u32> {
        let entry = self.get_entry(hash, row);

        match entry {
            RawEntryMut::Occupied(entry) => Some(*entry.get()),
            RawEntryMut::Vacant(entry) => {
                // bchk shenanigans:
                // it does not allow us to hold a `raw entry` and in the meantime
                // have &self access to get the length of keys
                // so we work with pointers instead
                let borrow = &entry;
                let borrow = borrow as *const RawVacantEntryMut<_, _, _> as usize;
                // ensure the bck forgets this guy
                #[allow(clippy::forget_non_drop)]
                std::mem::forget(entry);

                // OVERFLOW logic
                if FIXED && self.inner_map.len() > self.spill_size {
                    unsafe {
                        // take a hold of the entry again and ensure it gets dropped
                        let borrow =
                            borrow as *const RawVacantEntryMut<'a, Key, u32, IdBuildHasher>;
                        let _entry = std::ptr::read(borrow);
                    }
                    return None;
                }

                let aggregation_idx = self.running_aggregations.len() as u32;
                let key_offset = self.keys.len() as u32;
                let key_len = row.len() as u32;
                let key = Key::new(hash, key_offset, key_len);

                unsafe {
                    // take a hold of the entry again and ensure it gets dropped
                    let borrow = borrow as *const RawVacantEntryMut<'a, Key, u32, IdBuildHasher>;
                    let entry = std::ptr::read(borrow);
                    entry.insert(key, aggregation_idx);
                }

                for agg in self.agg_constructors.as_ref() {
                    self.running_aggregations.push(agg.split())
                }

                self.keys.extend_from_slice(row);
                Some(aggregation_idx)
            },
        }
    }

    /// # Safety
    /// Caller must ensure that `keys` and `agg_iters` are not depleted.
    /// # Returns &keys
    pub(super) unsafe fn insert(
        &mut self,
        hash: u64,
        key: &[u8],
        agg_iters: &mut [SeriesPhysIter],
        chunk_index: IdxSize,
    ) -> bool {
        let agg_idx = match self.insert_key(hash, key) {
            // overflow
            None => return true,
            Some(agg_idx) => agg_idx,
        };

        // apply the aggregation
        for (i, agg_iter) in agg_iters.iter_mut().enumerate() {
            let i = agg_idx as usize + i;
            let agg_fn = unsafe { self.running_aggregations.get_unchecked_release_mut(i) };

            agg_fn.pre_agg(chunk_index, agg_iter.as_mut())
        }
        // no overflow
        false
    }

    pub(super) fn combine(&mut self, other: &Self) {
        self.combine_impl(other, |_hash| true)
    }

    pub(super) fn combine_on_partition<const FIXED_OTHER: bool>(
        &mut self,
        partition: usize,
        other: &AggHashTable<FIXED_OTHER>,
    ) {
        self.combine_impl(other, |hash| {
            partition == hash_to_partition(hash, PARTITION_SIZE)
        })
    }

    pub(super) fn combine_impl<const FIXED_OTHER: bool, C>(
        &mut self,
        other: &AggHashTable<FIXED_OTHER>,
        on_condition: C,
    )
    // takes a hash and if true, this keys will be combined
    where
        C: Fn(u64) -> bool,
    {
        let spill_size = self.spill_size;
        self.spill_size = usize::MAX;
        for (key_other, agg_idx_other) in other.inner_map.iter() {
            // SAFETY: idx is from the hashmap, so is in bounds
            let row = unsafe { other.get_keys_row(key_other) };

            if on_condition(key_other.hash) {
                // SAFETY: will not overflow as we set it to usize::MAX;
                let agg_idx_self = unsafe {
                    self.insert_key(key_other.hash, row)
                        .unwrap_unchecked_release()
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
        }
        self.spill_size = spill_size;
    }

    pub(super) fn finalize(&mut self, slice: &mut Option<(i64, usize)>) -> DataFrame {
        let local_len = self.inner_map.len();
        let (skip_len, take_len) = if let Some((offset, slice_len)) = slice {
            if *offset as usize >= local_len {
                *offset -= local_len as i64;
                return DataFrame::empty_with_schema(&self.output_schema);
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
        let mut running_aggregations = std::mem::take(&mut self.running_aggregations);

        let mut agg_builders = self
            .agg_constructors
            .iter()
            .map(|ac| AnyValueBufferTrusted::new(&ac.dtype(), take_len))
            .collect::<Vec<_>>();
        let num_aggs = self.agg_constructors.len();
        let mut key_rows = Vec::with_capacity(take_len);

        inner_map
            .into_iter()
            .skip(skip_len)
            .take(take_len)
            .for_each(|(k, agg_offset)| {
                unsafe {
                    key_rows.push_unchecked(self.get_keys_row(&k));
                }

                let start = agg_offset as usize;
                let end = start + num_aggs;
                for (i, buffer) in (start..end).zip(agg_builders.iter_mut()) {
                    unsafe {
                        let running_agg = running_aggregations.get_unchecked_release_mut(i);
                        let av = running_agg.finalize();
                        // SAFETY: finalize creates owned AnyValues
                        buffer.add_unchecked_owned_physical(&av);
                    }
                }
            });

        let key_dtypes = self
            .output_schema
            .iter_dtypes()
            .take(self.num_keys)
            .map(|dtype| dtype.to_physical().to_arrow(CompatLevel::newest()))
            .collect::<Vec<_>>();
        let fields = vec![Default::default(); self.num_keys];
        let key_columns =
            unsafe { polars_row::decode::decode_rows(&mut key_rows, &fields, &key_dtypes) };

        let mut cols = Vec::with_capacity(self.num_keys + self.agg_constructors.len());
        cols.extend(
            key_columns
                .into_iter()
                .map(|arr| Series::try_from(("", arr)).unwrap()),
        );
        cols.extend(agg_builders.into_iter().map(|buf| buf.into_series()));
        physical_agg_to_logical(&mut cols, &self.output_schema);
        unsafe { DataFrame::new_no_checks(cols) }
    }
}

unsafe impl<const FIXED: bool> Send for AggHashTable<FIXED> {}
unsafe impl<const FIXED: bool> Sync for AggHashTable<FIXED> {}
