use super::*;
use crate::pipeline::PARTITION_SIZE;

const OB_SIZE: usize = 2048;

#[derive(Clone)]
struct SpillPartitions {
    // outer vec: partitions (factor of 2)
    // inner vec: number of keys + number of aggregated columns
    keys_aggs_partitioned: PartitionVec<Vec<AnyValueBufferTrusted<'static>>>,
    hash_partitioned: PartitionVec<Vec<u64>>,
    chunk_index_partitioned: PartitionVec<Vec<IdxSize>>,
    num_keys: usize,
    spilled: bool,
    // this only fills during the reduce phase IFF
    // there are spilled tuples
    finished_payloads: PartitionVec<Vec<SpillPayload>>,
    keys_dtypes: Arc<[DataType]>,
    agg_dtypes: Arc<[DataType]>,
}

impl SpillPartitions {
    fn new(keys: Arc<[DataType]>, aggs: Arc<[DataType]>) -> Self {
        let hash_partitioned = vec![];
        let chunk_index_partitioned = vec![];

        // construct via split so that preallocation succeeds
        Self {
            keys_aggs_partitioned: vec![],
            hash_partitioned,
            chunk_index_partitioned,
            num_keys: keys.as_ref().len(),
            spilled: false,
            finished_payloads: vec![],
            keys_dtypes: keys,
            agg_dtypes: aggs,
        }
        .split()
    }

    fn split(&self) -> Self {
        let n_columns = self.keys_dtypes.as_ref().len() + self.agg_dtypes.as_ref().len();

        let keys_aggs_partitioned = (0..PARTITION_SIZE)
            .map(|_| {
                let mut buf = Vec::with_capacity(n_columns);
                for dtype in self.keys_dtypes.as_ref() {
                    let builder = AnyValueBufferTrusted::new(dtype, OB_SIZE);
                    buf.push(builder);
                }
                for dtype in self.agg_dtypes.as_ref() {
                    let builder = AnyValueBufferTrusted::new(dtype, OB_SIZE);
                    buf.push(builder);
                }
                buf
            })
            .collect();

        let hash_partitioned = vec![Vec::with_capacity(OB_SIZE); PARTITION_SIZE];
        let chunk_index_partitioned = vec![Vec::with_capacity(OB_SIZE); PARTITION_SIZE];

        Self {
            keys_aggs_partitioned,
            hash_partitioned,
            chunk_index_partitioned,
            num_keys: self.num_keys,
            spilled: false,
            finished_payloads: vec![],
            keys_dtypes: self.keys_dtypes.clone(),
            agg_dtypes: self.agg_dtypes.clone(),
        }
    }
}

impl SpillPartitions {
    fn pre_alloc(&mut self) {
        if !self.spilled {
            let n_columns = self.keys_dtypes.as_ref().len() + self.agg_dtypes.as_ref().len();

            self.keys_aggs_partitioned = (0..PARTITION_SIZE)
                .map(|_| {
                    let mut buf = Vec::with_capacity(n_columns);
                    for dtype in self.keys_dtypes.as_ref() {
                        let builder = AnyValueBufferTrusted::new(dtype, OB_SIZE);
                        buf.push(builder);
                    }
                    for dtype in self.agg_dtypes.as_ref() {
                        let builder = AnyValueBufferTrusted::new(dtype, OB_SIZE);
                        buf.push(builder);
                    }
                    buf
                })
                .collect();

            self.hash_partitioned = vec![Vec::with_capacity(OB_SIZE); PARTITION_SIZE];
            self.chunk_index_partitioned = vec![Vec::with_capacity(OB_SIZE); PARTITION_SIZE];
        }
    }
    /// Returns (partition, overflowing hashes, chunk_indexes, keys and aggs)
    fn insert(
        &mut self,
        hash: u64,
        chunk_idx: IdxSize,
        keys: &[AnyValue<'_>],
        aggs: &mut [SeriesPhysIter],
    ) -> Option<(usize, SpillPayload)> {
        self.pre_alloc();
        let partition = hash_to_partition(hash, self.keys_aggs_partitioned.len());
        self.spilled = true;
        unsafe {
            let keys_aggs = self
                .keys_aggs_partitioned
                .get_unchecked_release_mut(partition);
            let hashes = self.hash_partitioned.get_unchecked_release_mut(partition);
            let chunk_indexes = self
                .chunk_index_partitioned
                .get_unchecked_release_mut(partition);

            hashes.push(hash);
            chunk_indexes.push(chunk_idx);

            // amortize the loop counter
            for i in 0..keys.len() {
                let av = keys.get_unchecked(i);
                let buf = keys_aggs.get_unchecked_mut(i);
                // safety: we can trust the input types to be of consistent dtype
                buf.add_unchecked_borrowed_physical(av);
            }
            let mut i = keys.len();
            for agg in aggs {
                let av = agg.next().unwrap_unchecked_release();
                let buf = keys_aggs.get_unchecked_mut(i);
                buf.add_unchecked_borrowed_physical(&av);
                i += 1;
            }

            if hashes.len() >= OB_SIZE {
                let mut new_hashes = Vec::with_capacity(OB_SIZE);
                let mut new_chunk_indexes = Vec::with_capacity(OB_SIZE);
                std::mem::swap(&mut new_hashes, hashes);
                std::mem::swap(&mut new_chunk_indexes, chunk_indexes);

                Some((
                    partition,
                    SpillPayload {
                        hashes: new_hashes,
                        chunk_idx: new_chunk_indexes,
                        keys_and_aggs: keys_aggs.iter_mut().map(|b| b.reset(OB_SIZE)).collect(),
                        num_keys: self.num_keys,
                    },
                ))
            } else {
                None
            }
        }
    }

    fn finish(&mut self) {
        if self.spilled {
            let all_spilled = self.get_all_spilled().collect::<Vec<_>>();
            for (partition_i, payload) in all_spilled {
                let buf = if let Some(buf) = self.finished_payloads.get_mut(partition_i) {
                    buf
                } else {
                    self.finished_payloads.push(vec![]);
                    self.finished_payloads.last_mut().unwrap()
                };
                buf.push(payload)
            }
        }
    }

    fn combine(&mut self, other: &mut Self) {
        match (self.spilled, other.spilled) {
            (false, true) => std::mem::swap(self, other),
            (true, false) => {}
            (false, false) => {}
            (true, true) => {
                self.finish();
                other.finish();
                let other_payloads = std::mem::take(&mut other.finished_payloads);

                for (part_self, part_other) in self
                    .finished_payloads
                    .iter_mut()
                    .zip(other_payloads.into_iter())
                {
                    part_self.extend(part_other)
                }
            }
        }
    }

    fn get_all_spilled(&mut self) -> impl Iterator<Item = (usize, SpillPayload)> + '_ {
        // todo! allocate
        let mut flattened = vec![];
        let finished_payloads = std::mem::take(&mut self.finished_payloads);
        for (part, payloads) in finished_payloads.into_iter().enumerate() {
            for payload in payloads {
                flattened.push((part, payload))
            }
        }

        (0..PARTITION_SIZE)
            .map(|partition| unsafe {
                let keys_aggs = self
                    .keys_aggs_partitioned
                    .get_unchecked_release_mut(partition);
                let hashes = self.hash_partitioned.get_unchecked_release_mut(partition);
                let chunk_indexes = self
                    .chunk_index_partitioned
                    .get_unchecked_release_mut(partition);
                let hashes = std::mem::take(hashes);
                let chunk_idx = std::mem::take(chunk_indexes);

                (
                    partition,
                    SpillPayload {
                        hashes,
                        chunk_idx,
                        keys_and_aggs: keys_aggs.iter_mut().map(|b| b.reset(0)).collect(),
                        num_keys: self.num_keys,
                    },
                )
            })
            .chain(flattened.into_iter())
    }
}

pub(super) struct ThreadLocalTable {
    inner_map: AggHashTable<true>,
    spill_partitions: SpillPartitions,
}

impl ThreadLocalTable {
    pub(super) fn new(
        agg_constructors: Arc<[AggregateFunction]>,
        key_dtypes: Arc<[DataType]>,
        output_schema: SchemaRef,
    ) -> Self {
        let agg_dtypes: Arc<[DataType]> = Arc::from(
            agg_constructors
                .iter()
                .map(|agg| agg.dtype())
                .collect::<Vec<_>>(),
        );
        let spill_partitions = SpillPartitions::new(key_dtypes.clone(), agg_dtypes);

        Self {
            inner_map: AggHashTable::new(
                agg_constructors,
                key_dtypes.as_ref(),
                output_schema,
                Some(256),
            ),
            spill_partitions,
        }
    }

    pub(super) fn split(&self) -> Self {
        // should be called before any chunk is processed
        debug_assert!(self.inner_map.is_empty());

        Self {
            inner_map: self.inner_map.split(),
            spill_partitions: self.spill_partitions.clone(),
        }
    }

    pub(super) fn get_inner_map_mut(&mut self) -> &mut AggHashTable<true> {
        &mut self.inner_map
    }

    /// # Safety
    /// Caller must ensure that `keys` and `agg_iters` are not depleted.
    pub(super) unsafe fn insert(
        &mut self,
        hash: u64,
        keys: &mut [SeriesPhysIter],
        agg_iters: &mut [SeriesPhysIter],
        chunk_index: IdxSize,
    ) -> Option<(usize, SpillPayload)> {
        if let Some(keys) = self.inner_map.insert(hash, keys, agg_iters, chunk_index) {
            self.spill_partitions
                .insert(hash, chunk_index, keys, agg_iters)
        } else {
            None
        }
    }

    pub(super) fn combine(&mut self, other: &mut Self) {
        self.inner_map.combine(&mut other.inner_map);
        self.spill_partitions.combine(&mut other.spill_partitions);
    }

    pub(super) fn finalize(&mut self, slice: &mut Option<(i64, usize)>) -> Option<DataFrame> {
        if !self.spill_partitions.spilled {
            Some(self.inner_map.finalize(slice))
        } else {
            None
        }
    }

    pub(super) fn get_all_spilled(&mut self) -> impl Iterator<Item = (usize, SpillPayload)> + '_ {
        self.spill_partitions.get_all_spilled()
    }
}
