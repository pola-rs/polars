use std::io::Write;

use polars_core::hashing::this_partition;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;

use super::super::constants::PARTITION_HASHMAP_STATE_NAME;
use super::*;
use crate::pipeline::PARTITION_SIZE;

impl<K: PolarsNumericType> PrimitiveGroupbySink<K>
where
    ChunkedArray<K>: IntoSeries,
    K::Native: Hash + Serialize + DeserializeOwned,
{
    pub(super) fn sink_ooc(
        &mut self,
        context: &PExecutionContext,
        chunk: DataChunk,
    ) -> PolarsResult<SinkResult> {
        let s = self.prepare_key_and_aggregation_series(context, &chunk)?;
        // cow -> &series -> &dyn series_trait -> &chunkedarray
        let ca: &ChunkedArray<K> = s.as_ref().as_ref();

        // ensure the hashes are set
        s.vec_hash(self.hb.clone(), &mut self.hashes).unwrap();

        let arr = ca.downcast_iter().next().unwrap();
        let pre_agg_len = self.pre_agg_partitions.len();

        // set all bits to false
        self.ooc_state.reset_ooc_filter_rows(ca.len());
        self.ooc_agg_idx.clear();

        // different from standard sink
        // we only set aggregation idx when the entry in the hashmap already
        // exists. This way we don't grow the hashmap
        // rows that are not processed are sinked to disk and loaded in a second pass
        for (iteration_idx, (opt_v, &h)) in arr.iter().zip(self.hashes.iter()).enumerate() {
            let opt_v = opt_v.copied();
            if let Some(agg_idx) =
                try_insert_and_get(h, opt_v, pre_agg_len, &mut self.pre_agg_partitions)
            {
                self.ooc_agg_idx.push(agg_idx);
            } else {
                // set this row to true: e.g. processed ooc
                // safety: we correctly set the length with `reset_ooc_filter_rows`
                unsafe {
                    self.ooc_state.set_row_as_ooc(iteration_idx);
                }
            }
        }

        apply_aggregation(
            &self.ooc_agg_idx,
            &chunk,
            self.number_of_aggs(),
            &self.aggregation_series,
            &self.agg_fns,
            &mut self.aggregators,
        );

        self.aggregation_series.clear();
        self.ooc_state.dump(chunk.data, &mut self.hashes);

        Ok(SinkResult::CanHaveMoreInput)
    }

    fn load_partition_state(&mut self, partition: u32) {
        let iot = self.ooc_state.io_thread.lock().unwrap();
        let mut io_dir = iot.as_ref().unwrap().dir.clone();
        io_dir.push(format!("{partition}_{PARTITION_HASHMAP_STATE_NAME}"));
        if io_dir.exists() {
            let file = std::fs::File::open(io_dir).unwrap();
            let state = bincode::deserialize_from::<_, HashMapState<K::Native>>(file).unwrap();

            let aggs = state.aggregators;
            let pre_agg_len = self.pre_agg_partitions.len();

            for k in state.keys {
                let part = hash_to_partition(k.hash, pre_agg_len);
                let current_partition = unsafe { self.pre_agg_partitions.get_unchecked_release_mut(part) };

                let entry = current_partition
                    .raw_entry_mut()
                    .from_hash(k.hash, |khm| khm.value == k.value);
                match entry {
                    RawEntryMut::Vacant(entry) => {
                        let offset =
                            unsafe { NumCast::from(current_aggregators.len()).unwrap_unchecked_release() };
                        let key = Key {
                            hash: h,
                            value: opt_v,
                        };
                        entry.insert(key, offset);
                        // initialize the aggregators
                        for agg_fn in agg_fns {
                            current_aggregators.push(agg_fn.split2())
                        }
                        offset
                    }
                    RawEntryMut::Occupied(entry) => *entry.get(),
                }
                todo!()
            }

        }
    }

    pub(super) fn pre_finalize_ooc(&mut self) -> PolarsResult<()> {
        dbg!("pre-finalize-ooc");
        let total_len = self
            .pre_agg_partitions
            .iter()
            .map(|map| map.len())
            .sum::<usize>();
        // an estimation
        let cap = total_len * 2 / PARTITION_SIZE;
        let iot = self.ooc_state.io_thread.lock().unwrap();
        let io_dir = iot.as_ref().unwrap().dir.clone();
        drop(iot);

        // we partition the hashmaps into our partitions on disk
        // we create:
        // - a vec with keys
        // - a vec with values
        // these vecs will then be serialized to disk on used later to intialize the state of the groupby
        // sink of that partition
        POOL.install(|| {
            (0..PARTITION_SIZE)
                .into_par_iter()
                .for_each(|partition_idx| {
                    let mut partition_path = io_dir.clone();
                    partition_path.push(format!("{partition_idx}"));

                    // if exists, we must store the state of the hashmaps for this partition
                    if partition_path.exists() {
                        let mut keys = Vec::with_capacity(cap);
                        let mut aggregators = Vec::with_capacity(self.number_of_aggs() * cap);

                        self.pre_agg_partitions.iter().for_each(|agg_map| {
                            agg_map.iter().for_each(|(k, idx)| {
                                if this_partition(k.hash, partition_idx as u64, PARTITION_SIZE as u64) {
                                    keys.push(*k);
                                    let start = *idx as usize;
                                    let end = start + self.number_of_aggs();

                                    // safety: idx is in bounds
                                    unsafe {
                                        aggregators.extend_from_slice(
                                            self.aggregators.get_unchecked(start..end),
                                        )
                                    }
                                }
                            });
                        });
                        let state = HashMapState { keys, aggregators };
                        let ser_state = bincode::serialize(&state).unwrap();
                        drop(state);

                        // we write the partitioned hashmap state
                        // under `groupby/uuid/{partition_idx}_name`
                        // not in the partition directories themselves as there should
                        // only be ipc files there.
                        let mut path = io_dir.clone();
                        path.push(format!("{partition_idx}_{PARTITION_HASHMAP_STATE_NAME}"));

                        let mut file = std::fs::File::create(path).unwrap();
                        file.write_all(&ser_state).unwrap();
                    }
                })
        });
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct HashMapState<T: NumericNative> {
    keys: Vec<Key<Option<T>>>,
    aggregators: Vec<AggregateFunction>,
}

fn try_insert_and_get<T>(
    h: u64,
    opt_v: Option<T>,
    pre_agg_len: usize,
    pre_agg_partitions: &mut Vec<PlIdHashMap<Key<Option<T>>, IdxSize>>,
) -> Option<IdxSize>
where
    T: NumericNative + Hash,
{
    let part = hash_to_partition(h, pre_agg_len);
    let current_partition = unsafe { pre_agg_partitions.get_unchecked_release_mut(part) };

    let entry = current_partition
        .raw_entry_mut()
        .from_hash(h, |k| k.value == opt_v);
    match entry {
        RawEntryMut::Vacant(_) => None,
        RawEntryMut::Occupied(entry) => Some(*entry.get()),
    }
}
