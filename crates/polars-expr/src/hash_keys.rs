use arrow::array::{BinaryArray, PrimitiveArray, UInt64Array};
use arrow::compute::utils::combine_validities_and_many;
use polars_compute::gather::binary::take_unchecked;
use polars_core::frame::DataFrame;
use polars_core::prelude::row_encode::_get_rows_encoded_unordered;
use polars_core::prelude::PlRandomState;
use polars_core::series::Series;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::index::ChunkId;
use polars_utils::itertools::Itertools;
use polars_utils::vec::PushUnchecked;
use polars_utils::IdxSize;

/// Represents a DataFrame plus a hash per row, intended for keys in grouping
/// or joining. The hashes may or may not actually be physically pre-computed,
/// this depends per type.
#[derive(Clone, Debug)]
pub enum HashKeys {
    RowEncoded(RowEncodedKeys),
    Single(SingleKeys),
}

impl HashKeys {
    pub fn from_df(
        df: &DataFrame,
        random_state: PlRandomState,
        null_is_valid: bool,
        force_row_encoding: bool,
    ) -> Self {
        if df.width() > 1 || force_row_encoding {
            let keys = df.get_columns();
            let mut keys_encoded = _get_rows_encoded_unordered(keys).unwrap().into_array();

            if !null_is_valid {
                let validities = keys
                    .iter()
                    .map(|c| c.as_materialized_series().rechunk_validity())
                    .collect_vec();
                let combined = combine_validities_and_many(&validities);
                keys_encoded.set_validity(combined);
            }

            // TODO: use vechash? Not supported yet for lists.
            // let mut hashes = Vec::with_capacity(df.height());
            // columns_to_hashes(df.get_columns(), Some(random_state), &mut hashes).unwrap();

            let hashes = keys_encoded
                .values_iter()
                .map(|k| random_state.hash_one(k))
                .collect();
            Self::RowEncoded(RowEncodedKeys {
                hashes: PrimitiveArray::from_vec(hashes),
                keys: keys_encoded,
            })
        } else {
            todo!()
            // Self::Single(SingleKeys {
            //     random_state,
            //     hashes: todo!(),
            //     keys: df[0].as_materialized_series().clone(),
            // })
        }
    }

    pub fn len(&self) -> usize {
        match self {
            HashKeys::RowEncoded(s) => s.keys.len(),
            HashKeys::Single(s) => s.keys.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// After this call partition_idxs[p] will contain the indices of hashes
    /// that belong to partition p, and the cardinality sketches are updated
    /// accordingly.
    pub fn gen_partition_idxs(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
        sketches: &mut [CardinalitySketch],
        partition_nulls: bool,
    ) {
        if sketches.is_empty() {
            match self {
                Self::RowEncoded(s) => s.gen_partition_idxs::<false>(
                    partitioner,
                    partition_idxs,
                    sketches,
                    partition_nulls,
                ),
                Self::Single(s) => s.gen_partition_idxs::<false>(
                    partitioner,
                    partition_idxs,
                    sketches,
                    partition_nulls,
                ),
            }
        } else {
            match self {
                Self::RowEncoded(s) => s.gen_partition_idxs::<true>(
                    partitioner,
                    partition_idxs,
                    sketches,
                    partition_nulls,
                ),
                Self::Single(s) => s.gen_partition_idxs::<true>(
                    partitioner,
                    partition_idxs,
                    sketches,
                    partition_nulls,
                ),
            }
        }
    }

    /// Generates indices for a chunked gather such that the ith key gathers
    /// the next gathers_per_key[i] elements from the partition[i]th chunk.
    pub fn gen_partitioned_gather_idxs(
        &self,
        partitioner: &HashPartitioner,
        gathers_per_key: &[IdxSize],
        gather_idxs: &mut Vec<ChunkId<32>>,
    ) {
        match self {
            Self::RowEncoded(s) => {
                s.gen_partitioned_gather_idxs(partitioner, gathers_per_key, gather_idxs)
            },
            Self::Single(s) => {
                s.gen_partitioned_gather_idxs(partitioner, gathers_per_key, gather_idxs)
            },
        }
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn gather(&self, idxs: &[IdxSize]) -> Self {
        match self {
            Self::RowEncoded(s) => Self::RowEncoded(s.gather(idxs)),
            Self::Single(s) => Self::Single(s.gather(idxs)),
        }
    }

    pub fn sketch_cardinality(&self, sketch: &mut CardinalitySketch) {
        match self {
            HashKeys::RowEncoded(s) => s.sketch_cardinality(sketch),
            HashKeys::Single(s) => s.sketch_cardinality(sketch),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RowEncodedKeys {
    pub hashes: UInt64Array,
    pub keys: BinaryArray<i64>,
}

impl RowEncodedKeys {
    pub fn gen_partition_idxs<const BUILD_SKETCHES: bool>(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
        sketches: &mut [CardinalitySketch],
        partition_nulls: bool,
    ) {
        assert!(partition_idxs.len() == partitioner.num_partitions());
        assert!(!BUILD_SKETCHES || sketches.len() == partitioner.num_partitions());
        for p in partition_idxs.iter_mut() {
            p.clear();
        }

        if let Some(validity) = self.keys.validity() {
            for (i, (h, is_v)) in self.hashes.values_iter().zip(validity).enumerate() {
                if is_v {
                    unsafe {
                        // SAFETY: we assured the number of partitions matches.
                        let p = partitioner.hash_to_partition(*h);
                        partition_idxs.get_unchecked_mut(p).push(i as IdxSize);
                        if BUILD_SKETCHES {
                            sketches.get_unchecked_mut(p).insert(*h);
                        }
                    }
                } else if partition_nulls {
                    // Arbitrarily put nulls in partition 0.
                    unsafe {
                        partition_idxs.get_unchecked_mut(0).push(i as IdxSize);
                    }
                }
            }
        } else {
            for (i, h) in self.hashes.values_iter().enumerate() {
                unsafe {
                    // SAFETY: we assured the number of partitions matches.
                    let p = partitioner.hash_to_partition(*h);
                    partition_idxs.get_unchecked_mut(p).push(i as IdxSize);
                    if BUILD_SKETCHES {
                        sketches.get_unchecked_mut(p).insert(*h);
                    }
                }
            }
        }
    }

    pub fn gen_partitioned_gather_idxs(
        &self,
        partitioner: &HashPartitioner,
        gathers_per_key: &[IdxSize],
        gather_idxs: &mut Vec<ChunkId<32>>,
    ) {
        assert!(gathers_per_key.len() == self.keys.len());
        unsafe {
            let mut offsets = vec![0; partitioner.num_partitions()];
            for (hash, &n) in self.hashes.values_iter().zip(gathers_per_key) {
                let p = partitioner.hash_to_partition(*hash);
                let offset = *offsets.get_unchecked(p);
                for i in offset..offset + n {
                    gather_idxs.push(ChunkId::store(p as IdxSize, i));
                }
                *offsets.get_unchecked_mut(p) += n;
            }
        }
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn gather(&self, idxs: &[IdxSize]) -> Self {
        let mut hashes = Vec::with_capacity(idxs.len());
        for idx in idxs {
            hashes.push_unchecked(*self.hashes.values().get_unchecked(*idx as usize));
        }
        let idx_arr = arrow::ffi::mmap::slice(idxs);
        let keys = take_unchecked(&self.keys, &idx_arr);
        Self {
            hashes: PrimitiveArray::from_vec(hashes),
            keys,
        }
    }

    pub fn sketch_cardinality(&self, sketch: &mut CardinalitySketch) {
        if let Some(validity) = self.keys.validity() {
            for (h, is_v) in self.hashes.values_iter().zip(validity) {
                if is_v {
                    sketch.insert(*h);
                }
            }
        } else {
            for h in self.hashes.values_iter() {
                sketch.insert(*h);
            }
        }
    }
}

/// Single keys. Does not pre-hash for boolean & integer types, only for strings
/// and nested types.
#[derive(Clone, Debug)]
pub struct SingleKeys {
    pub random_state: PlRandomState,
    pub hashes: Option<Vec<u64>>,
    pub keys: Series,
}

impl SingleKeys {
    pub fn gen_partition_idxs<const BUILD_SKETCHES: bool>(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
        _sketches: &mut [CardinalitySketch],
        _partition_nulls: bool,
    ) {
        assert!(partitioner.num_partitions() == partition_idxs.len());
        for p in partition_idxs.iter_mut() {
            p.clear();
        }

        todo!()
    }

    #[allow(clippy::ptr_arg)] // Remove when implemented.
    pub fn gen_partitioned_gather_idxs(
        &self,
        _partitioner: &HashPartitioner,
        _gathers_per_key: &[IdxSize],
        _gather_idxs: &mut Vec<ChunkId<32>>,
    ) {
        todo!()
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn gather(&self, idxs: &[IdxSize]) -> Self {
        let hashes = self.hashes.as_ref().map(|hashes| {
            let mut out = Vec::with_capacity(idxs.len());
            for idx in idxs {
                out.push_unchecked(*hashes.get_unchecked(*idx as usize));
            }
            out
        });
        Self {
            random_state: self.random_state.clone(),
            hashes,
            keys: self.keys.take_slice_unchecked(idxs),
        }
    }

    pub fn sketch_cardinality(&self, _sketch: &mut CardinalitySketch) {
        todo!()
    }
}
