use arrow::array::BinaryArray;
use arrow::compute::take::binary::take_unchecked;
use arrow::compute::utils::combine_validities_and_many;
use polars_core::frame::DataFrame;
use polars_core::prelude::row_encode::_get_rows_encoded_unordered;
use polars_core::prelude::PlRandomState;
use polars_core::series::Series;
use polars_utils::hashing::HashPartitioner;
use polars_utils::itertools::Itertools;
use polars_utils::vec::PushUnchecked;
use polars_utils::IdxSize;

/// Represents a DataFrame plus a hash per row, intended for keys in grouping
/// or joining. The hashes may or may not actually be physically pre-computed,
/// this depends per type.
pub enum HashKeys {
    RowEncoded(RowEncodedKeys),
    Single(SingleKeys),
}

impl HashKeys {
    pub fn from_df(df: &DataFrame, random_state: PlRandomState, null_is_valid: bool, force_row_encoding: bool) -> Self {
        if df.width() > 1 || force_row_encoding {
            let keys = df
                .get_columns()
                .iter()
                .map(|c| c.as_materialized_series().clone())
                .collect_vec();
            let mut keys_encoded = _get_rows_encoded_unordered(&keys[..]).unwrap().into_array();
            
            if !null_is_valid {
                let validities = keys.iter().map(|c| c.rechunk_validity()).collect_vec();
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
                hashes,
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

    pub fn gen_partition_idxs(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
    ) {
        match self {
            Self::RowEncoded(s) => s.gen_partition_idxs(partitioner, partition_idxs),
            Self::Single(s) => s.gen_partition_idxs(partitioner, partition_idxs),
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
}

pub struct RowEncodedKeys {
    pub hashes: Vec<u64>,
    pub keys: BinaryArray<i64>,
}

impl RowEncodedKeys {
    pub fn gen_partition_idxs(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
    ) {
        assert!(partitioner.num_partitions() == partition_idxs.len());
        for (i, h) in self.hashes.iter().enumerate() {
            unsafe {
                // SAFETY: we assured the number of partitions matches.
                let p = partitioner.hash_to_partition(*h);
                partition_idxs.get_unchecked_mut(p).push(i as IdxSize);
            }
        }
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn gather(&self, idxs: &[IdxSize]) -> Self {
        let mut hashes = Vec::with_capacity(idxs.len());
        for idx in idxs {
            hashes.push_unchecked(*self.hashes.get_unchecked(*idx as usize));
        }
        let idx_arr = arrow::ffi::mmap::slice(idxs);
        let keys = take_unchecked(&self.keys, &idx_arr);
        Self { hashes, keys }
    }
}

/// Single keys. Does not pre-hash for boolean & integer types, only for strings
/// and nested types.
pub struct SingleKeys {
    pub random_state: PlRandomState,
    pub hashes: Option<Vec<u64>>,
    pub keys: Series,
}

impl SingleKeys {
    pub fn gen_partition_idxs(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
    ) {
        assert!(partitioner.num_partitions() == partition_idxs.len());
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
}
