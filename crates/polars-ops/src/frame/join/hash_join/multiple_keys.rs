use hashbrown::HashMap;
use polars_core::hashing::{populate_multiple_key_hashmap, IdBuildHasher, IdxHash};
use polars_utils::hashing::hash_to_partition;
use polars_utils::idx_vec::IdxVec;
use polars_utils::unitvec;

use super::*;

/// Compare the rows of two [`DataFrame`]s
pub(crate) unsafe fn compare_df_rows2(
    left: &DataFrame,
    right: &DataFrame,
    left_idx: usize,
    right_idx: usize,
    join_nulls: bool,
) -> bool {
    for (l, r) in left.get_columns().iter().zip(right.get_columns()) {
        let l = l.get_unchecked(left_idx);
        let r = r.get_unchecked(right_idx);
        if !l.eq_missing(&r, join_nulls) {
            return false;
        }
    }
    true
}

pub(crate) fn create_probe_table(
    hashes: &[UInt64Chunked],
    keys: &DataFrame,
) -> Vec<HashMap<IdxHash, IdxVec, IdBuildHasher>> {
    let n_partitions = _set_partition_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions)
            .into_par_iter()
            .map(|part_no| {
                let mut hash_tbl: HashMap<IdxHash, IdxVec, IdBuildHasher> =
                    HashMap::with_capacity_and_hasher(_HASHMAP_INIT_SIZE, Default::default());

                let mut offset = 0;
                for hashes in hashes {
                    for hashes in hashes.data_views() {
                        let len = hashes.len();
                        let mut idx = 0;
                        hashes.iter().for_each(|h| {
                            // partition hashes by thread no.
                            // So only a part of the hashes go to this hashmap
                            if part_no == hash_to_partition(*h, n_partitions) {
                                let idx = idx + offset;
                                populate_multiple_key_hashmap(
                                    &mut hash_tbl,
                                    idx,
                                    *h,
                                    keys,
                                    || unitvec![idx],
                                    |v| v.push(idx),
                                )
                            }
                            idx += 1;
                        });

                        offset += len as IdxSize;
                    }
                }
                hash_tbl
            })
            .collect()
    })
}

pub(crate) fn get_offsets(probe_hashes: &[UInt64Chunked]) -> Vec<usize> {
    probe_hashes
        .iter()
        .map(|ph| ph.len())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect()
}
