use std::hash::Hash;

use ahash::RandomState;
use arrow::legacy::trusted_len::TrustedLen;
use arrow::legacy::utils::CustomIterTools;
use hashbrown::hash_map::RawEntryMut;
use hashbrown::HashMap;
use polars_core::hashing::partition::this_partition;
use polars_core::prelude::*;
use polars_core::utils::_set_partition_size;
use polars_core::POOL;
use rayon::prelude::*;

pub(crate) fn prepare_hashed_relation_threaded<T, I>(
    iters: Vec<I>,
) -> Vec<HashMap<T, (bool, Vec<IdxSize>), RandomState>>
where
    I: Iterator<Item = T> + Send + TrustedLen,
    T: Send + Hash + Eq + Sync + Copy,
{
    let n_partitions = _set_partition_size();
    let (hashes_and_keys, build_hasher) = create_hash_and_keys_threaded_vectorized(iters, None);

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions)
            .into_par_iter()
            .map(|partition_no| {
                let build_hasher = build_hasher.clone();
                let hashes_and_keys = &hashes_and_keys;
                let partition_no = partition_no as u64;
                let mut hash_tbl: HashMap<T, (bool, Vec<IdxSize>), RandomState> =
                    HashMap::with_hasher(build_hasher);

                let n_threads = n_partitions as u64;
                let mut offset = 0;
                for hashes_and_keys in hashes_and_keys {
                    let len = hashes_and_keys.len();
                    hashes_and_keys
                        .iter()
                        .enumerate()
                        .for_each(|(idx, (h, k))| {
                            let idx = idx as IdxSize;
                            // partition hashes by thread no.
                            // So only a part of the hashes go to this hashmap
                            if this_partition(*h, partition_no, n_threads) {
                                let idx = idx + offset;
                                let entry = hash_tbl
                                    .raw_entry_mut()
                                    // uses the key to check equality to find and entry
                                    .from_key_hashed_nocheck(*h, k);

                                match entry {
                                    RawEntryMut::Vacant(entry) => {
                                        entry.insert_hashed_nocheck(*h, *k, (false, vec![idx]));
                                    },
                                    RawEntryMut::Occupied(mut entry) => {
                                        let (_k, v) = entry.get_key_value_mut();
                                        v.1.push(idx);
                                    },
                                }
                            }
                        });

                    offset += len as IdxSize;
                }
                hash_tbl
            })
            .collect()
    })
}

pub(crate) fn create_hash_and_keys_threaded_vectorized<I, T>(
    iters: Vec<I>,
    build_hasher: Option<RandomState>,
) -> (Vec<Vec<(u64, T)>>, RandomState)
where
    I: IntoIterator<Item = T> + Send,
    I::IntoIter: TrustedLen,
    T: Send + Hash + Eq,
{
    let build_hasher = build_hasher.unwrap_or_default();
    let hashes = POOL.install(|| {
        iters
            .into_par_iter()
            .map(|iter| {
                // create hashes and keys
                iter.into_iter()
                    .map(|val| (build_hasher.hash_one(&val), val))
                    .collect_trusted::<Vec<_>>()
            })
            .collect()
    });
    (hashes, build_hasher)
}
