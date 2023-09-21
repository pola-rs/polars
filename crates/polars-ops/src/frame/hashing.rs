use arrow::{
    bitmap::utils::get_bit_unchecked,
    array::{
        Array, BinaryArray
    }
};
use hashbrown::hash_map::RawEntryMut;
use hashbrown::HashMap;
#[cfg(feature = "group_by_list")]
use polars_arrow::kernels::list_bytes_iter::numeric_list_bytes_iter;
use polars_arrow::utils::CustomIterTools;
use rayon::prelude::*;
use xxhash_rust::xxh3::xxh3_64_with_seed;

use ahash::RandomState;
use polars_core::prelude::*;
use polars_core::POOL;
use std::hash::{Hash, Hasher, BuildHasher};
use polars_arrow::trusted_len::TrustedLen;
use polars_core::hashing::_boost_hash_combine;
use polars_core::hashing::partition::{this_partition, AsU64};

pub(crate) fn prepare_hashed_relation_threaded<T, I>(
    iters: Vec<I>,
) -> Vec<HashMap<T, (bool, Vec<IdxSize>), RandomState>>
    where
        I: Iterator<Item = T> + Send + TrustedLen,
        T: Send + Hash + Eq + Sync + Copy,
{
    let n_partitions = iters.len();
    assert!(n_partitions.is_power_of_two());

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
                    .map(|val| {
                        let mut hasher = build_hasher.build_hasher();
                        val.hash(&mut hasher);
                        (hasher.finish(), val)
                    })
                    .collect_trusted::<Vec<_>>()
            })
            .collect()
    });
    (hashes, build_hasher)
}

pub(crate) fn series_to_hashes(
    keys: &[Series],
    build_hasher: Option<RandomState>,
    hashes: &mut Vec<u64>,
) -> PolarsResult<RandomState> {
    let build_hasher = build_hasher.unwrap_or_default();

    let mut iter = keys.iter();
    let first = iter.next().expect("at least one key");
    first.vec_hash(build_hasher.clone(), hashes)?;

    for keys in iter {
        keys.vec_hash_combine(build_hasher.clone(), hashes)?;
    }

    Ok(build_hasher)
}
