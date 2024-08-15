use arrow::array::{MutablePrimitiveArray, PrimitiveArray};
use arrow::legacy::utils::CustomIterTools;
use polars_utils::hashing::hash_to_partition;
use polars_utils::idx_vec::IdxVec;
use polars_utils::nulls::IsNull;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};
use polars_utils::unitvec;

use super::*;

pub(crate) fn create_hash_and_keys_threaded_vectorized<I, T>(
    iters: Vec<I>,
    build_hasher: Option<PlRandomState>,
) -> (Vec<Vec<(u64, T)>>, PlRandomState)
where
    I: IntoIterator<Item = T> + Send,
    I::IntoIter: TrustedLen,
    T: TotalHash + TotalEq + Send + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Hash + Eq,
{
    let build_hasher = build_hasher.unwrap_or_default();
    let hashes = POOL.install(|| {
        iters
            .into_par_iter()
            .map(|iter| {
                // create hashes and keys
                #[allow(clippy::needless_borrows_for_generic_args)]
                iter.into_iter()
                    .map(|val| (build_hasher.hash_one(&val.to_total_ord()), val))
                    .collect_trusted::<Vec<_>>()
            })
            .collect()
    });
    (hashes, build_hasher)
}

pub(crate) fn prepare_hashed_relation_threaded<T, I>(
    iters: Vec<I>,
) -> Vec<PlHashMap<<T as ToTotalOrd>::TotalOrdItem, (bool, IdxVec)>>
where
    I: Iterator<Item = T> + Send + TrustedLen,
    T: Send + Sync + TotalHash + TotalEq + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Sync + Hash + Eq,
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
                let mut hash_tbl: PlHashMap<T::TotalOrdItem, (bool, IdxVec)> =
                    PlHashMap::with_hasher(build_hasher);

                let mut offset = 0;
                for hashes_and_keys in hashes_and_keys {
                    let len = hashes_and_keys.len();
                    hashes_and_keys
                        .iter()
                        .enumerate()
                        .for_each(|(idx, (h, k))| {
                            let k = k.to_total_ord();
                            let idx = idx as IdxSize;
                            // partition hashes by thread no.
                            // So only a part of the hashes go to this hashmap
                            if partition_no == hash_to_partition(*h, n_partitions) {
                                let idx = idx + offset;
                                let entry = hash_tbl
                                    .raw_entry_mut()
                                    // uses the key to check equality to find and entry
                                    .from_key_hashed_nocheck(*h, &k);

                                match entry {
                                    RawEntryMut::Vacant(entry) => {
                                        entry.insert_hashed_nocheck(*h, k, (false, unitvec![idx]));
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

/// Probe the build table and add tuples to the results.
#[allow(clippy::too_many_arguments)]
fn probe_outer<T, F, G, H>(
    probe_hashes: &[Vec<(u64, T)>],
    hash_tbls: &mut [PlHashMap<<T as ToTotalOrd>::TotalOrdItem, (bool, IdxVec)>],
    results: &mut (
        MutablePrimitiveArray<IdxSize>,
        MutablePrimitiveArray<IdxSize>,
    ),
    n_tables: usize,
    // Function that get index_a, index_b when there is a match and pushes to result
    swap_fn_match: F,
    // Function that get index_a when there is no match and pushes to result
    swap_fn_no_match: G,
    // Function that get index_b from the build table that did not match any in A and pushes to result
    swap_fn_drain: H,
    join_nulls: bool,
) where
    T: TotalHash + TotalEq + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Hash + Eq + IsNull,
    // idx_a, idx_b -> ...
    F: Fn(IdxSize, IdxSize) -> (Option<IdxSize>, Option<IdxSize>),
    // idx_a -> ...
    G: Fn(IdxSize) -> (Option<IdxSize>, Option<IdxSize>),
    // idx_b -> ...
    H: Fn(IdxSize) -> (Option<IdxSize>, Option<IdxSize>),
{
    // needed for the partition shift instead of modulo to make sense
    let mut idx_a = 0;
    for probe_hashes in probe_hashes {
        for (h, key) in probe_hashes {
            let key = key.to_total_ord();
            let h = *h;
            // probe table that contains the hashed value
            let current_probe_table =
                unsafe { hash_tbls.get_unchecked_mut(hash_to_partition(h, n_tables)) };

            let entry = current_probe_table
                .raw_entry_mut()
                .from_key_hashed_nocheck(h, &key);

            match entry {
                // match and remove
                RawEntryMut::Occupied(mut occupied) => {
                    if key.is_null() && !join_nulls {
                        let (l, r) = swap_fn_no_match(idx_a);
                        results.0.push(l);
                        results.1.push(r);
                    } else {
                        let (tracker, indexes_b) = occupied.get_mut();
                        *tracker = true;
                        for (l, r) in indexes_b.iter().map(|&idx_b| swap_fn_match(idx_a, idx_b)) {
                            results.0.push(l);
                            results.1.push(r);
                        }
                    }
                },
                // no match
                RawEntryMut::Vacant(_) => {
                    let (l, r) = swap_fn_no_match(idx_a);
                    results.0.push(l);
                    results.1.push(r);
                },
            }
            idx_a += 1;
        }
    }

    for hash_tbl in hash_tbls {
        hash_tbl.iter().for_each(|(_k, (tracker, indexes_b))| {
            // remaining joined values from the right table
            if !*tracker {
                for (l, r) in indexes_b.iter().map(|&idx_b| swap_fn_drain(idx_b)) {
                    results.0.push(l);
                    results.1.push(r);
                }
            }
        });
    }
}

/// Hash join outer. Both left and right can have no match so Options
pub(super) fn hash_join_tuples_outer<T, I, J>(
    probe: Vec<I>,
    build: Vec<J>,
    swapped: bool,
    validate: JoinValidation,
    join_nulls: bool,
) -> PolarsResult<(PrimitiveArray<IdxSize>, PrimitiveArray<IdxSize>)>
where
    I: IntoIterator<Item = T>,
    J: IntoIterator<Item = T>,
    <J as IntoIterator>::IntoIter: TrustedLen + Send,
    <I as IntoIterator>::IntoIter: TrustedLen + Send,
    T: Send + Sync + TotalHash + TotalEq + IsNull + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Sync + Hash + Eq + IsNull,
{
    let probe = probe.into_iter().map(|i| i.into_iter()).collect::<Vec<_>>();
    let build = build.into_iter().map(|i| i.into_iter()).collect::<Vec<_>>();
    // This function is partially multi-threaded.
    // Parts that are done in parallel:
    //  - creation of the probe tables
    //  - creation of the hashes

    // during the probe phase values are removed from the tables, that's done single threaded to
    // keep it lock free.

    let size = probe
        .iter()
        .map(|a| a.size_hint().1.unwrap())
        .sum::<usize>()
        + build
            .iter()
            .map(|b| b.size_hint().1.unwrap())
            .sum::<usize>();
    let mut results = (
        MutablePrimitiveArray::with_capacity(size),
        MutablePrimitiveArray::with_capacity(size),
    );

    // prepare hash table
    let mut hash_tbls = if validate.needs_checks() {
        let expected_size = build.iter().map(|i| i.size_hint().0).sum();
        let hash_tbls = prepare_hashed_relation_threaded(build);
        let build_size = hash_tbls.iter().map(|m| m.len()).sum();
        validate.validate_build(build_size, expected_size, swapped)?;
        hash_tbls
    } else {
        prepare_hashed_relation_threaded(build)
    };
    let random_state = hash_tbls[0].hasher().clone();

    // we pre hash the probing values
    let (probe_hashes, _) = create_hash_and_keys_threaded_vectorized(probe, Some(random_state));

    let n_tables = hash_tbls.len();

    // probe the hash table.
    // Note: indexes from b that are not matched will be None, Some(idx_b)
    // Therefore we remove the matches and the remaining will be joined from the right

    // branch is because we want to only do the swap check once
    if swapped {
        probe_outer(
            &probe_hashes,
            &mut hash_tbls,
            &mut results,
            n_tables,
            |idx_a, idx_b| (Some(idx_b), Some(idx_a)),
            |idx_a| (None, Some(idx_a)),
            |idx_b| (Some(idx_b), None),
            join_nulls,
        )
    } else {
        probe_outer(
            &probe_hashes,
            &mut hash_tbls,
            &mut results,
            n_tables,
            |idx_a, idx_b| (Some(idx_a), Some(idx_b)),
            |idx_a| (Some(idx_a), None),
            |idx_b| (None, Some(idx_b)),
            join_nulls,
        )
    }
    Ok((results.0.into(), results.1.into()))
}
