use polars_core::utils::flatten::flatten_par;
use polars_utils::hashing::{DirtyHash, hash_to_partition};
use polars_utils::nulls::IsNull;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

use super::*;

/// Only keeps track of membership in right table
pub(super) fn build_table_semi_anti<T, I>(
    keys: Vec<I>,
    nulls_equal: bool,
) -> Vec<PlHashSet<<T as ToTotalOrd>::TotalOrdItem>>
where
    T: TotalHash + TotalEq + DirtyHash + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Sync + Hash + Eq + DirtyHash + IsNull,
    I: IntoIterator<Item = T> + Copy + Send + Sync,
{
    let n_partitions = _set_partition_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    par_map_collect(n_partitions, &|partition_no| {
        let mut hash_tbl: PlHashSet<T::TotalOrdItem> = PlHashSet::with_capacity(_HASHMAP_INIT_SIZE);
        for keys in &keys {
            keys.into_iter().for_each(|k| {
                let k = k.to_total_ord();
                if partition_no == hash_to_partition(k.dirty_hash(), n_partitions)
                    && (!k.is_null() || nulls_equal)
                {
                    hash_tbl.insert(k);
                }
            });
        }
        hash_tbl
    })
}

/// Per-partition probe indices whose match status equals `keep_matches`
/// (`true` = semi, `false` = anti). Returns `Vec<Vec<IdxSize>>` rather than a
/// `ParallelIterator` so the rayon plumbing isn't monomorphized per `T`/`I`.
fn semi_anti_impl<T, I>(
    probe: Vec<I>,
    build: Vec<I>,
    nulls_equal: bool,
    keep_matches: bool,
) -> Vec<Vec<IdxSize>>
where
    I: IntoIterator<Item = T> + Copy + Send + Sync,
    T: TotalHash + TotalEq + DirtyHash + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Sync + Hash + Eq + DirtyHash + IsNull,
{
    // first we hash one relation
    let hash_sets = build_table_semi_anti(build, nulls_equal);

    // we determine the offset so that we later know which index to store in the join tuples
    let offsets = probe_to_offsets(&probe);

    let n_tables = hash_sets.len();

    // next we probe the other relation
    par_map_collect(probe.len(), &|i| {
        let probe_iter = probe[i].into_iter();
        let offset = offsets[i];

        // assume the result tuples equal length of the no. of hashes processed by this thread.
        let mut results = Vec::with_capacity(probe_iter.size_hint().1.unwrap());

        probe_iter.enumerate().for_each(|(idx_a, k)| {
            let k = k.to_total_ord();
            let idx_a = (idx_a + offset) as IdxSize;
            // probe table that contains the hashed value
            let current_probe_table =
                unsafe { hash_sets.get_unchecked(hash_to_partition(k.dirty_hash(), n_tables)) };

            // we already hashed, so we don't have to hash again.
            if current_probe_table.get(&k).is_some() == keep_matches {
                results.push(idx_a);
            }
        });
        results
    })
}

pub(super) fn hash_join_tuples_left_anti<T, I>(
    probe: Vec<I>,
    build: Vec<I>,
    nulls_equal: bool,
) -> Vec<IdxSize>
where
    I: IntoIterator<Item = T> + Copy + Send + Sync,
    T: TotalHash + TotalEq + DirtyHash + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Sync + Hash + Eq + DirtyHash + IsNull,
{
    let parts = semi_anti_impl(probe, build, nulls_equal, false);
    flatten_par(&parts)
}

pub(super) fn hash_join_tuples_left_semi<T, I>(
    probe: Vec<I>,
    build: Vec<I>,
    nulls_equal: bool,
) -> Vec<IdxSize>
where
    I: IntoIterator<Item = T> + Copy + Send + Sync,
    T: TotalHash + TotalEq + DirtyHash + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Sync + Hash + Eq + DirtyHash + IsNull,
{
    let parts = semi_anti_impl(probe, build, nulls_equal, true);
    flatten_par(&parts)
}
