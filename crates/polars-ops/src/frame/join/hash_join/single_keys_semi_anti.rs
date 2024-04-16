use polars_utils::hashing::{hash_to_partition, DirtyHash};
use polars_utils::nulls::IsNull;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

use super::*;

/// Only keeps track of membership in right table
pub(super) fn build_table_semi_anti<T, I>(
    keys: Vec<I>,
    join_nulls: bool,
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
    let par_iter = (0..n_partitions).into_par_iter().map(|partition_no| {
        let mut hash_tbl: PlHashSet<T::TotalOrdItem> = PlHashSet::with_capacity(_HASHMAP_INIT_SIZE);
        for keys in &keys {
            keys.into_iter().for_each(|k| {
                let k = k.to_total_ord();
                if partition_no == hash_to_partition(k.dirty_hash(), n_partitions)
                    && (!k.is_null() || join_nulls)
                {
                    hash_tbl.insert(k);
                }
            });
        }
        hash_tbl
    });
    POOL.install(|| par_iter.collect())
}

/// Construct a ParallelIterator, but doesn't iterate it. This means the caller
/// context (or wherever it gets iterated) should be in POOL.install.
fn semi_anti_impl<T, I>(
    probe: Vec<I>,
    build: Vec<I>,
    join_nulls: bool,
) -> impl ParallelIterator<Item = (IdxSize, bool)>
where
    I: IntoIterator<Item = T> + Copy + Send + Sync,
    T: TotalHash + TotalEq + DirtyHash + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Sync + Hash + Eq + DirtyHash + IsNull,
{
    // first we hash one relation
    let hash_sets = build_table_semi_anti(build, join_nulls);

    // we determine the offset so that we later know which index to store in the join tuples
    let offsets = probe_to_offsets(&probe);

    let n_tables = hash_sets.len();

    // next we probe the other relation
    // This is not wrapped in POOL.install because it is not being iterated here
    probe
        .into_par_iter()
        .zip(offsets)
        // probes_hashes: Vec<u64> processed by this thread
        // offset: offset index
        .flat_map(move |(probe, offset)| {
            // local reference
            let hash_sets = &hash_sets;
            let probe_iter = probe.into_iter();

            // assume the result tuples equal length of the no. of hashes processed by this thread.
            let mut results = Vec::with_capacity(probe_iter.size_hint().1.unwrap());

            probe_iter.enumerate().for_each(|(idx_a, k)| {
                let k = k.to_total_ord();
                let idx_a = (idx_a + offset) as IdxSize;
                // probe table that contains the hashed value
                let current_probe_table =
                    unsafe { hash_sets.get_unchecked(hash_to_partition(k.dirty_hash(), n_tables)) };

                // we already hashed, so we don't have to hash again.
                let value = current_probe_table.get(&k);

                match value {
                    // left and right matches
                    Some(_) => results.push((idx_a, true)),
                    // only left values, right = null
                    None => results.push((idx_a, false)),
                }
            });
            results
        })
}

pub(super) fn hash_join_tuples_left_anti<T, I>(
    probe: Vec<I>,
    build: Vec<I>,
    join_nulls: bool,
) -> Vec<IdxSize>
where
    I: IntoIterator<Item = T> + Copy + Send + Sync,
    T: TotalHash + TotalEq + DirtyHash + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Sync + Hash + Eq + DirtyHash + IsNull,
{
    let par_iter = semi_anti_impl(probe, build, join_nulls)
        .filter(|tpls| !tpls.1)
        .map(|tpls| tpls.0);
    POOL.install(|| par_iter.collect())
}

pub(super) fn hash_join_tuples_left_semi<T, I>(
    probe: Vec<I>,
    build: Vec<I>,
    join_nulls: bool,
) -> Vec<IdxSize>
where
    I: IntoIterator<Item = T> + Copy + Send + Sync,
    T: TotalHash + TotalEq + DirtyHash + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Sync + Hash + Eq + DirtyHash + IsNull,
{
    let par_iter = semi_anti_impl(probe, build, join_nulls)
        .filter(|tpls| tpls.1)
        .map(|tpls| tpls.0);
    POOL.install(|| par_iter.collect())
}
