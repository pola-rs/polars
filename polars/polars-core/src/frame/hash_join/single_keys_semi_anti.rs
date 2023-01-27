use super::*;

/// Only keeps track of membership in right table
pub(super) fn create_probe_table_semi_anti<T, IntoSlice>(keys: Vec<IntoSlice>) -> Vec<PlHashSet<T>>
where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    IntoSlice: AsRef<[T]> + Send + Sync,
{
    let n_partitions = _set_partition_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions).into_par_iter().map(|partition_no| {
            let partition_no = partition_no as u64;

            let mut hash_tbl: PlHashSet<T> = PlHashSet::with_capacity(HASHMAP_INIT_SIZE);

            let n_partitions = n_partitions as u64;
            for keys in &keys {
                let keys = keys.as_ref();
                keys.iter().for_each(|k| {
                    if this_partition(k.as_u64(), partition_no, n_partitions) {
                        hash_tbl.insert(*k);
                    }
                });
            }
            hash_tbl
        })
    })
    .collect()
}

pub(super) fn semi_anti_impl<T, IntoSlice>(
    probe: Vec<IntoSlice>,
    build: Vec<IntoSlice>,
) -> impl ParallelIterator<Item = (IdxSize, bool)>
where
    IntoSlice: AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    // first we hash one relation
    let hash_sets = create_probe_table_semi_anti(build);

    // we determine the offset so that we later know which index to store in the join tuples
    let offsets = probe_to_offsets(&probe);

    let n_tables = hash_sets.len() as u64;
    debug_assert!(n_tables.is_power_of_two());

    // next we probe the other relation
    POOL.install(move || {
        probe
            .into_par_iter()
            .zip(offsets)
            // probes_hashes: Vec<u64> processed by this thread
            // offset: offset index
            .flat_map(move |(probe, offset)| {
                // local reference
                let hash_sets = &hash_sets;
                let probe = probe.as_ref();

                // assume the result tuples equal length of the no. of hashes processed by this thread.
                let mut results = Vec::with_capacity(probe.len());

                probe.iter().enumerate().for_each(|(idx_a, k)| {
                    let idx_a = (idx_a + offset) as IdxSize;
                    // probe table that contains the hashed value
                    let current_probe_table = unsafe {
                        get_hash_tbl_threaded_join_partitioned(k.as_u64(), hash_sets, n_tables)
                    };

                    // we already hashed, so we don't have to hash again.
                    let value = current_probe_table.get(k);

                    match value {
                        // left and right matches
                        Some(_) => results.push((idx_a, true)),
                        // only left values, right = null
                        None => results.push((idx_a, false)),
                    }
                });
                results
            })
    })
}

pub(super) fn hash_join_tuples_left_anti<T, IntoSlice>(
    probe: Vec<IntoSlice>,
    build: Vec<IntoSlice>,
) -> Vec<IdxSize>
where
    IntoSlice: AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    semi_anti_impl(probe, build)
        .filter(|tpls| !tpls.1)
        .map(|tpls| tpls.0)
        .collect()
}

pub(super) fn hash_join_tuples_left_semi<T, IntoSlice>(
    probe: Vec<IntoSlice>,
    build: Vec<IntoSlice>,
) -> Vec<IdxSize>
where
    IntoSlice: AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    semi_anti_impl(probe, build)
        .filter(|tpls| tpls.1)
        .map(|tpls| tpls.0)
        .collect()
}
