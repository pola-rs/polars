use super::*;

/// Probe the build table and add tuples to the results (inner join)
pub(super) fn probe_inner<T, F>(
    probe: &[T],
    hash_tbls: &[PlHashMap<T, Vec<IdxSize>>],
    results: &mut Vec<(IdxSize, IdxSize)>,
    local_offset: usize,
    n_tables: u64,
    swap_fn: F,
) where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    F: Fn(IdxSize, IdxSize) -> (IdxSize, IdxSize),
{
    assert!(hash_tbls.len().is_power_of_two());
    probe.iter().enumerate().for_each(|(idx_a, k)| {
        let idx_a = (idx_a + local_offset) as IdxSize;
        // probe table that contains the hashed value
        let current_probe_table =
            unsafe { get_hash_tbl_threaded_join_partitioned(k.as_u64(), hash_tbls, n_tables) };

        let value = current_probe_table.get(k);

        if let Some(indexes_b) = value {
            let tuples = indexes_b.iter().map(|&idx_b| swap_fn(idx_a, idx_b));
            results.extend(tuples);
        }
    });
}

pub(super) fn hash_join_tuples_inner<T, IntoSlice>(
    probe: Vec<IntoSlice>,
    build: Vec<IntoSlice>,
    // Because b should be the shorter relation we could need to swap to keep left left and right right.
    swap: bool,
) -> (Vec<IdxSize>, Vec<IdxSize>)
where
    IntoSlice: AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    // NOTE: see the left join for more elaborate comments

    // first we hash one relation
    let hash_tbls = create_probe_table(build);

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());
    let offsets = probe_to_offsets(&probe);
    // next we probe the other relation
    // code duplication is because we want to only do the swap check once
    POOL.install(|| {
        probe
            .into_par_iter()
            .zip(offsets)
            .flat_map(|(probe, offset)| {
                let probe = probe.as_ref();
                // local reference
                let hash_tbls = &hash_tbls;
                let mut results = Vec::with_capacity(probe.len());
                let local_offset = offset;

                // branch is to hoist swap out of the inner loop.
                if swap {
                    probe_inner(
                        probe,
                        hash_tbls,
                        &mut results,
                        local_offset,
                        n_tables,
                        |idx_a, idx_b| (idx_b, idx_a),
                    )
                } else {
                    probe_inner(
                        probe,
                        hash_tbls,
                        &mut results,
                        local_offset,
                        n_tables,
                        |idx_a, idx_b| (idx_a, idx_b),
                    )
                }

                results
            })
            .unzip()
    })
}
