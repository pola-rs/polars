use super::*;

pub(super) type LeftJoinIndices = (IdxSize, Option<IdxSize>);

#[inline]
pub(super) fn on_match_left_join_extend(
    results: &mut Vec<LeftJoinIndices>,
    indexes_b: &[IdxSize],
    idx_a: IdxSize,
) {
    results.extend(indexes_b.iter().map(|&idx_b| (idx_a, Some(idx_b))))
}

pub(super) fn hash_join_tuples_left<T, IntoSlice>(
    probe: Vec<IntoSlice>,
    build: Vec<IntoSlice>,
) -> Vec<LeftJoinIndices>
where
    IntoSlice: AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    // first we hash one relation
    let hash_tbls = create_probe_table(build);

    // we determine the offset so that we later know which index to store in the join tuples
    let offsets = probe_to_offsets(&probe);

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());

    // next we probe the other relation
    POOL.install(move || {
        probe
            .into_par_iter()
            .zip(offsets)
            // probes_hashes: Vec<u64> processed by this thread
            // offset: offset index
            .map(move |(probe, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;
                let probe = probe.as_ref();

                // assume the result tuples equal length of the no. of hashes processed by this thread.
                let mut results = Vec::with_capacity(probe.len());

                probe.iter().enumerate().for_each(|(idx_a, k)| {
                    let idx_a = (idx_a + offset) as IdxSize;
                    // probe table that contains the hashed value
                    let current_probe_table = unsafe {
                        get_hash_tbl_threaded_join_partitioned(k.as_u64(), hash_tbls, n_tables)
                    };

                    // we already hashed, so we don't have to hash again.
                    let value = current_probe_table.get(k);

                    match value {
                        // left and right matches
                        Some(indexes_b) => {
                            on_match_left_join_extend(&mut results, indexes_b, idx_a);
                        }
                        // only left values, right = null
                        None => results.push((idx_a, None)),
                    }
                });
                results
            })
            .flatten()
            .collect()
    })
}
