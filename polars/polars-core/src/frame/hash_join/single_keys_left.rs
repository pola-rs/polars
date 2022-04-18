use std::io::Write;
use super::*;
use arrow::Either;
use polars_utils::flatten;

pub(super) type LeftJoinIndices = (IdxSize, Option<IdxSize>);
pub(super) type LeftJoinChunkIndices = ([IdxSize; 2], Option<[IdxSize; 2]>);
pub(super) type LeftJoinResult =
    Either<(Vec<IdxSize>, Vec<Option<IdxSize>>), (Vec<[IdxSize; 2]>, Vec<Option<[IdxSize; 2]>>)>;

#[inline]
pub(super) fn on_match_left_join_extend(
    results: &mut Vec<LeftJoinIndices>,
    indexes_b: &[IdxSize],
    idx_a: IdxSize,
) {
    results.extend(indexes_b.iter().map(|&idx_b| (idx_a, Some(idx_b))))
}

pub(super) fn hash_join_tuples_left<'a, T: 'a, IntoSlice>(
    probe: Vec<IntoSlice>,
    build: Vec<IntoSlice>,
    // map the global indices to [chunk_idx, array_idx]
    // only needed if we have non contiguous memory
    chunk_mapping: Option<&'a [[IdxSize; 2]]>,
) -> LeftJoinResult
where
    IntoSlice: 'a + AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    // first we hash one relation
    let hash_tbls = create_probe_table(build);

    // we determine the offset so that we later know which index to store in the join tuples
    let offsets = probe_to_offsets(&probe);

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());

    // next we probe the other relation
    let result: Vec<LeftJoinResult> = POOL.install(move || {
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
                let mut result_idx_left = Vec::with_capacity(probe.len());
                let mut result_idx_right = Vec::with_capacity(probe.len());

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
                            result_idx_left.extend(std::iter::repeat(idx_a).take(indexes_b.len()));
                            result_idx_right.extend(indexes_b.iter().copied().map(Some))
                        }
                        // only left values, right = null
                        None => {
                            result_idx_left.push(idx_a);
                            result_idx_right.push(None);
                        }
                    }
                });
                match chunk_mapping {
                    None => LeftJoinResult::Left((result_idx_left, result_idx_right)),
                    Some(mapping) => {
                        let left = unsafe {
                            result_idx_left
                                .iter()
                                .map(|idx| *mapping.get_unchecked(*idx as usize))
                                .collect::<Vec<_>>()
                        };
                        let right = unsafe {
                            result_idx_right
                                .iter()
                                .map(|opt_idx| {
                                    opt_idx.map(|idx| *mapping.get_unchecked(idx as usize))
                                })
                                .collect::<Vec<_>>()
                        };

                        LeftJoinResult::Right((left, right))
                    }
                }
            })
            .collect()
    });

    // single chunk
    if result[0].is_left() {
        let mut join_idx_left = result
            .iter()
            .map(|join_idx| &join_idx.as_ref().left().unwrap().0)
            .collect::<Vec<_>>();
        let mut join_idx_right = result
            .iter()
            .map(|join_idx| &join_idx.as_ref().left().unwrap().1)
            .collect::<Vec<_>>();
        let (join_idx_left, join_idx_right) = rayon::join(
            || flatten(&join_idx_left, None),
            || flatten(&join_idx_right, None)
        );
        Either::Left((join_idx_left, join_idx_right))
    } else {
        let mut join_idx_left = result
            .iter()
            .map(|join_idx| &join_idx.as_ref().right().unwrap().0)
            .collect::<Vec<_>>();
        let mut join_idx_right = result
            .iter()
            .map(|join_idx| &join_idx.as_ref().right().unwrap().1)
            .collect::<Vec<_>>();

        let (join_idx_left, join_idx_right) = rayon::join(
            || flatten(&join_idx_left, None),
            || flatten(&join_idx_right, None)
        );
        Either::Right((join_idx_left, join_idx_right))
    }
}
