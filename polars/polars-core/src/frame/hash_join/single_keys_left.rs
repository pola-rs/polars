use super::single_keys::create_probe_table;
use super::*;
use crate::frame::hash_join::single_keys::probe_to_offsets;
use crate::utils::flatten::flatten_par;

#[cfg(feature = "chunked_ids")]
unsafe fn apply_mapping(idx: Vec<IdxSize>, chunk_mapping: &[ChunkId]) -> Vec<ChunkId> {
    idx.iter()
        .map(|idx| *chunk_mapping.get_unchecked(*idx as usize))
        .collect()
}

#[cfg(feature = "chunked_ids")]
unsafe fn apply_opt_mapping(
    idx: Vec<Option<IdxSize>>,
    chunk_mapping: &[ChunkId],
) -> Vec<Option<ChunkId>> {
    idx.iter()
        .map(|opt_idx| opt_idx.map(|idx| *chunk_mapping.get_unchecked(idx as usize)))
        .collect()
}

#[cfg(feature = "chunked_ids")]
pub(super) fn finish_left_join_mappings(
    result_idx_left: Vec<IdxSize>,
    result_idx_right: Vec<Option<IdxSize>>,
    chunk_mapping_left: Option<&[ChunkId]>,
    chunk_mapping_right: Option<&[ChunkId]>,
) -> LeftJoinIds {
    let left = match chunk_mapping_left {
        None => ChunkJoinIds::Left(result_idx_left),
        Some(mapping) => ChunkJoinIds::Right(unsafe { apply_mapping(result_idx_left, mapping) }),
    };

    let right = match chunk_mapping_right {
        None => ChunkJoinOptIds::Left(result_idx_right),
        Some(mapping) => {
            ChunkJoinOptIds::Right(unsafe { apply_opt_mapping(result_idx_right, mapping) })
        }
    };
    (left, right)
}

#[cfg(not(feature = "chunked_ids"))]
pub(super) fn finish_left_join_mappings(
    _result_idx_left: Vec<IdxSize>,
    _result_idx_right: Vec<Option<IdxSize>>,
    _chunk_mapping_left: Option<&[ChunkId]>,
    _chunk_mapping_right: Option<&[ChunkId]>,
) -> LeftJoinIds {
    (_result_idx_left, _result_idx_right)
}

pub(super) fn flatten_left_join_ids(result: Vec<LeftJoinIds>) -> LeftJoinIds {
    #[cfg(feature = "chunked_ids")]
    {
        let left = if result[0].0.is_left() {
            let lefts = result
                .iter()
                .map(|join_id| join_id.0.as_ref().left().unwrap())
                .collect::<Vec<_>>();
            let lefts = flatten_par(&lefts);
            ChunkJoinIds::Left(lefts)
        } else {
            let lefts = result
                .iter()
                .map(|join_id| join_id.0.as_ref().right().unwrap())
                .collect::<Vec<_>>();
            let lefts = flatten_par(&lefts);
            ChunkJoinIds::Right(lefts)
        };

        let right = if result[0].1.is_left() {
            let rights = result
                .iter()
                .map(|join_id| join_id.1.as_ref().left().unwrap())
                .collect::<Vec<_>>();
            let rights = flatten_par(&rights);
            ChunkJoinOptIds::Left(rights)
        } else {
            let rights = result
                .iter()
                .map(|join_id| join_id.1.as_ref().right().unwrap())
                .collect::<Vec<_>>();
            let rights = flatten_par(&rights);
            ChunkJoinOptIds::Right(rights)
        };

        (left, right)
    }
    #[cfg(not(feature = "chunked_ids"))]
    {
        let lefts = result.iter().map(|join_id| &join_id.0).collect::<Vec<_>>();
        let rights = result.iter().map(|join_id| &join_id.1).collect::<Vec<_>>();
        let lefts = flatten_par(&lefts);
        let rights = flatten_par(&rights);
        (lefts, rights)
    }
}

pub(super) fn hash_join_tuples_left<T, IntoSlice>(
    probe: Vec<IntoSlice>,
    build: Vec<IntoSlice>,
    // map the global indices to [chunk_idx, array_idx]
    // only needed if we have non contiguous memory
    chunk_mapping_left: Option<&[ChunkId]>,
    chunk_mapping_right: Option<&[ChunkId]>,
    validate: JoinValidation,
) -> PolarsResult<LeftJoinIds>
where
    IntoSlice: AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    // first we hash one relation
    let hash_tbls = create_probe_table(build);
    if validate.needs_checks() {
        let build_size = hash_tbls.iter().map(|m| m.len()).sum();
        let expected_size = probe.iter().map(|v| v.as_ref().len()).sum();
        validate.validate_build(build_size, expected_size, false)?;
    }

    // we determine the offset so that we later know which index to store in the join tuples
    let offsets = probe_to_offsets(&probe);

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());

    // next we probe the other relation
    let result: Vec<LeftJoinIds> = POOL.install(move || {
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
                finish_left_join_mappings(
                    result_idx_left,
                    result_idx_right,
                    chunk_mapping_left,
                    chunk_mapping_right,
                )
            })
            .collect()
    });

    Ok(flatten_left_join_ids(result))
}
