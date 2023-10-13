use polars_core::utils::flatten;
use polars_utils::iter::EnumerateIdxTrait;
use polars_utils::sync::SyncPtr;

use super::*;

pub(super) fn probe_inner<T, F, I>(
    probe: I,
    hash_tbls: &[PlHashMap<T, Vec<IdxSize>>],
    results: &mut Vec<(IdxSize, IdxSize)>,
    local_offset: IdxSize,
    n_tables: u64,
    swap_fn: F,
) where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    I: IntoIterator<Item = T>,
    // <I as IntoIterator>::IntoIter: TrustedLen,
    F: Fn(IdxSize, IdxSize) -> (IdxSize, IdxSize),
{
    assert!(hash_tbls.len().is_power_of_two());
    probe.into_iter().enumerate_idx().for_each(|(idx_a, k)| {
        let idx_a = idx_a + local_offset;
        // probe table that contains the hashed value
        let current_probe_table =
            unsafe { get_hash_tbl_threaded_join_partitioned(k.as_u64(), hash_tbls, n_tables) };

        let value = current_probe_table.get(&k);

        if let Some(indexes_b) = value {
            let tuples = indexes_b.iter().map(|&idx_b| swap_fn(idx_a, idx_b));
            results.extend(tuples);
        }
    });
}

pub(super) fn hash_join_tuples_inner<T, I>(
    probe: Vec<I>,
    build: Vec<I>,
    // Because b should be the shorter relation we could need to swap to keep left left and right right.
    swapped: bool,
    validate: JoinValidation,
) -> PolarsResult<(Vec<IdxSize>, Vec<IdxSize>)>
where
    I: IntoIterator<Item = T> + Send + Sync + Copy,
    // <I as IntoIterator>::IntoIter: TrustedLen,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    // NOTE: see the left join for more elaborate comments
    // first we hash one relation
    let hash_tbls = if validate.needs_checks() {
        let expected_size = build
            .iter()
            .map(|v| v.into_iter().size_hint().1.unwrap())
            .sum();
        let hash_tbls = build_tables(build);
        let build_size = hash_tbls.iter().map(|m| m.len()).sum();
        validate.validate_build(build_size, expected_size, swapped)?;
        hash_tbls
    } else {
        build_tables(build)
    };

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());
    let offsets = probe_to_offsets(&probe);
    // next we probe the other relation
    // code duplication is because we want to only do the swap check once
    let out = POOL.install(|| {
        let tuples = probe
            .into_par_iter()
            .zip(offsets)
            .map(|(probe, offset)| {
                let probe = probe.into_iter();
                // local reference
                let hash_tbls = &hash_tbls;
                let mut results = Vec::with_capacity(probe.size_hint().1.unwrap());
                let local_offset = offset as IdxSize;

                // branch is to hoist swap out of the inner loop.
                if swapped {
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
            .collect::<Vec<_>>();

        // parallel materialization
        let (cap, offsets) = flatten::cap_and_offsets(&tuples);
        let mut left = Vec::with_capacity(cap);
        let mut right = Vec::with_capacity(cap);

        let left_ptr = unsafe { SyncPtr::new(left.as_mut_ptr()) };
        let right_ptr = unsafe { SyncPtr::new(right.as_mut_ptr()) };

        tuples
            .into_par_iter()
            .zip(offsets)
            .for_each(|(tuples, offset)| unsafe {
                let left_ptr: *mut IdxSize = left_ptr.get();
                let left_ptr = left_ptr.add(offset);
                let right_ptr: *mut IdxSize = right_ptr.get();
                let right_ptr = right_ptr.add(offset);

                // amortize loop counter
                for i in 0..tuples.len() {
                    let tuple = tuples.get_unchecked(i);
                    let left_row_idx = tuple.0;
                    let right_row_idx = tuple.1;

                    std::ptr::write(left_ptr.add(i), left_row_idx);
                    std::ptr::write(right_ptr.add(i), right_row_idx);
                }
            });
        unsafe {
            left.set_len(cap);
            right.set_len(cap);
        }

        (left, right)
    });
    Ok(out)
}
