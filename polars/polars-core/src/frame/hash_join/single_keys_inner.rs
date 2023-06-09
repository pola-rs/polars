use polars_utils::iter::EnumerateIdxTrait;
use polars_utils::sync::SyncPtr;

use super::single_keys::create_probe_table;
use super::*;
use crate::frame::hash_join::single_keys::probe_to_offsets;
use crate::utils::flatten;

/// Probe the build table and add tuples to the results (inner join)
pub(super) fn probe_inner<T, F>(
    probe: &[T],
    hash_tbls: &[PlHashMap<T, Vec<IdxSize>>],
    results: &mut Vec<(IdxSize, IdxSize)>,
    local_offset: IdxSize,
    n_tables: u64,
    swap_fn: F,
) where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    F: Fn(IdxSize, IdxSize) -> (IdxSize, IdxSize),
{
    assert!(hash_tbls.len().is_power_of_two());
    probe.iter().enumerate_idx().for_each(|(idx_a, k)| {
        let idx_a = idx_a + local_offset;
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
    validate: JoinValidation,
) -> PolarsResult<(Vec<IdxSize>, Vec<IdxSize>)>
where
    IntoSlice: AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    // NOTE: see the left join for more elaborate comments

    // first we hash one relation
    let hash_tbls = create_probe_table(build);
    if validate.needs_checks() {
        let build_size = hash_tbls.iter().map(|m| m.len()).sum();
        let expected_size = probe.iter().map(|v| v.as_ref().len()).sum();
        validate.validate_build(build_size, expected_size, swap)?;
    }

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
                let probe = probe.as_ref();
                // local reference
                let hash_tbls = &hash_tbls;
                let mut results = Vec::with_capacity(probe.len());
                let local_offset = offset as IdxSize;

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
