use polars_core::utils::flatten;
use polars_utils::hashing::{hash_to_partition, DirtyHash};
use polars_utils::idx_vec::IdxVec;
use polars_utils::itertools::Itertools;
use polars_utils::nulls::IsNull;
use polars_utils::sync::SyncPtr;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

use super::*;

pub(super) fn probe_inner<T, F, I>(
    probe: I,
    hash_tbls: &[PlHashMap<<T as ToTotalOrd>::TotalOrdItem, IdxVec>],
    results: &mut Vec<(IdxSize, IdxSize)>,
    local_offset: IdxSize,
    n_tables: usize,
    swap_fn: F,
) where
    T: TotalHash + TotalEq + DirtyHash + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Hash + Eq + DirtyHash,
    I: IntoIterator<Item = T>,
    F: Fn(IdxSize, IdxSize) -> (IdxSize, IdxSize),
{
    probe.into_iter().enumerate_idx().for_each(|(idx_a, k)| {
        let k = k.to_total_ord();
        let idx_a = idx_a + local_offset;
        // probe table that contains the hashed value
        let current_probe_table =
            unsafe { hash_tbls.get_unchecked(hash_to_partition(k.dirty_hash(), n_tables)) };

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
    join_nulls: bool,
) -> PolarsResult<(Vec<IdxSize>, Vec<IdxSize>)>
where
    I: IntoIterator<Item = T> + Send + Sync + Clone,
    T: Send + Sync + Copy + TotalHash + TotalEq + DirtyHash + ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Send + Sync + Copy + Hash + Eq + DirtyHash + IsNull,
{
    // NOTE: see the left join for more elaborate comments
    // first we hash one relation
    let hash_tbls = if validate.needs_checks() {
        let expected_size = build
            .iter()
            .map(|v| v.clone().into_iter().size_hint().1.unwrap())
            .sum();
        let hash_tbls = build_tables(build, join_nulls);
        let build_size = hash_tbls.iter().map(|m| m.len()).sum();
        validate.validate_build(build_size, expected_size, swapped)?;
        hash_tbls
    } else {
        build_tables(build, join_nulls)
    };

    let n_tables = hash_tbls.len();
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
