use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Sub;

use ahash::RandomState;
use arrow::types::NativeType;
use num::Zero;
use rayon::prelude::*;

use super::*;
use crate::frame::groupby::hashing::HASHMAP_INIT_SIZE;
#[cfg(feature = "dtype-categorical")]
use crate::frame::hash_join::_check_categorical_src;
use crate::frame::hash_join::{
    create_probe_table, get_hash_tbl_threaded_join_partitioned, multiple_keys as mk, prepare_strs,
};
use crate::utils::{split_ca, split_df};
use crate::vector_hasher::{df_rows_to_hashes_threaded, AsU64};
use crate::POOL;

pub(super) unsafe fn join_asof_backward_with_indirection_and_tolerance<
    T: PartialOrd + Copy + Sub<Output = T> + Debug,
>(
    val_l: T,
    right: &[T],
    offsets: &[IdxSize],
    tolerance: T,
) -> (Option<IdxSize>, usize) {
    if offsets.is_empty() {
        return (None, 0);
    }
    let mut previous_idx = *offsets.get_unchecked(0);
    let first = *right.get_unchecked(previous_idx as usize);
    if val_l < first {
        (None, 0)
    } else {
        for (idx, &offset) in offsets.iter().enumerate() {
            let val_r = *right.get_unchecked(offset as usize);

            // the point that is larger is not allowed
            if val_r > val_l {
                // compute the distance of previous point, that one was still backwards
                let previous_value = *right.get_unchecked(previous_idx as usize);
                let dist = val_l - previous_value;
                return if dist > tolerance {
                    (None, idx)
                } else {
                    (Some(previous_idx), idx)
                };
            }
            previous_idx = offset
        }
        // check remaining values that still suffice the distance constraint
        let previous_value = *right.get_unchecked(previous_idx as usize);
        let dist = val_l - previous_value;
        if dist > tolerance {
            (None, offsets.len())
        } else {
            (Some(previous_idx), offsets.len())
        }
    }
}

pub(super) unsafe fn join_asof_forward_with_indirection_and_tolerance<
    T: PartialOrd + Copy + Sub<Output = T> + Debug,
>(
    val_l: T,
    right: &[T],
    offsets: &[IdxSize],
    tolerance: T,
) -> (Option<IdxSize>, usize) {
    if offsets.is_empty() {
        return (None, 0);
    }
    let last_offset = *offsets.get_unchecked(offsets.len() - 1);
    let last_value = *right.get_unchecked(last_offset as usize);
    if val_l <= last_value {
        for (idx, &offset) in offsets.iter().enumerate() {
            let val_r = *right.get_unchecked(offset as usize);
            if val_r >= val_l {
                let dist = val_r - val_l;
                return if dist > tolerance {
                    (None, idx)
                } else {
                    (Some(offset), idx)
                };
            }
        }
    }
    (None, offsets.len())
}

pub(super) unsafe fn join_asof_backward_with_indirection<T: PartialOrd + Copy + Debug>(
    val_l: T,
    right: &[T],
    offsets: &[IdxSize],
    // only there to have the same function signature
    _: T,
) -> (Option<IdxSize>, usize) {
    if offsets.is_empty() {
        return (None, 0);
    }
    let mut previous = *offsets.get_unchecked(0);
    let first = *right.get_unchecked(previous as usize);
    if val_l < first {
        (None, 0)
    } else {
        for (idx, &offset) in offsets.iter().enumerate() {
            let val_r = *right.get_unchecked(offset as usize);
            if val_r > val_l {
                return (Some(previous), idx);
            }
            previous = offset
        }
        (Some(previous), offsets.len())
    }
}

pub(super) unsafe fn join_asof_forward_with_indirection<T: PartialOrd + Copy + Debug>(
    val_l: T,
    right: &[T],
    offsets: &[IdxSize],
    // only there to have the same function signature
    _: T,
) -> (Option<IdxSize>, usize) {
    if offsets.is_empty() {
        return (None, 0);
    }
    let last_offset = *offsets.get_unchecked(offsets.len() - 1);
    let last_value = *right.get_unchecked(last_offset as usize);
    if val_l <= last_value {
        for (idx, &offset) in offsets.iter().enumerate() {
            let val_r = *right.get_unchecked(offset as usize);
            if val_r >= val_l {
                return (Some(offset), idx);
            }
        }
    }
    (None, offsets.len())
}

// process the group taken by the `by` operation and keep track of the offset.
// we don't process a group at once but per `index_left` we find the `right_index` and keep track
// of the offsets we have already processed in a separate hashmap. Then on a next iteration we can
// continue from that offsets location.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn process_group<K, T>(
    k: K,
    idx_left: IdxSize,
    tolerance: T,
    indexes_right: &[IdxSize],
    right_tbl_offsets: &mut PlHashMap<K, (usize, Option<IdxSize>)>,
    join_asof_fn: unsafe fn(T, &[T], &[IdxSize], T) -> (Option<IdxSize>, usize),
    left_asof: &[T],
    right_asof: &[T],
    results: &mut Vec<Option<IdxSize>>,
    forward: bool,
) where
    K: Hash + PartialEq + Eq,
    T: NativeType + Sub<Output = T> + PartialOrd + num::Zero,
{
    let (offset_slice, mut previous_join_idx) =
        *right_tbl_offsets.get(&k).unwrap_or(&(0usize, None));
    debug_assert!((idx_left as usize) < left_asof.len());
    let val_l = unsafe { *left_asof.get_unchecked(idx_left as usize) };
    // Safety;
    // elide bound checks
    let (join_idx, offset_slice_add) =
        unsafe { join_asof_fn(val_l, right_asof, &indexes_right[offset_slice..], tolerance) };
    let offset_slice = offset_slice + offset_slice_add;

    match join_idx {
        Some(_) => {
            results.push(join_idx);
            right_tbl_offsets.insert(k, (offset_slice, join_idx));
        }
        None => {
            if forward {
                previous_join_idx = None;
            }
            if tolerance > num::zero() {
                if let Some(idx) = previous_join_idx {
                    debug_assert!((idx as usize) < right_asof.len());
                    let val_r = unsafe { *right_asof.get_unchecked(idx as usize) };
                    let dist = val_l - val_r;
                    if dist > tolerance {
                        previous_join_idx = None;
                    }
                }
            }
            results.push(previous_join_idx)
        }
    }
}

fn asof_join_by_numeric<T, S>(
    by_left: &ChunkedArray<S>,
    by_right: &ChunkedArray<S>,
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
    tolerance: Option<AnyValue<'static>>,
    strategy: AsofStrategy,
) -> PolarsResult<Vec<Option<IdxSize>>>
where
    T: PolarsNumericType,
    S: PolarsNumericType,
    S::Native: Hash + Eq + AsU64,
{
    #[allow(clippy::type_complexity)]
    let (join_asof_fn, tolerance, forward): (
        unsafe fn(T::Native, &[T::Native], &[IdxSize], T::Native) -> (Option<IdxSize>, usize),
        _,
        _,
    ) = match (tolerance, strategy) {
        (Some(tolerance), AsofStrategy::Backward) => {
            let tol = tolerance.extract::<T::Native>().unwrap();
            (
                join_asof_backward_with_indirection_and_tolerance,
                tol,
                false,
            )
        }
        (None, AsofStrategy::Backward) => (
            join_asof_backward_with_indirection,
            T::Native::zero(),
            false,
        ),
        (Some(tolerance), AsofStrategy::Forward) => {
            let tol = tolerance.extract::<T::Native>().unwrap();
            (join_asof_forward_with_indirection_and_tolerance, tol, true)
        }
        (None, AsofStrategy::Forward) => {
            (join_asof_forward_with_indirection, T::Native::zero(), true)
        }
    };

    let left_asof = left_asof.rechunk();
    let err = |_: PolarsError| {
        PolarsError::ComputeError("Keys are not allowed to have null values in asof join.".into())
    };
    let left_asof = left_asof.cont_slice().map_err(err)?;

    let right_asof = right_asof.rechunk();
    let right_asof = right_asof.cont_slice().map_err(err)?;

    let n_threads = POOL.current_num_threads();
    let splitted_left = split_ca(by_left, n_threads).unwrap();
    let splitted_right = split_ca(by_right, n_threads).unwrap();

    let vals_left = splitted_left
        .iter()
        .map(|ca| ca.cont_slice().unwrap())
        .collect::<Vec<_>>();
    let vals_right = splitted_right
        .iter()
        .map(|ca| ca.cont_slice().unwrap())
        .collect::<Vec<_>>();

    let hash_tbls = create_probe_table(vals_right);

    // we determine the offset so that we later know which index to store in the join tuples
    let offsets = vals_left
        .iter()
        .map(|ph| ph.len())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect::<Vec<_>>();

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());

    // next we probe the right relation
    Ok(POOL.install(|| {
        vals_left
            .into_par_iter()
            .zip(offsets)
            // probes_hashes: Vec<u64> processed by this thread
            // offset: offset index
            .flat_map(|(vals_left, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;

                // assume the result tuples equal length of the no. of hashes processed by this thread.
                let mut results = Vec::with_capacity(vals_left.len());

                let mut right_tbl_offsets = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);

                vals_left.iter().enumerate().for_each(|(idx_a, k)| {
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
                            process_group(
                                *k,
                                idx_a,
                                tolerance,
                                indexes_b,
                                &mut right_tbl_offsets,
                                join_asof_fn,
                                left_asof,
                                right_asof,
                                &mut results,
                                forward,
                            );
                        }
                        // only left values, right = null
                        None => results.push(None),
                    }
                });
                results
            })
            .collect()
    }))
}

fn asof_join_by_utf8<T>(
    by_left: &Utf8Chunked,
    by_right: &Utf8Chunked,
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
    tolerance: Option<AnyValue<'static>>,
    strategy: AsofStrategy,
) -> Vec<Option<IdxSize>>
where
    T: PolarsNumericType,
{
    #[allow(clippy::type_complexity)]
    let (join_asof_fn, tolerance, forward): (
        unsafe fn(T::Native, &[T::Native], &[IdxSize], T::Native) -> (Option<IdxSize>, usize),
        _,
        _,
    ) = match (tolerance, strategy) {
        (Some(tolerance), AsofStrategy::Backward) => {
            let tol = tolerance.extract::<T::Native>().unwrap();
            (
                join_asof_backward_with_indirection_and_tolerance,
                tol,
                false,
            )
        }
        (None, AsofStrategy::Backward) => (
            join_asof_backward_with_indirection,
            T::Native::zero(),
            false,
        ),
        (Some(tolerance), AsofStrategy::Forward) => {
            let tol = tolerance.extract::<T::Native>().unwrap();
            (join_asof_forward_with_indirection_and_tolerance, tol, true)
        }
        (None, AsofStrategy::Forward) => {
            (join_asof_forward_with_indirection, T::Native::zero(), true)
        }
    };

    let left_asof = left_asof.rechunk();
    let left_asof = left_asof.cont_slice().unwrap();

    let right_asof = right_asof.rechunk();
    let right_asof = right_asof.cont_slice().unwrap();

    let n_threads = POOL.current_num_threads();
    let splitted_by_left = split_ca(by_left, n_threads).unwrap();
    let splitted_right = split_ca(by_right, n_threads).unwrap();

    let hb = RandomState::default();
    let vals_left = prepare_strs(&splitted_by_left, &hb);
    let vals_right = prepare_strs(&splitted_right, &hb);

    let hash_tbls = create_probe_table(vals_right);

    // we determine the offset so that we later know which index to store in the join tuples
    let offsets = vals_left
        .iter()
        .map(|ph| ph.len())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect::<Vec<_>>();

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());

    // next we probe the right relation
    POOL.install(|| {
        vals_left
            .into_par_iter()
            .zip(offsets)
            // probes_hashes: Vec<u64> processed by this thread
            // offset: offset index
            .flat_map(|(vals_left, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;

                // assume the result tuples equal length of the no. of hashes processed by this thread.
                let mut results = Vec::with_capacity(vals_left.len());

                let mut right_tbl_offsets = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);

                vals_left.iter().enumerate().for_each(|(idx_a, k)| {
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
                            process_group(
                                *k,
                                idx_a,
                                tolerance,
                                indexes_b,
                                &mut right_tbl_offsets,
                                join_asof_fn,
                                left_asof,
                                right_asof,
                                &mut results,
                                forward,
                            );
                        }
                        // only left values, right = null
                        None => results.push(None),
                    }
                });
                results
            })
            .collect()
    })
}

// TODO! optimize this. This does a full scan backwards. Use the same strategy as in the single `by`
// implementations
fn asof_join_by_multiple<T>(
    a: &mut DataFrame,
    b: &mut DataFrame,
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
    tolerance: Option<AnyValue<'static>>,
    strategy: AsofStrategy,
) -> Vec<Option<IdxSize>>
where
    T: PolarsNumericType,
{
    #[allow(clippy::type_complexity)]
    let (join_asof_fn, tolerance, forward): (
        unsafe fn(T::Native, &[T::Native], &[IdxSize], T::Native) -> (Option<IdxSize>, usize),
        _,
        _,
    ) = match (tolerance, strategy) {
        (Some(tolerance), AsofStrategy::Backward) => {
            let tol = tolerance.extract::<T::Native>().unwrap();
            (
                join_asof_backward_with_indirection_and_tolerance,
                tol,
                false,
            )
        }
        (None, AsofStrategy::Backward) => (
            join_asof_backward_with_indirection,
            T::Native::zero(),
            false,
        ),
        (Some(tolerance), AsofStrategy::Forward) => {
            let tol = tolerance.extract::<T::Native>().unwrap();
            (join_asof_forward_with_indirection_and_tolerance, tol, true)
        }
        (None, AsofStrategy::Forward) => {
            (join_asof_forward_with_indirection, T::Native::zero(), true)
        }
    };
    let left_asof = left_asof.rechunk();
    let left_asof = left_asof.cont_slice().unwrap();

    let right_asof = right_asof.rechunk();
    let right_asof = right_asof.cont_slice().unwrap();

    let n_threads = POOL.current_num_threads();
    let dfs_a = split_df(a, n_threads).unwrap();
    let dfs_b = split_df(b, n_threads).unwrap();

    let (build_hashes, random_state) = df_rows_to_hashes_threaded(&dfs_b, None).unwrap();
    let (probe_hashes, _) = df_rows_to_hashes_threaded(&dfs_a, Some(random_state)).unwrap();

    let hash_tbls = mk::create_probe_table(&build_hashes, b);
    // early drop to reduce memory pressure
    drop(build_hashes);

    let n_tables = hash_tbls.len() as u64;
    let offsets = mk::get_offsets(&probe_hashes);

    // next we probe the other relation
    // code duplication is because we want to only do the swap check once
    POOL.install(|| {
        probe_hashes
            .into_par_iter()
            .zip(offsets)
            .flat_map(|(probe_hashes, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;

                // assume the result tuples equal length of the no. of hashes processed by this thread.
                let mut results = Vec::with_capacity(probe_hashes.len());
                let mut right_tbl_offsets = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);

                let local_offset = offset;

                let mut idx_a = local_offset as IdxSize;
                for probe_hashes in probe_hashes.data_views() {
                    for (idx, &h) in probe_hashes.iter().enumerate() {
                        debug_assert!(idx + offset < left_asof.len());
                        // probe table that contains the hashed value
                        let current_probe_table = unsafe {
                            get_hash_tbl_threaded_join_partitioned(h, hash_tbls, n_tables)
                        };

                        let entry = current_probe_table.raw_entry().from_hash(h, |idx_hash| {
                            let idx_b = idx_hash.idx;
                            // Safety:
                            // indices in a join operation are always in bounds.
                            unsafe { mk::compare_df_rows2(a, b, idx_a as usize, idx_b as usize) }
                        });

                        match entry {
                            // left and right matches
                            Some((k, indexes_b)) => {
                                process_group(
                                    // take the first idx as unique identifier of that group.
                                    k.idx,
                                    idx_a,
                                    tolerance,
                                    indexes_b,
                                    &mut right_tbl_offsets,
                                    join_asof_fn,
                                    left_asof,
                                    right_asof,
                                    &mut results,
                                    forward,
                                );
                            }
                            // only left values, right = null
                            None => results.push(None),
                        }
                        idx_a += 1;
                    }
                }

                results
            })
            .collect()
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_join<T: PolarsNumericType>(
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
    left_by_s: &Series,
    right_by_s: &Series,
    left_by: &mut DataFrame,
    right_by: &mut DataFrame,
    strategy: AsofStrategy,
    tolerance: Option<AnyValue<'static>>,
) -> PolarsResult<Vec<Option<IdxSize>>> {
    let out = if left_by.width() == 1 {
        match left_by_s.dtype() {
            DataType::Utf8 => asof_join_by_utf8(
                left_by_s.utf8().unwrap(),
                right_by_s.utf8().unwrap(),
                left_asof,
                right_asof,
                tolerance,
                strategy,
            ),
            _ => {
                if left_by_s.bit_repr_is_large() {
                    let left_by = left_by_s.bit_repr_large();
                    let right_by = right_by_s.bit_repr_large();
                    asof_join_by_numeric(
                        &left_by, &right_by, left_asof, right_asof, tolerance, strategy,
                    )?
                } else {
                    let left_by = left_by_s.bit_repr_small();
                    let right_by = right_by_s.bit_repr_small();
                    asof_join_by_numeric(
                        &left_by, &right_by, left_asof, right_asof, tolerance, strategy,
                    )?
                }
            }
        }
    } else {
        for (lhs, rhs) in left_by.get_columns().iter().zip(right_by.get_columns()) {
            check_asof_columns(lhs, rhs)?;
            #[cfg(feature = "dtype-categorical")]
            _check_categorical_src(lhs.dtype(), rhs.dtype())?;
        }
        asof_join_by_multiple(
            left_by, right_by, left_asof, right_asof, tolerance, strategy,
        )
    };
    Ok(out)
}

impl DataFrame {
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    pub fn _join_asof_by(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
        left_by: Vec<String>,
        right_by: Vec<String>,
        strategy: AsofStrategy,
        tolerance: Option<AnyValue<'static>>,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<DataFrame> {
        let left_asof = self.column(left_on)?.to_physical_repr();
        let right_asof = other.column(right_on)?.to_physical_repr();
        let right_asof_name = right_asof.name();
        let left_asof_name = left_asof.name();

        check_asof_columns(&left_asof, &right_asof)?;

        let mut left_by = self.select_physical(left_by)?;
        let mut right_by = other.select_physical(right_by)?;

        let left_by_s = left_by.get_columns()[0].to_physical_repr().into_owned();
        let right_by_s = right_by.get_columns()[0].to_physical_repr().into_owned();

        let right_join_tuples = with_match_physical_numeric_polars_type!(left_asof.dtype(), |$T| {
            let left_asof: &ChunkedArray<$T> = left_asof.as_ref().as_ref().as_ref();
            let right_asof: &ChunkedArray<$T> = right_asof.as_ref().as_ref().as_ref();

            dispatch_join(
                left_asof,
                right_asof,
                &left_by_s,
                &right_by_s,
                &mut left_by,
                &mut right_by,
                strategy,
                tolerance
            )
        })?;

        let mut drop_these = right_by.get_column_names();
        if left_asof_name == right_asof_name {
            drop_these.push(right_asof_name);
        }

        let cols = other
            .get_columns()
            .iter()
            .filter_map(|s| {
                if drop_these.contains(&s.name()) {
                    None
                } else {
                    Some(s.clone())
                }
            })
            .collect();
        let other = DataFrame::new_no_checks(cols);

        let mut left = self.clone();
        let mut right_join_tuples = &*right_join_tuples;

        if let Some((offset, len)) = slice {
            left = left.slice(offset, len);
            right_join_tuples = slice_slice(right_join_tuples, offset, len);
        }

        // Safety:
        // join tuples are in bounds
        let right_df = unsafe {
            other.take_opt_iter_unchecked(
                right_join_tuples
                    .iter()
                    .map(|opt_idx| opt_idx.map(|idx| idx as usize)),
            )
        };

        _finish_join(left, right_df, None)
    }

    /// This is similar to a left-join except that we match on nearest key rather than equal keys.
    /// The keys must be sorted to perform an asof join. This is a special implementation of an asof join
    /// that searches for the nearest keys within a subgroup set by `by`.
    #[allow(clippy::too_many_arguments)]
    pub fn join_asof_by<I, S>(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
        left_by: I,
        right_by: I,
        strategy: AsofStrategy,
        tolerance: Option<AnyValue<'static>>,
    ) -> PolarsResult<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let left_by = left_by
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        let right_by = right_by
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        self._join_asof_by(
            other, left_on, right_on, left_by, right_by, strategy, tolerance, None,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_asof_by() -> PolarsResult<()> {
        let a = df![
        "a" => [-1, 2, 3, 3, 3, 4],
        "b" => ["a", "b", "c", "d", "e", "f"]
        ]?;

        let b = df![
         "a" => [1, 2, 3, 3],
            "b" => ["a", "b", "c", "d"],
            "right_vals" => [1, 2, 3, 4]
        ]?;

        let out = a.join_asof_by(&b, "a", "a", ["b"], ["b"], AsofStrategy::Backward, None)?;
        assert_eq!(out.get_column_names(), &["a", "b", "right_vals"]);
        let out = out.column("right_vals").unwrap();
        let out = out.i32().unwrap();
        assert_eq!(
            Vec::from(out),
            &[None, Some(2), Some(3), Some(4), None, None]
        );
        Ok(())
    }

    #[test]
    fn test_asof_by2() -> PolarsResult<()> {
        let trades = df![
            "time" => [23i64, 38, 48, 48, 48],
            "ticker" => ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
            "groups_numeric" => [1, 1, 2, 2, 3],
            "bid" => [51.95, 51.95, 720.77, 720.92, 98.0]
        ]?;

        let quotes = df![
                   "time" => [23i64,
        23,
        30,
        41,
        48,
        49,
        72,
        75],
                   "ticker" => ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
                   "groups_numeric" => [2, 1, 1, 1, 2, 3, 2, 1],
                   "bid" => [720.5, 51.95, 51.97, 51.99, 720.5, 97.99, 720.5, 52.01]

               ]?;

        let out = trades.join_asof_by(
            &quotes,
            "time",
            "time",
            ["ticker"],
            ["ticker"],
            AsofStrategy::Backward,
            None,
        )?;
        let a = out.column("bid_right").unwrap();
        let a = a.f64().unwrap();
        let expected = &[Some(51.95), Some(51.97), Some(720.5), Some(720.5), None];

        assert_eq!(Vec::from(a), expected);

        let out = trades.join_asof_by(
            &quotes,
            "time",
            "time",
            ["groups_numeric"],
            ["groups_numeric"],
            AsofStrategy::Backward,
            None,
        )?;
        let a = out.column("bid_right").unwrap();
        let a = a.f64().unwrap();

        assert_eq!(Vec::from(a), expected);

        Ok(())
    }

    #[test]
    fn test_asof_by3() -> PolarsResult<()> {
        let a = df![
        "a" => [ -1,   2,   2,   3,   3,   3,   4],
        "b" => ["a", "a", "b", "c", "d", "e", "f"]
        ]?;

        let b = df![
                 "a" => [  1,   3,   2,   3,   2],
                 "b" => ["a", "a", "b", "c", "d"],
        "right_vals" => [  1,   3,   2,   3,   4]
        ]?;

        let out = a.join_asof_by(&b, "a", "a", ["b"], ["b"], AsofStrategy::Forward, None)?;
        assert_eq!(out.get_column_names(), &["a", "b", "right_vals"]);
        let out = out.column("right_vals").unwrap();
        let out = out.i32().unwrap();
        assert_eq!(
            Vec::from(out),
            &[Some(1), Some(3), Some(2), Some(3), None, None, None]
        );

        let out = a.join_asof_by(
            &b,
            "a",
            "a",
            ["b"],
            ["b"],
            AsofStrategy::Forward,
            Some(AnyValue::Int32(1)),
        )?;
        assert_eq!(out.get_column_names(), &["a", "b", "right_vals"]);
        let out = out.column("right_vals").unwrap();
        let out = out.i32().unwrap();
        assert_eq!(
            Vec::from(out),
            &[None, Some(3), Some(2), Some(3), None, None, None]
        );

        Ok(())
    }

    #[test]
    fn test_asof_by4() -> PolarsResult<()> {
        let trades = df![
            "time" =>    [23i64,     38,     48,     48,     48],
            "ticker" => ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
            "groups_numeric" => [1, 1, 2, 2, 3],
            "bid" => [51.95, 51.95, 720.77, 720.92, 98.0]
        ]?;

        let quotes = df![
            "time" =>    [23i64,     23,     30,     41,     48,     49,     72,     75],
            "ticker" => ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
            "bid" =>     [720.5,  51.95,  51.97,  51.99,  720.5,  97.99,  720.5,  52.01],
            "groups_numeric" => [2, 1, 1, 1, 2, 3, 2, 1],

        ]?;
        /*
        trades:
        shape: (5, 4)
        ┌──────┬────────┬────────────────┬────────┐
        │ time ┆ ticker ┆ groups_numeric ┆ bid    │
        │ ---  ┆ ---    ┆ ---            ┆ ---    │
        │ i64  ┆ str    ┆ i32            ┆ f64    │
        ╞══════╪════════╪════════════════╪════════╡
        │ 23   ┆ MSFT   ┆ 1              ┆ 51.95  │
        │ 38   ┆ MSFT   ┆ 1              ┆ 51.95  │
        │ 48   ┆ GOOG   ┆ 2              ┆ 720.77 │
        │ 48   ┆ GOOG   ┆ 2              ┆ 720.92 │
        │ 48   ┆ AAPL   ┆ 3              ┆ 98.0   │
        └──────┴────────┴────────────────┴────────┘
        quotes:
        shape: (8, 4)
        ┌──────┬────────┬───────┬────────────────┐
        │ time ┆ ticker ┆ bid   ┆ groups_numeric │
        │ ---  ┆ ---    ┆ ---   ┆ ---            │
        │ i64  ┆ str    ┆ f64   ┆ i32            │
        ╞══════╪════════╪═══════╪════════════════╡
        │ 23   ┆ GOOG   ┆ 720.5 ┆ 2              │
        │ 23   ┆ MSFT   ┆ 51.95 ┆ 1              │
        │ 30   ┆ MSFT   ┆ 51.97 ┆ 1              │
        │ 41   ┆ MSFT   ┆ 51.99 ┆ 1              │
        │ 48   ┆ GOOG   ┆ 720.5 ┆ 2              │
        │ 49   ┆ AAPL   ┆ 97.99 ┆ 3              │
        │ 72   ┆ GOOG   ┆ 720.5 ┆ 2              │
        │ 75   ┆ MSFT   ┆ 52.01 ┆ 1              │
        └──────┴────────┴───────┴────────────────┘
        */

        let out = trades.join_asof_by(
            &quotes,
            "time",
            "time",
            ["ticker"],
            ["ticker"],
            AsofStrategy::Forward,
            None,
        )?;
        let a = out.column("bid_right").unwrap();
        let a = a.f64().unwrap();
        let expected = &[
            Some(51.95),
            Some(51.99),
            Some(720.5),
            Some(720.5),
            Some(97.99),
        ];

        assert_eq!(Vec::from(a), expected);

        let out = trades.join_asof_by(
            &quotes,
            "time",
            "time",
            ["groups_numeric"],
            ["groups_numeric"],
            AsofStrategy::Forward,
            None,
        )?;
        let a = out.column("bid_right").unwrap();
        let a = a.f64().unwrap();

        assert_eq!(Vec::from(a), expected);

        Ok(())
    }
}
