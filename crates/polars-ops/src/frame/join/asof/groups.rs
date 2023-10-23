use std::hash::Hash;

use ahash::RandomState;
use num_traits::Zero;
use polars_core::hashing::partition::AsU64;
use polars_core::hashing::{_df_rows_to_hashes_threaded_vertical, _HASHMAP_INIT_SIZE};
use polars_core::utils::{split_ca, split_df};
use polars_core::POOL;
use polars_utils::abs_diff::AbsDiff;
use rayon::prelude::*;
use smartstring::alias::String as SmartString;

use super::*;
use crate::frame::IntoDf;


fn asof_join_by_numeric<T, S, A, F>(
    by_left: &ChunkedArray<S>,
    by_right: &ChunkedArray<S>,
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
    filter: F,
) -> PolarsResult<Vec<Option<IdxSize>>>
where
    T: PolarsDataType,
    S: PolarsNumericType,
    S::Native: Hash + Eq + AsU64,
    A: for<'a> AsofJoinState<T::Physical<'a>>,
    F: Sync + for<'a> Fn(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    let left_asof = left_asof.rechunk();
    let right_asof = right_asof.rechunk();
    let left_val_arr = left_asof.downcast_iter().next().unwrap();
    let right_val_arr = right_asof.downcast_iter().next().unwrap();

    let n_threads = POOL.current_num_threads();
    let splitted_left = split_ca(by_left, n_threads).unwrap();
    let splitted_right = split_ca(by_right, n_threads).unwrap();

    // TODO: handle nulls more efficiently. Right now we just join on the value
    // ignoring the validity mask, and ignore the nulls later.
    let right_slices = splitted_right
        .iter()
        .map(|ca| ca.downcast_iter().next().unwrap().values_iter())
        .collect();
    let hash_tbls = build_tables(right_slices);

    // We determine the offset of each thread block so that we later know which
    // index to store in the join tuples.
    let offsets = splitted_left
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

    // Now we probe the right hand side for each left hand side.
    Ok(POOL
        .install(|| {
            splitted_left
                .into_par_iter()
                .zip(offsets)
                .flat_map(|(by_left, offset)| {
                    let mut results = Vec::with_capacity(by_left.len());
                    let mut group_states: PlHashMap<S::Native, A> =
                        PlHashMap::with_capacity(_HASHMAP_INIT_SIZE);

                    let by_left_chunk = by_left.downcast_iter().next().unwrap();
                    for (rel_idx_left, opt_by_left_k) in by_left_chunk.iter().enumerate() {
                        let Some(by_left_k) = opt_by_left_k else {
                            results.push(None);
                            continue;
                        };
                        let idx_left = (rel_idx_left + offset) as IdxSize;
                        let Some(left_val) = left_val_arr.get(idx_left as usize) else {
                            results.push(None);
                            continue;
                        };

                        let group_probe_table = unsafe {
                            get_hash_tbl_threaded_join_partitioned(
                                by_left_k.as_u64(),
                                &hash_tbls,
                                n_tables,
                            )
                        };
                        let Some(right_grp_idxs) = group_probe_table.get(by_left_k) else {
                            results.push(None);
                            continue;
                        };

                        let grp_state = group_states.entry(*by_left_k).or_default();
                        let r_idx = grp_state.next(
                            &left_val,
                            |i| right_val_arr.get(right_grp_idxs[i as usize] as usize),
                            right_grp_idxs.len() as IdxSize,
                        );
                        
                        let ret = r_idx
                            .map(|i| right_grp_idxs[i as usize])
                            .filter(|i| {
                                let right_val = right_val_arr.get(*i as usize).unwrap();
                                filter(left_val, right_val)
                            });
                        results.push(ret);
                    }
                    results
                })
        })
        .collect())
}

fn asof_join_by_binary<T, A, F>(
    by_left: &BinaryChunked,
    by_right: &BinaryChunked,
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
    filter: F,
) -> Vec<Option<IdxSize>>
where
    T: PolarsDataType,
    A: for<'a> AsofJoinState<T::Physical<'a>>,
    F: Sync + for<'a> Fn(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    let left_asof = left_asof.rechunk();
    let right_asof = right_asof.rechunk();
    let left_val_arr = left_asof.downcast_iter().next().unwrap();
    let right_val_arr = right_asof.downcast_iter().next().unwrap();

    let n_threads = POOL.current_num_threads();
    let splitted_by_left = split_ca(by_left, n_threads).unwrap();
    let splitted_right = split_ca(by_right, n_threads).unwrap();

    let hb = RandomState::default();
    let prep_by_left = prepare_bytes(&splitted_by_left, &hb);
    let prep_by_right = prepare_bytes(&splitted_right, &hb);
    let hash_tbls = build_tables(prep_by_right);

    // We determine the offset of each thread block so that we later know which
    // index to store in the join tuples.
    let offsets = prep_by_left
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

    // Now we probe the right hand side for each left hand side.
    POOL.install(|| {
        prep_by_left
            .into_par_iter()
            .zip(offsets)
            .flat_map(|(by_left, offset)| {
                let mut results = Vec::with_capacity(by_left.len());
                let mut group_states: PlHashMap<_, A> =
                    PlHashMap::with_capacity(_HASHMAP_INIT_SIZE);

                for (rel_idx_left, by_left_k) in by_left.iter().enumerate() {
                    let idx_left = (rel_idx_left + offset) as IdxSize;
                    let Some(left_val) = left_val_arr.get(idx_left as usize) else {
                        results.push(None);
                        continue;
                    };

                    let group_probe_table = unsafe {
                        get_hash_tbl_threaded_join_partitioned(
                            by_left_k.as_u64(),
                            &hash_tbls,
                            n_tables,
                        )
                    };
                    let Some(right_grp_idxs) = group_probe_table.get(by_left_k) else {
                        results.push(None);
                        continue;
                    };

                    let grp_state = group_states.entry(*by_left_k).or_default();
                    let r_idx = grp_state.next(
                        &left_val,
                        |i| right_val_arr.get(right_grp_idxs[i as usize] as usize),
                        right_grp_idxs.len() as IdxSize,
                    );
                    
                    let ret = r_idx
                        .map(|i| right_grp_idxs[i as usize])
                        .filter(|i| {
                            let right_val = right_val_arr.get(*i as usize).unwrap();
                            filter(left_val, right_val)
                        });
                    results.push(ret);
                }
                results
            })
            .collect()
    })
}

/*
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
        },
        (None, AsofStrategy::Backward) => (
            join_asof_backward_with_indirection,
            T::Native::zero(),
            false,
        ),
        (Some(tolerance), AsofStrategy::Forward) => {
            let tol = tolerance.extract::<T::Native>().unwrap();
            (join_asof_forward_with_indirection_and_tolerance, tol, true)
        },
        (None, AsofStrategy::Forward) => {
            (join_asof_forward_with_indirection, T::Native::zero(), true)
        },
        (Some(tolerance), AsofStrategy::Nearest) => {
            let tol = tolerance.extract::<T::Native>().unwrap();
            (join_asof_nearest_with_indirection_and_tolerance, tol, false)
        },
        (None, AsofStrategy::Nearest) => {
            (join_asof_nearest_with_indirection, T::Native::zero(), false)
        },
    };
    let left_asof = left_asof.rechunk();
    let left_asof = left_asof.cont_slice().unwrap();

    let right_asof = right_asof.rechunk();
    let right_asof = right_asof.cont_slice().unwrap();

    let n_threads = POOL.current_num_threads();
    let dfs_a = split_df(a, n_threads).unwrap();
    let dfs_b = split_df(b, n_threads).unwrap();

    let (build_hashes, random_state) = _df_rows_to_hashes_threaded_vertical(&dfs_b, None).unwrap();
    let (probe_hashes, _) =
        _df_rows_to_hashes_threaded_vertical(&dfs_a, Some(random_state)).unwrap();

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
                let mut right_tbl_offsets = PlHashMap::with_capacity(_HASHMAP_INIT_SIZE);

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
                            },
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
*/

#[allow(clippy::too_many_arguments)]
fn dispatch_join_by_type<T, A, F>(
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
    left_by: &mut DataFrame,
    right_by: &mut DataFrame,
    filter: F,
) -> PolarsResult<Vec<Option<IdxSize>>>
where
    T: PolarsDataType,
    A: for<'a> AsofJoinState<T::Physical<'a>>,
    F: Sync + for<'a> Fn(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    let out = if left_by.width() == 1 {
        let left_by_s = left_by.get_columns()[0].to_physical_repr().into_owned();
        let right_by_s = right_by.get_columns()[0].to_physical_repr().into_owned();
        let left_dtype = left_by_s.dtype();
        let right_dtype = right_by_s.dtype();
        polars_ensure!(left_dtype == right_dtype,
            ComputeError: "mismatching dtypes in 'by' parameter of asof-join: `{}` and `{}`", left_dtype, right_dtype
        );
        match left_dtype {
            DataType::Utf8 => {
                let left_by = &left_by_s.utf8().unwrap().as_binary();
                let right_by = right_by_s.utf8().unwrap().as_binary();
                asof_join_by_binary::<T, A, _>(&left_by, &right_by, left_asof, right_asof, filter)
            },
            DataType::Binary => {
                let left_by = &left_by_s.binary().unwrap();
                let right_by = right_by_s.binary().unwrap();
                asof_join_by_binary::<T, A, _>(&left_by, &right_by, left_asof, right_asof, filter)
            }
            _ => {
                if left_by_s.bit_repr_is_large() {
                    let left_by = left_by_s.bit_repr_large();
                    let right_by = right_by_s.bit_repr_large();
                    asof_join_by_numeric::<T, UInt64Type, A, _>(
                        &left_by, &right_by, left_asof, right_asof, filter,
                    )?
                } else {
                    let left_by = left_by_s.bit_repr_small();
                    let right_by = right_by_s.bit_repr_small();
                    asof_join_by_numeric::<T, UInt32Type, A, _>(
                        &left_by, &right_by, left_asof, right_asof, filter,
                    )?
                }
            },
        }
    } else {
        for (lhs, rhs) in left_by.get_columns().iter().zip(right_by.get_columns()) {
            polars_ensure!(lhs.dtype() == rhs.dtype(),
                ComputeError: "mismatching dtypes in 'by' parameter of asof-join: `{}` and `{}`", lhs.dtype(), rhs.dtype()
            );
            #[cfg(feature = "dtype-categorical")]
            _check_categorical_src(lhs.dtype(), rhs.dtype())?;
        }
        todo!()
        // asof_join_by_multiple(
        //     left_by, right_by, left_asof, right_asof, tolerance, strategy,
        // )
    };
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_join_strategy<T: PolarsDataType>(
    left_asof: &ChunkedArray<T>,
    right_asof: &Series,
    left_by: &mut DataFrame,
    right_by: &mut DataFrame,
    strategy: AsofStrategy,
) -> PolarsResult<Vec<Option<IdxSize>>>
where
    for<'a> T::Physical<'a>: PartialOrd
{
    let right_asof = left_asof.unpack_series_matching_type(right_asof)?;

    let filter = |_a: T::Physical<'_>, _b: T::Physical<'_>| true;
    match strategy {
        AsofStrategy::Backward => dispatch_join_by_type::<T, AsofJoinBackwardState, _>(
            left_asof, right_asof, left_by, right_by, filter,
        ),
        AsofStrategy::Forward => dispatch_join_by_type::<T, AsofJoinForwardState, _>(
            left_asof, right_asof, left_by, right_by, filter,
        ),
        AsofStrategy::Nearest => unimplemented!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_join_strategy_numeric<T: PolarsNumericType>(
    left_asof: &ChunkedArray<T>,
    right_asof: &Series,
    left_by: &mut DataFrame,
    right_by: &mut DataFrame,
    strategy: AsofStrategy,
    tolerance: Option<AnyValue<'static>>,
) -> PolarsResult<Vec<Option<IdxSize>>> {
    let right_ca = left_asof.unpack_series_matching_type(right_asof)?;

    if let Some(tol) = tolerance {
        let native_tolerance: T::Native = tol.try_extract()?;
        let abs_tolerance = native_tolerance.abs_diff(T::Native::zero());
        let filter = |a: T::Native, b: T::Native| a.abs_diff(b) <= abs_tolerance;
        match strategy {
            AsofStrategy::Backward => dispatch_join_by_type::<T, AsofJoinBackwardState, _>(
                left_asof, right_ca, left_by, right_by, filter,
            ),
            AsofStrategy::Forward => dispatch_join_by_type::<T, AsofJoinForwardState, _>(
                left_asof, right_ca, left_by, right_by, filter,
            ),
            AsofStrategy::Nearest => dispatch_join_by_type::<T, AsofJoinNearestState, _>(
                left_asof, right_ca, left_by, right_by, filter,
            ),
        }
    } else {
        let filter = |_a: T::Physical<'_>, _b: T::Physical<'_>| true;
        match strategy {
            AsofStrategy::Backward => dispatch_join_by_type::<T, AsofJoinBackwardState, _>(
                left_asof, right_ca, left_by, right_by, filter,
            ),
            AsofStrategy::Forward => dispatch_join_by_type::<T, AsofJoinForwardState, _>(
                left_asof, right_ca, left_by, right_by, filter,
            ),
            AsofStrategy::Nearest => dispatch_join_by_type::<T, AsofJoinNearestState, _>(
                left_asof, right_ca, left_by, right_by, filter,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_join_type(
    left_asof: &Series,
    right_asof: &Series,
    left_by: &mut DataFrame,
    right_by: &mut DataFrame,
    strategy: AsofStrategy,
    tolerance: Option<AnyValue<'static>>,
) -> PolarsResult<Vec<Option<IdxSize>>> {
    match left_asof.dtype() {
        DataType::Int64 => {
            let ca = left_asof.i64().unwrap();
            dispatch_join_strategy_numeric(ca, &right_asof, left_by, right_by, strategy, tolerance)
        },
        DataType::Int32 => {
            let ca = left_asof.i32().unwrap();
            dispatch_join_strategy_numeric(ca, &right_asof, left_by, right_by, strategy, tolerance)
        },
        DataType::UInt64 => {
            let ca = left_asof.u64().unwrap();
            dispatch_join_strategy_numeric(ca, &right_asof, left_by, right_by, strategy, tolerance)
        },
        DataType::UInt32 => {
            let ca = left_asof.u32().unwrap();
            dispatch_join_strategy_numeric(ca, &right_asof, left_by, right_by, strategy, tolerance)
        },
        DataType::Float32 => {
            let ca = left_asof.f32().unwrap();
            dispatch_join_strategy_numeric(ca, &right_asof, left_by, right_by, strategy, tolerance)
        },
        DataType::Float64 => {
            let ca = left_asof.f64().unwrap();
            dispatch_join_strategy_numeric(ca, &right_asof, left_by, right_by, strategy, tolerance)
        },
        DataType::Boolean => {
            let ca = left_asof.bool().unwrap();
            dispatch_join_strategy::<BooleanType>(ca, &right_asof, left_by, right_by, strategy)
        },
        DataType::Binary => {
            let ca = left_asof.binary().unwrap();
            dispatch_join_strategy::<BinaryType>(ca, &right_asof, left_by, right_by, strategy)
        },
        DataType::Utf8 => {
            let ca = left_asof.utf8().unwrap();
            let right_binary = right_asof.cast(&DataType::Binary).unwrap();
            dispatch_join_strategy::<BinaryType>(
                &ca.as_binary(),
                &right_binary,
                left_by,
                right_by,
                strategy,
            )
        },
        _ => {
            let left_asof = left_asof.cast(&DataType::Int32).unwrap();
            let right_asof = right_asof.cast(&DataType::Int32).unwrap();
            let ca = left_asof.i32().unwrap();
            dispatch_join_strategy_numeric(ca, &right_asof, left_by, right_by, strategy, tolerance)
        },
    }
}

pub trait AsofJoinBy: IntoDf {
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    fn _join_asof_by(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
        left_by: Vec<SmartString>,
        right_by: Vec<SmartString>,
        strategy: AsofStrategy,
        tolerance: Option<AnyValue<'static>>,
        suffix: Option<&str>,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<DataFrame> {
        let (self_sliced_slot, other_sliced_slot); // Keeps temporaries alive.
        let (self_df, other_df);
        if let Some((offset, len)) = slice {
            self_sliced_slot = self.to_df().slice(offset, len);
            other_sliced_slot = other.slice(offset, len);
            self_df = &self_sliced_slot;
            other_df = &other_sliced_slot;
        } else {
            self_df = self.to_df();
            other_df = other;
        }

        let left_asof = self_df.column(left_on)?.to_physical_repr();
        let right_asof = other_df.column(right_on)?.to_physical_repr();
        let right_asof_name = right_asof.name();
        let left_asof_name = left_asof.name();
        check_asof_columns(
            &left_asof,
            &right_asof,
            tolerance.is_some(),
            left_by.is_empty() && right_by.is_empty(),
        )?;

        let mut left_by = self_df.select(left_by)?;
        let mut right_by = other_df.select(right_by)?;

        unsafe {
            for (l, r) in left_by
                .get_columns_mut()
                .iter_mut()
                .zip(right_by.get_columns_mut().iter_mut())
            {
                #[cfg(feature = "dtype-categorical")]
                _check_categorical_src(l.dtype(), r.dtype())?;
                *l = l.to_physical_repr().into_owned();
                *r = r.to_physical_repr().into_owned();
            }
        }

        let right_join_tuples = dispatch_join_type(
            &*left_asof,
            &*right_asof,
            &mut left_by,
            &mut right_by,
            strategy,
            tolerance,
        )?;

        let mut drop_these = right_by.get_column_names();
        if left_asof_name == right_asof_name {
            drop_these.push(right_asof_name);
        }

        let cols = other_df
            .get_columns()
            .iter()
            .filter(|s| !drop_these.contains(&s.name()))
            .cloned()
            .collect();
        let proj_other_df = DataFrame::new_no_checks(cols);

        let left = self_df.clone();
        let right_join_tuples = &*right_join_tuples;

        // SAFETY: join tuples are in bounds.
        let right_df = unsafe {
            proj_other_df.take_unchecked(&right_join_tuples.iter().copied().collect_ca(""))
        };

        _finish_join(left, right_df, suffix)
    }

    /// This is similar to a left-join except that we match on nearest key
    /// rather than equal keys. The keys must be sorted to perform an asof join.
    /// This is a special implementation of an asof join that searches for the
    /// nearest keys within a subgroup set by `by`.
    #[allow(clippy::too_many_arguments)]
    fn join_asof_by<I, S>(
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
        let self_df = self.to_df();
        let left_by = left_by.into_iter().map(|s| s.as_ref().into()).collect();
        let right_by = right_by.into_iter().map(|s| s.as_ref().into()).collect();
        self_df._join_asof_by(
            other, left_on, right_on, left_by, right_by, strategy, tolerance, None, None,
        )
    }
}

impl AsofJoinBy for DataFrame {}

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
