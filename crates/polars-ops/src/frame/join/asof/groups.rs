use std::hash::Hash;

use num_traits::Zero;
use polars_core::hashing::_HASHMAP_INIT_SIZE;
use polars_core::prelude::*;
use polars_core::series::BitRepr;
use polars_core::utils::flatten::flatten_nullable;
use polars_core::utils::split_and_flatten;
use polars_core::{with_match_physical_float_polars_type, POOL};
use polars_utils::abs_diff::AbsDiff;
use polars_utils::hashing::{hash_to_partition, DirtyHash};
use polars_utils::nulls::IsNull;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};
use rayon::prelude::*;

use super::*;
use crate::frame::join::{prepare_binary, prepare_keys_multiple};

fn compute_len_offsets<I: IntoIterator<Item = usize>>(iter: I) -> Vec<usize> {
    let mut cumlen = 0;
    iter.into_iter()
        .map(|l| {
            let offset = cumlen;
            cumlen += l;
            offset
        })
        .collect()
}

#[inline(always)]
fn materialize_nullable(idx: Option<IdxSize>) -> NullableIdxSize {
    match idx {
        Some(t) => NullableIdxSize::from(t),
        None => NullableIdxSize::null(),
    }
}

fn asof_in_group<'a, T, A, F>(
    left_val: T::Physical<'a>,
    right_val_arr: &'a T::Array,
    right_grp_idxs: &[IdxSize],
    group_states: &mut PlHashMap<IdxSize, A>,
    filter: F,
    allow_eq: bool,
) -> Option<IdxSize>
where
    T: PolarsDataType,
    A: AsofJoinState<T::Physical<'a>>,
    F: Fn(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    // We use the index of the first element in a group as an identifier to
    // associate with the group state.
    let id = right_grp_idxs.first()?;
    let grp_state = group_states.entry(*id).or_insert_with(|| A::new(allow_eq));

    unsafe {
        let r_grp_idx = grp_state.next(
            &left_val,
            |i| {
                // SAFETY: the group indices are valid, and next() only calls with
                // i < right_grp_idxs.len().
                right_val_arr.get_unchecked(*right_grp_idxs.get_unchecked(i as usize) as usize)
            },
            right_grp_idxs.len() as IdxSize,
        )?;

        // SAFETY: r_grp_idx is valid, as is r_idx (which must be non-null) if
        // we get here.
        let r_idx = *right_grp_idxs.get_unchecked(r_grp_idx as usize);
        let right_val = right_val_arr.value_unchecked(r_idx as usize);
        filter(left_val, right_val).then_some(r_idx)
    }
}

fn asof_join_by_numeric<T, S, A, F>(
    by_left: &ChunkedArray<S>,
    by_right: &ChunkedArray<S>,
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
    filter: F,
    allow_eq: bool,
) -> PolarsResult<IdxArr>
where
    T: PolarsDataType,
    S: PolarsNumericType,
    S::Native: TotalHash + TotalEq + DirtyHash + ToTotalOrd,
    <S::Native as ToTotalOrd>::TotalOrdItem: Send + Sync + Copy + Hash + Eq + DirtyHash + IsNull,
    A: for<'a> AsofJoinState<T::Physical<'a>>,
    F: Sync + for<'a> Fn(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    let (left_asof, right_asof) = POOL.join(|| left_asof.rechunk(), || right_asof.rechunk());
    let left_val_arr = left_asof.downcast_iter().next().unwrap();
    let right_val_arr = right_asof.downcast_iter().next().unwrap();

    let n_threads = POOL.current_num_threads();
    // `strict` is false so that we always flatten. Even if there are more chunks than threads.
    let split_by_left = split_and_flatten(by_left, n_threads);
    let split_by_right = split_and_flatten(by_right, n_threads);
    let offsets = compute_len_offsets(split_by_left.iter().map(|s| s.len()));

    // TODO: handle nulls more efficiently. Right now we just join on the value
    // ignoring the validity mask, and ignore the nulls later.
    let right_slices = split_by_right
        .iter()
        .map(|ca| {
            assert_eq!(ca.chunks().len(), 1);
            ca.downcast_iter().next().unwrap().values_iter().copied()
        })
        .collect();
    let hash_tbls = build_tables(right_slices, false);
    let n_tables = hash_tbls.len();

    // Now we probe the right hand side for each left hand side.
    let out = split_by_left
        .into_par_iter()
        .zip(offsets)
        .map(|(by_left, offset)| {
            let mut results = Vec::with_capacity(by_left.len());
            let mut group_states: PlHashMap<IdxSize, A> =
                PlHashMap::with_capacity(_HASHMAP_INIT_SIZE);

            assert_eq!(by_left.chunks().len(), 1);
            let by_left_chunk = by_left.downcast_iter().next().unwrap();
            for (rel_idx_left, opt_by_left_k) in by_left_chunk.iter().enumerate() {
                let Some(by_left_k) = opt_by_left_k else {
                    results.push(NullableIdxSize::null());
                    continue;
                };
                let by_left_k = by_left_k.to_total_ord();
                let idx_left = (rel_idx_left + offset) as IdxSize;
                let Some(left_val) = left_val_arr.get(idx_left as usize) else {
                    results.push(NullableIdxSize::null());
                    continue;
                };

                let group_probe_table = unsafe {
                    hash_tbls.get_unchecked(hash_to_partition(by_left_k.dirty_hash(), n_tables))
                };
                let Some(right_grp_idxs) = group_probe_table.get(&by_left_k) else {
                    results.push(NullableIdxSize::null());
                    continue;
                };
                let id = asof_in_group::<T, A, &F>(
                    left_val,
                    right_val_arr,
                    right_grp_idxs.as_slice(),
                    &mut group_states,
                    &filter,
                    allow_eq,
                );
                results.push(materialize_nullable(id));
            }
            results
        });

    let bufs = POOL.install(|| out.collect::<Vec<_>>());
    Ok(flatten_nullable(&bufs))
}

fn asof_join_by_binary<B, T, A, F>(
    by_left: &ChunkedArray<B>,
    by_right: &ChunkedArray<B>,
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
    filter: F,
    allow_eq: bool,
) -> IdxArr
where
    B: PolarsDataType,
    for<'b> <B::Array as StaticArray>::ValueT<'b>: AsRef<[u8]>,
    T: PolarsDataType,
    A: for<'a> AsofJoinState<T::Physical<'a>>,
    F: Sync + for<'a> Fn(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    let (left_asof, right_asof) = POOL.join(|| left_asof.rechunk(), || right_asof.rechunk());
    let left_val_arr = left_asof.downcast_iter().next().unwrap();
    let right_val_arr = right_asof.downcast_iter().next().unwrap();

    let (prep_by_left, prep_by_right, _, _) = prepare_binary::<B>(by_left, by_right, false);
    let offsets = compute_len_offsets(prep_by_left.iter().map(|s| s.len()));
    let hash_tbls = build_tables(prep_by_right, false);
    let n_tables = hash_tbls.len();

    // Now we probe the right hand side for each left hand side.
    let iter = prep_by_left
        .into_par_iter()
        .zip(offsets)
        .map(|(by_left, offset)| {
            let mut results = Vec::with_capacity(by_left.len());
            let mut group_states: PlHashMap<_, A> = PlHashMap::with_capacity(_HASHMAP_INIT_SIZE);

            for (rel_idx_left, by_left_k) in by_left.iter().enumerate() {
                let idx_left = (rel_idx_left + offset) as IdxSize;
                let Some(left_val) = left_val_arr.get(idx_left as usize) else {
                    results.push(NullableIdxSize::null());
                    continue;
                };

                let group_probe_table = unsafe {
                    hash_tbls.get_unchecked(hash_to_partition(by_left_k.dirty_hash(), n_tables))
                };
                let Some(right_grp_idxs) = group_probe_table.get(by_left_k) else {
                    results.push(NullableIdxSize::null());
                    continue;
                };
                let id = asof_in_group::<T, A, &F>(
                    left_val,
                    right_val_arr,
                    right_grp_idxs.as_slice(),
                    &mut group_states,
                    &filter,
                    allow_eq,
                );

                results.push(materialize_nullable(id));
            }
            results
        });
    let bufs = POOL.install(|| iter.collect::<Vec<_>>());
    flatten_nullable(&bufs)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_join_by_type<T, A, F>(
    left_asof: &ChunkedArray<T>,
    right_asof: &ChunkedArray<T>,
    left_by: &mut DataFrame,
    right_by: &mut DataFrame,
    filter: F,
    allow_eq: bool,
) -> PolarsResult<IdxArr>
where
    T: PolarsDataType,
    A: for<'a> AsofJoinState<T::Physical<'a>>,
    F: Sync + for<'a> Fn(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    let out = if left_by.width() == 1 {
        let left_by_s = left_by.get_columns()[0].to_physical_repr();
        let right_by_s = right_by.get_columns()[0].to_physical_repr();
        let left_dtype = left_by_s.dtype();
        let right_dtype = right_by_s.dtype();
        polars_ensure!(left_dtype == right_dtype,
            ComputeError: "mismatching dtypes in 'by' parameter of asof-join: `{left_dtype}` and `{right_dtype}`",
        );
        match left_dtype {
            DataType::String => {
                let left_by = &left_by_s.str().unwrap().as_binary();
                let right_by = right_by_s.str().unwrap().as_binary();
                asof_join_by_binary::<BinaryType, T, A, F>(
                    left_by, &right_by, left_asof, right_asof, filter, allow_eq,
                )
            },
            DataType::Binary => {
                let left_by = &left_by_s.binary().unwrap();
                let right_by = right_by_s.binary().unwrap();
                asof_join_by_binary::<BinaryType, T, A, F>(
                    left_by, right_by, left_asof, right_asof, filter, allow_eq,
                )
            },
            x if x.is_float() => {
                with_match_physical_float_polars_type!(left_by_s.dtype(), |$T| {
                    let left_by: &ChunkedArray<$T> = left_by_s.as_materialized_series().as_ref().as_ref().as_ref();
                    let right_by: &ChunkedArray<$T> = right_by_s.as_materialized_series().as_ref().as_ref().as_ref();
                    asof_join_by_numeric::<T, $T, A, F>(
                        left_by, right_by, left_asof, right_asof, filter, allow_eq
                    )?
                })
            },
            _ => {
                let left_by = left_by_s.bit_repr();
                let right_by = right_by_s.bit_repr();

                let (Some(left_by), Some(right_by)) = (left_by, right_by) else {
                    polars_bail!(nyi = "Dispatch join for {left_dtype} and {right_dtype}");
                };

                use BitRepr as B;
                match (left_by, right_by) {
                    (B::Small(left_by), B::Small(right_by)) => {
                        asof_join_by_numeric::<T, UInt32Type, A, F>(
                            &left_by, &right_by, left_asof, right_asof, filter, allow_eq,
                        )?
                    },
                    (B::Large(left_by), B::Large(right_by)) => {
                        asof_join_by_numeric::<T, UInt64Type, A, F>(
                            &left_by, &right_by, left_asof, right_asof, filter, allow_eq,
                        )?
                    },
                    // We have already asserted that the datatypes are the same.
                    _ => unreachable!(),
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

        // TODO: @scalar-opt.
        let left_by_series: Vec<_> = left_by.materialized_column_iter().cloned().collect();
        let right_by_series: Vec<_> = right_by.materialized_column_iter().cloned().collect();
        let lhs_keys = prepare_keys_multiple(&left_by_series, false)?;
        let rhs_keys = prepare_keys_multiple(&right_by_series, false)?;
        asof_join_by_binary::<BinaryOffsetType, T, A, F>(
            &lhs_keys, &rhs_keys, left_asof, right_asof, filter, allow_eq,
        )
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
    allow_eq: bool,
) -> PolarsResult<IdxArr>
where
    for<'a> T::Physical<'a>: PartialOrd,
{
    let right_asof = left_asof.unpack_series_matching_type(right_asof)?;

    let filter = |_a: T::Physical<'_>, _b: T::Physical<'_>| true;
    match strategy {
        AsofStrategy::Backward => dispatch_join_by_type::<T, AsofJoinBackwardState, _>(
            left_asof, right_asof, left_by, right_by, filter, allow_eq,
        ),
        AsofStrategy::Forward => dispatch_join_by_type::<T, AsofJoinForwardState, _>(
            left_asof, right_asof, left_by, right_by, filter, allow_eq,
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
    allow_eq: bool,
) -> PolarsResult<IdxArr> {
    let right_ca = left_asof.unpack_series_matching_type(right_asof)?;

    if let Some(tol) = tolerance {
        let native_tolerance: T::Native = tol.try_extract()?;
        let abs_tolerance = native_tolerance.abs_diff(T::Native::zero());
        let filter = |a: T::Native, b: T::Native| a.abs_diff(b) <= abs_tolerance;
        match strategy {
            AsofStrategy::Backward => dispatch_join_by_type::<T, AsofJoinBackwardState, _>(
                left_asof, right_ca, left_by, right_by, filter, allow_eq,
            ),
            AsofStrategy::Forward => dispatch_join_by_type::<T, AsofJoinForwardState, _>(
                left_asof, right_ca, left_by, right_by, filter, allow_eq,
            ),
            AsofStrategy::Nearest => dispatch_join_by_type::<T, AsofJoinNearestState, _>(
                left_asof, right_ca, left_by, right_by, filter, allow_eq,
            ),
        }
    } else {
        let filter = |_a: T::Physical<'_>, _b: T::Physical<'_>| true;
        match strategy {
            AsofStrategy::Backward => dispatch_join_by_type::<T, AsofJoinBackwardState, _>(
                left_asof, right_ca, left_by, right_by, filter, allow_eq,
            ),
            AsofStrategy::Forward => dispatch_join_by_type::<T, AsofJoinForwardState, _>(
                left_asof, right_ca, left_by, right_by, filter, allow_eq,
            ),
            AsofStrategy::Nearest => dispatch_join_by_type::<T, AsofJoinNearestState, _>(
                left_asof, right_ca, left_by, right_by, filter, allow_eq,
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
    allow_eq: bool,
) -> PolarsResult<IdxArr> {
    match left_asof.dtype() {
        DataType::Int64 => {
            let ca = left_asof.i64().unwrap();
            dispatch_join_strategy_numeric(
                ca, right_asof, left_by, right_by, strategy, tolerance, allow_eq,
            )
        },
        DataType::Int32 => {
            let ca = left_asof.i32().unwrap();
            dispatch_join_strategy_numeric(
                ca, right_asof, left_by, right_by, strategy, tolerance, allow_eq,
            )
        },
        DataType::UInt64 => {
            let ca = left_asof.u64().unwrap();
            dispatch_join_strategy_numeric(
                ca, right_asof, left_by, right_by, strategy, tolerance, allow_eq,
            )
        },
        DataType::UInt32 => {
            let ca = left_asof.u32().unwrap();
            dispatch_join_strategy_numeric(
                ca, right_asof, left_by, right_by, strategy, tolerance, allow_eq,
            )
        },
        DataType::Float32 => {
            let ca = left_asof.f32().unwrap();
            dispatch_join_strategy_numeric(
                ca, right_asof, left_by, right_by, strategy, tolerance, allow_eq,
            )
        },
        DataType::Float64 => {
            let ca = left_asof.f64().unwrap();
            dispatch_join_strategy_numeric(
                ca, right_asof, left_by, right_by, strategy, tolerance, allow_eq,
            )
        },
        DataType::Boolean => {
            let ca = left_asof.bool().unwrap();
            dispatch_join_strategy::<BooleanType>(
                ca, right_asof, left_by, right_by, strategy, allow_eq,
            )
        },
        DataType::Binary => {
            let ca = left_asof.binary().unwrap();
            dispatch_join_strategy::<BinaryType>(
                ca, right_asof, left_by, right_by, strategy, allow_eq,
            )
        },
        DataType::String => {
            let ca = left_asof.str().unwrap();
            let right_binary = right_asof.cast(&DataType::Binary).unwrap();
            dispatch_join_strategy::<BinaryType>(
                &ca.as_binary(),
                &right_binary,
                left_by,
                right_by,
                strategy,
                allow_eq,
            )
        },
        _ => {
            let left_asof = left_asof.cast(&DataType::Int32).unwrap();
            let right_asof = right_asof.cast(&DataType::Int32).unwrap();
            let ca = left_asof.i32().unwrap();
            dispatch_join_strategy_numeric(
                ca,
                &right_asof,
                left_by,
                right_by,
                strategy,
                tolerance,
                allow_eq,
            )
        },
    }
}

pub trait AsofJoinBy: IntoDf {
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    fn _join_asof_by(
        &self,
        other: &DataFrame,
        left_on: &Series,
        right_on: &Series,
        left_by: Vec<PlSmallStr>,
        right_by: Vec<PlSmallStr>,
        strategy: AsofStrategy,
        tolerance: Option<AnyValue<'static>>,
        suffix: Option<PlSmallStr>,
        slice: Option<(i64, usize)>,
        coalesce: bool,
        allow_eq: bool,
        check_sortedness: bool,
    ) -> PolarsResult<DataFrame> {
        let (self_sliced_slot, other_sliced_slot, left_slice_s, right_slice_s); // Keeps temporaries alive.
        let (self_df, other_df, left_key, right_key);
        if let Some((offset, len)) = slice {
            self_sliced_slot = self.to_df().slice(offset, len);
            other_sliced_slot = other.slice(offset, len);
            left_slice_s = left_on.slice(offset, len);
            right_slice_s = right_on.slice(offset, len);
            left_key = &left_slice_s;
            right_key = &right_slice_s;
            self_df = &self_sliced_slot;
            other_df = &other_sliced_slot;
        } else {
            self_df = self.to_df();
            other_df = other;
            left_key = left_on;
            right_key = right_on;
        }

        let left_asof = left_key.to_physical_repr();
        let right_asof = right_key.to_physical_repr();
        let right_asof_name = right_asof.name();
        let left_asof_name = left_asof.name();
        check_asof_columns(
            &left_asof,
            &right_asof,
            tolerance.is_some(),
            left_by.is_empty() && right_by.is_empty(),
            check_sortedness,
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
                *l = l.to_physical_repr();
                *r = r.to_physical_repr();
            }
        }

        let right_join_tuples = dispatch_join_type(
            &left_asof,
            &right_asof,
            &mut left_by,
            &mut right_by,
            strategy,
            tolerance,
            allow_eq,
        )?;

        let mut drop_these = right_by.get_column_names();
        if coalesce && left_asof_name == right_asof_name {
            drop_these.push(right_asof_name);
        }

        let cols = other_df
            .get_columns()
            .iter()
            .filter(|s| !drop_these.contains(&s.name()))
            .cloned()
            .collect();
        let proj_other_df = unsafe { DataFrame::new_no_checks(other_df.height(), cols) };

        let left = self_df.clone();

        // SAFETY: join tuples are in bounds.
        let right_df = unsafe {
            proj_other_df.take_unchecked(&IdxCa::with_chunk(PlSmallStr::EMPTY, right_join_tuples))
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
        allow_eq: bool,
        check_sortedness: bool,
    ) -> PolarsResult<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let self_df = self.to_df();
        let left_by = left_by.into_iter().map(|s| s.as_ref().into()).collect();
        let right_by = right_by.into_iter().map(|s| s.as_ref().into()).collect();
        let left_key = self_df.column(left_on)?.as_materialized_series();
        let right_key = other.column(right_on)?.as_materialized_series();
        self_df._join_asof_by(
            other,
            left_key,
            right_key,
            left_by,
            right_by,
            strategy,
            tolerance,
            None,
            None,
            true,
            allow_eq,
            check_sortedness,
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

        let out = a.join_asof_by(
            &b,
            "a",
            "a",
            ["b"],
            ["b"],
            AsofStrategy::Backward,
            None,
            true,
            true,
        )?;
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
            true,
            true,
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
            true,
            true,
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

        let out = a.join_asof_by(
            &b,
            "a",
            "a",
            ["b"],
            ["b"],
            AsofStrategy::Forward,
            None,
            true,
            true,
        )?;
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
            true,
            true,
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
            true,
            true,
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
            true,
            true,
        )?;
        let a = out.column("bid_right").unwrap();
        let a = a.f64().unwrap();

        assert_eq!(Vec::from(a), expected);

        Ok(())
    }
}
