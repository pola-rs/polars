use polars_core::functions::concat_df_horizontal;

use super::*;
use crate::prelude::*;

pub(super) fn left_join_from_series<P>(
    left: DataFrame,
    right: &DataFrame,
    s_left: &Series,
    s_right: &Series,
    eval_extra_predicates: P,
    args: JoinArgs,
    verbose: bool,
    drop_names: Option<Vec<PlSmallStr>>,
) -> PolarsResult<DataFrame>
where
    P: FnMut(&DataFrame) -> PolarsResult<bool>,
{
    let (df_left, df_right) = materialize_left_join_from_series(
        left,
        right,
        s_left,
        s_right,
        eval_extra_predicates,
        &args,
        verbose,
        drop_names,
    )?;
    _finish_join(df_left, df_right, args.suffix)
}

pub(super) fn right_join_from_series<P>(
    left: &DataFrame,
    right: DataFrame,
    s_left: &Series,
    s_right: &Series,
    eval_extra_predicates: P,
    args: JoinArgs,
    verbose: bool,
    drop_names: Option<Vec<PlSmallStr>>,
) -> PolarsResult<DataFrame>
where
    P: FnMut(&DataFrame) -> PolarsResult<bool>,
{
    // Swap the order of tables to do a right join.
    let (df_right, df_left) = materialize_left_join_from_series(
        right,
        left,
        s_right,
        s_left,
        eval_extra_predicates,
        &args,
        verbose,
        drop_names,
    )?;
    _finish_join(df_left, df_right, args.suffix)
}

pub fn materialize_left_join_from_series<P>(
    mut left: DataFrame,
    right_: &DataFrame,
    s_left: &Series,
    s_right: &Series,
    eval_extra_predicates: P,
    args: &JoinArgs,
    verbose: bool,
    drop_names: Option<Vec<PlSmallStr>>,
) -> PolarsResult<(DataFrame, DataFrame)>
where
    P: FnMut(&DataFrame) -> PolarsResult<bool>,
{
    #[cfg(feature = "dtype-categorical")]
    _check_categorical_src(s_left.dtype(), s_right.dtype())?;

    let mut s_left = s_left.clone();
    // Eagerly limit left if possible.
    if let Some((offset, len)) = args.slice {
        if offset == 0 {
            left = left.slice(0, len);
            s_left = s_left.slice(0, len);
        }
    }

    // Ensure that the chunks are aligned otherwise we go OOB.
    let mut right = Cow::Borrowed(right_);
    let mut s_right = s_right.clone();
    if left.should_rechunk() {
        left.as_single_chunk_par();
        s_left = s_left.rechunk();
    }
    if right.should_rechunk() {
        let mut other = right_.clone();
        other.as_single_chunk_par();
        right = Cow::Owned(other);
        s_right = s_right.rechunk();
    }

    let ids = sort_or_hash_left(&s_left, &s_right, verbose, args.validation, args.join_nulls)?;
    let ids = apply_extra_predicates(ids, &left, &right, eval_extra_predicates)?;
    let right = if let Some(drop_names) = drop_names {
        right.drop_many(drop_names)
    } else {
        right.drop(s_right.name()).unwrap()
    };
    Ok(materialize_left_join(&left, &right, ids, args))
}

fn apply_extra_predicates<P>(
    ids: LeftJoinIds,
    left: &DataFrame,
    right: &DataFrame,
    mut eval_extra_predicates: P,
) -> PolarsResult<LeftJoinIds>
where
    P: FnMut(&DataFrame) -> PolarsResult<bool>,
{
    let left_ids = ids.0.left().unwrap();
    let right_ids = ids.1.left().unwrap();
    debug_assert!(left_ids.len() == right_ids.len());

    // Construct single row DFs from selected left and right columns in order to evaluate the predicates
    // This will be terrible for performance
    let mut filtered_left_ids = Vec::with_capacity(left_ids.len());
    let mut filtered_right_ids = Vec::with_capacity(right_ids.len());

    let mut prev_left_id = None;
    let mut match_found = false;
    for (left_id, right_id) in left_ids.into_iter().zip(right_ids) {
        if let Some(prev_left_id) = prev_left_id {
            if prev_left_id != left_id {
                if !match_found {
                    filtered_left_ids.push(prev_left_id);
                    filtered_right_ids.push(NullableIdxSize::null());
                }
                match_found = false;
            }
        }

        if right_id.is_null_idx() {
            filtered_left_ids.push(left_id);
            filtered_right_ids.push(right_id);
            prev_left_id = None;
            match_found = false;
        } else {
            let left_row = unsafe { left.take_slice_unchecked(&[left_id]) };
            let right_row = unsafe { right.take_slice_unchecked(&[right_id.idx()]) };
            let combined_df = concat_df_horizontal(&[left_row, right_row], true)?;
            let predicate_result = eval_extra_predicates(&combined_df)?;
            if predicate_result {
                filtered_left_ids.push(left_id);
                filtered_right_ids.push(right_id);
                match_found = true;
            }
            prev_left_id = Some(left_id);
        }
    }

    if let Some(prev_left_id) = prev_left_id {
        if !match_found {
            filtered_left_ids.push(prev_left_id);
            filtered_right_ids.push(NullableIdxSize::null());
        }
    }

    Ok((
        ChunkJoinIds::Left(filtered_left_ids),
        ChunkJoinOptIds::Left(filtered_right_ids),
    ))
}

#[cfg(feature = "chunked_ids")]
fn materialize_left_join(
    left: &DataFrame,
    other: &DataFrame,
    ids: LeftJoinIds,
    args: &JoinArgs,
) -> (DataFrame, DataFrame) {
    let (left_idx, right_idx) = ids;
    let materialize_left = || match left_idx {
        ChunkJoinIds::Left(left_idx) => unsafe {
            let mut left_idx = &*left_idx;
            if let Some((offset, len)) = args.slice {
                left_idx = slice_slice(left_idx, offset, len);
            }
            left._create_left_df_from_slice(left_idx, true, args.slice.is_some(), true)
        },
        ChunkJoinIds::Right(left_idx) => unsafe {
            let mut left_idx = &*left_idx;
            if let Some((offset, len)) = args.slice {
                left_idx = slice_slice(left_idx, offset, len);
            }
            left.create_left_df_chunked(left_idx, true, args.slice.is_some())
        },
    };

    let materialize_right = || match right_idx {
        ChunkJoinOptIds::Left(right_idx) => unsafe {
            let mut right_idx = &*right_idx;
            if let Some((offset, len)) = args.slice {
                right_idx = slice_slice(right_idx, offset, len);
            }
            IdxCa::with_nullable_idx(right_idx, |idx| other.take_unchecked(idx))
        },
        ChunkJoinOptIds::Right(right_idx) => unsafe {
            let mut right_idx = &*right_idx;
            if let Some((offset, len)) = args.slice {
                right_idx = slice_slice(right_idx, offset, len);
            }
            other._take_opt_chunked_unchecked_hor_par(right_idx)
        },
    };
    POOL.join(materialize_left, materialize_right)
}

#[cfg(not(feature = "chunked_ids"))]
fn materialize_left_join(
    left: &DataFrame,
    other: &DataFrame,
    ids: LeftJoinIds,
    args: &JoinArgs,
) -> (DataFrame, DataFrame) {
    let (left_idx, right_idx) = ids;

    let mut left_idx = &*left_idx;
    if let Some((offset, len)) = args.slice {
        left_idx = slice_slice(left_idx, offset, len);
    }
    let materialize_left =
        || unsafe { left._create_left_df_from_slice(&left_idx, true, args.slice.is_some(), true) };

    let mut right_idx = &*right_idx;
    if let Some((offset, len)) = args.slice {
        right_idx = slice_slice(right_idx, offset, len);
    }
    let materialize_right = || {
        let right_idx = &*right_idx;
        unsafe { IdxCa::with_nullable_idx(right_idx, |idx| other.take_unchecked(idx)) }
    };
    POOL.join(materialize_left, materialize_right)
}
