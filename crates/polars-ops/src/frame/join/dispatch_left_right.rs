use super::*;
use crate::prelude::*;

pub(super) fn left_join_from_series(
    left: DataFrame,
    right: &DataFrame,
    s_left: &Series,
    s_right: &Series,
    args: JoinArgs,
    verbose: bool,
    drop_names: Option<&[&str]>,
) -> PolarsResult<DataFrame> {
    let (df_left, df_right) = materialize_left_join_from_series(
        left, right, s_left, s_right, &args, verbose, drop_names,
    )?;
    _finish_join(df_left, df_right, args.suffix.as_deref())
}

pub(super) fn right_join_from_series(
    left: &DataFrame,
    right: DataFrame,
    s_left: &Series,
    s_right: &Series,
    args: JoinArgs,
    verbose: bool,
    drop_names: Option<&[&str]>,
) -> PolarsResult<DataFrame> {
    // Swap the order of tables to do a right join.
    let (df_right, df_left) = materialize_left_join_from_series(
        right, left, s_right, s_left, &args, verbose, drop_names,
    )?;
    _finish_join(df_left, df_right, args.suffix.as_deref())
}

pub fn materialize_left_join_from_series(
    mut left: DataFrame,
    right_: &DataFrame,
    s_left: &Series,
    s_right: &Series,
    args: &JoinArgs,
    verbose: bool,
    drop_names: Option<&[&str]>,
) -> PolarsResult<(DataFrame, DataFrame)> {
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
    let right = if let Some(drop_names) = drop_names {
        right.drop_many(drop_names)
    } else {
        right.drop(s_right.name()).unwrap()
    };
    Ok(materialize_left_join(&left, &right, ids, args))
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
            left._create_left_df_from_slice(left_idx, true, true)
        },
        ChunkJoinIds::Right(left_idx) => unsafe {
            let mut left_idx = &*left_idx;
            if let Some((offset, len)) = args.slice {
                left_idx = slice_slice(left_idx, offset, len);
            }
            left.create_left_df_chunked(left_idx, true)
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
            other._take_opt_chunked_unchecked(right_idx)
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
    let materialize_left = || unsafe { left._create_left_df_from_slice(&left_idx, true, true) };

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
