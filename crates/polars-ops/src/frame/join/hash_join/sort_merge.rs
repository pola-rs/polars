#[cfg(feature = "performant")]
use arrow::legacy::kernels::sorted_join;
#[cfg(feature = "performant")]
use polars_core::utils::_split_offsets;
#[cfg(feature = "performant")]
use polars_core::utils::flatten::flatten_par;

use super::*;

#[cfg(feature = "performant")]
fn par_sorted_merge_left_impl<T>(
    s_left: &ChunkedArray<T>,
    s_right: &ChunkedArray<T>,
) -> (Vec<IdxSize>, Vec<NullableIdxSize>)
where
    T: PolarsNumericType,
{
    let offsets = _split_offsets(s_left.len(), POOL.current_num_threads());
    let s_left = s_left.rechunk();
    let s_right = s_right.rechunk();

    // we can unwrap because we should not have nulls
    let slice_left = s_left.cont_slice().unwrap();
    let slice_right = s_right.cont_slice().unwrap();

    let indexes = offsets.into_par_iter().map(|(offset, len)| {
        let slice_left = &slice_left[offset..offset + len];
        sorted_join::left::join(slice_left, slice_right, offset as IdxSize)
    });
    let indexes = POOL.install(|| indexes.collect::<Vec<_>>());

    let lefts = indexes.iter().map(|t| &t.0).collect::<Vec<_>>();
    let rights = indexes.iter().map(|t| &t.1).collect::<Vec<_>>();

    (flatten_par(&lefts), flatten_par(&rights))
}

#[cfg(feature = "performant")]
pub(super) fn par_sorted_merge_left(
    s_left: &Series,
    s_right: &Series,
) -> (Vec<IdxSize>, Vec<NullableIdxSize>) {
    // Don't use bit_repr here. It messes up sortedness.
    debug_assert_eq!(s_left.dtype(), s_right.dtype());
    let s_left = s_left.to_physical_repr();
    let s_right = s_right.to_physical_repr();

    match s_left.dtype() {
        #[cfg(feature = "dtype-i8")]
        DataType::Int8 => par_sorted_merge_left_impl(s_left.i8().unwrap(), s_right.i8().unwrap()),
        #[cfg(feature = "dtype-u8")]
        DataType::UInt8 => par_sorted_merge_left_impl(s_left.u8().unwrap(), s_right.u8().unwrap()),
        #[cfg(feature = "dtype-u16")]
        DataType::UInt16 => {
            par_sorted_merge_left_impl(s_left.u16().unwrap(), s_right.u16().unwrap())
        },
        #[cfg(feature = "dtype-i16")]
        DataType::Int16 => {
            par_sorted_merge_left_impl(s_left.i16().unwrap(), s_right.i16().unwrap())
        },
        DataType::UInt32 => {
            par_sorted_merge_left_impl(s_left.u32().unwrap(), s_right.u32().unwrap())
        },
        DataType::Int32 => {
            par_sorted_merge_left_impl(s_left.i32().unwrap(), s_right.i32().unwrap())
        },
        DataType::UInt64 => {
            par_sorted_merge_left_impl(s_left.u64().unwrap(), s_right.u64().unwrap())
        },
        DataType::Int64 => {
            par_sorted_merge_left_impl(s_left.i64().unwrap(), s_right.i64().unwrap())
        },
        DataType::Float32 => {
            par_sorted_merge_left_impl(s_left.f32().unwrap(), s_right.f32().unwrap())
        },
        DataType::Float64 => {
            par_sorted_merge_left_impl(s_left.f64().unwrap(), s_right.f64().unwrap())
        },
        _ => unreachable!(),
    }
}
#[cfg(feature = "performant")]
fn par_sorted_merge_inner_impl<T>(
    s_left: &ChunkedArray<T>,
    s_right: &ChunkedArray<T>,
) -> (Vec<IdxSize>, Vec<IdxSize>)
where
    T: PolarsNumericType,
{
    let offsets = _split_offsets(s_left.len(), POOL.current_num_threads());
    let s_left = s_left.rechunk();
    let s_right = s_right.rechunk();

    // we can unwrap because we should not have nulls
    let slice_left = s_left.cont_slice().unwrap();
    let slice_right = s_right.cont_slice().unwrap();

    let indexes = offsets.into_par_iter().map(|(offset, len)| {
        let slice_left = &slice_left[offset..offset + len];
        sorted_join::inner::join(slice_left, slice_right, offset as IdxSize)
    });
    let indexes = POOL.install(|| indexes.collect::<Vec<_>>());

    let lefts = indexes.iter().map(|t| &t.0).collect::<Vec<_>>();
    let rights = indexes.iter().map(|t| &t.1).collect::<Vec<_>>();

    (flatten_par(&lefts), flatten_par(&rights))
}

#[cfg(feature = "performant")]
pub(super) fn par_sorted_merge_inner_no_nulls(
    s_left: &Series,
    s_right: &Series,
) -> (Vec<IdxSize>, Vec<IdxSize>) {
    // Don't use bit_repr here. It messes up sortedness.
    debug_assert_eq!(s_left.dtype(), s_right.dtype());
    let s_left = s_left.to_physical_repr();
    let s_right = s_right.to_physical_repr();

    match s_left.dtype() {
        #[cfg(feature = "dtype-i8")]
        DataType::Int8 => par_sorted_merge_inner_impl(s_left.i8().unwrap(), s_right.i8().unwrap()),
        #[cfg(feature = "dtype-u8")]
        DataType::UInt8 => par_sorted_merge_inner_impl(s_left.u8().unwrap(), s_right.u8().unwrap()),
        #[cfg(feature = "dtype-u16")]
        DataType::UInt16 => {
            par_sorted_merge_inner_impl(s_left.u16().unwrap(), s_right.u16().unwrap())
        },
        #[cfg(feature = "dtype-i16")]
        DataType::Int16 => {
            par_sorted_merge_inner_impl(s_left.i16().unwrap(), s_right.i16().unwrap())
        },
        DataType::UInt32 => {
            par_sorted_merge_inner_impl(s_left.u32().unwrap(), s_right.u32().unwrap())
        },
        DataType::Int32 => {
            par_sorted_merge_inner_impl(s_left.i32().unwrap(), s_right.i32().unwrap())
        },
        DataType::UInt64 => {
            par_sorted_merge_inner_impl(s_left.u64().unwrap(), s_right.u64().unwrap())
        },
        DataType::Int64 => {
            par_sorted_merge_inner_impl(s_left.i64().unwrap(), s_right.i64().unwrap())
        },
        DataType::Float32 => {
            par_sorted_merge_inner_impl(s_left.f32().unwrap(), s_right.f32().unwrap())
        },
        DataType::Float64 => {
            par_sorted_merge_inner_impl(s_left.f64().unwrap(), s_right.f64().unwrap())
        },
        _ => unreachable!(),
    }
}

#[cfg(feature = "performant")]
fn to_left_join_ids(left_idx: Vec<IdxSize>, right_idx: Vec<NullableIdxSize>) -> LeftJoinIds {
    #[cfg(feature = "chunked_ids")]
    {
        (Either::Left(left_idx), Either::Left(right_idx))
    }

    #[cfg(not(feature = "chunked_ids"))]
    {
        (left_idx, right_idx)
    }
}

#[cfg(feature = "performant")]
fn create_reverse_map_from_arg_sort(mut arg_sort: IdxCa) -> Vec<IdxSize> {
    let arr = unsafe { arg_sort.chunks_mut() }.pop().unwrap();
    primitive_to_vec::<IdxSize>(arr).unwrap()
}

#[cfg(not(feature = "performant"))]
pub(crate) fn _sort_or_hash_inner(
    s_left: &Series,
    s_right: &Series,
    _verbose: bool,
    validate: JoinValidation,
    join_nulls: bool,
) -> PolarsResult<(InnerJoinIds, bool)> {
    s_left.hash_join_inner(s_right, validate, join_nulls)
}

#[cfg(feature = "performant")]
pub(crate) fn _sort_or_hash_inner(
    s_left: &Series,
    s_right: &Series,
    verbose: bool,
    validate: JoinValidation,
    join_nulls: bool,
) -> PolarsResult<(InnerJoinIds, bool)> {
    // We check if keys are sorted.
    // - If they are we can do a sorted merge join
    // If one of the keys is not, it can still be faster to sort that key and use
    // the `arg_sort` indices to revert the sort once the join keys are determined.
    let size_factor_rhs = s_right.len() as f32 / s_left.len() as f32;
    let size_factor_lhs = s_left.len() as f32 / s_right.len() as f32;
    let size_factor_acceptable = std::env::var("POLARS_JOIN_SORT_FACTOR")
        .map(|s| s.parse::<f32>().unwrap())
        .unwrap_or(1.0);
    let is_numeric = s_left.dtype().to_physical().is_numeric();

    if validate.needs_checks() {
        return s_left.hash_join_inner(s_right, validate, join_nulls);
    }

    let no_nulls = s_left.null_count() == 0 && s_right.null_count() == 0;
    match (s_left.is_sorted_flag(), s_right.is_sorted_flag(), no_nulls) {
        (IsSorted::Ascending, IsSorted::Ascending, true) if is_numeric => {
            if verbose {
                eprintln!("inner join: keys are sorted: use sorted merge join");
            }
            Ok((par_sorted_merge_inner_no_nulls(s_left, s_right), true))
        },
        (IsSorted::Ascending, _, true)
            if is_numeric && size_factor_rhs < size_factor_acceptable =>
        {
            if verbose {
                eprintln!("right key will be descending sorted in inner join operation.")
            }

            let sort_idx = s_right.arg_sort(SortOptions {
                descending: false,
                nulls_last: false,
                multithreaded: true,
                maintain_order: false,
            });
            let s_right = unsafe { s_right.take_unchecked(&sort_idx) };
            let ids = par_sorted_merge_inner_no_nulls(s_left, &s_right);
            let reverse_idx_map = create_reverse_map_from_arg_sort(sort_idx);

            let (left, mut right) = ids;

            POOL.install(|| {
                right.par_iter_mut().for_each(|idx| {
                    *idx = unsafe { *reverse_idx_map.get_unchecked(*idx as usize) };
                });
            });

            Ok(((left, right), true))
        },
        (_, IsSorted::Ascending, true)
            if is_numeric && size_factor_lhs < size_factor_acceptable =>
        {
            if verbose {
                eprintln!("left key will be descending sorted in inner join operation.")
            }

            let sort_idx = s_left.arg_sort(SortOptions {
                descending: false,
                nulls_last: false,
                multithreaded: true,
                maintain_order: false,
            });
            let s_left = unsafe { s_left.take_unchecked(&sort_idx) };
            let ids = par_sorted_merge_inner_no_nulls(&s_left, s_right);
            let reverse_idx_map = create_reverse_map_from_arg_sort(sort_idx);

            let (mut left, right) = ids;

            POOL.install(|| {
                left.par_iter_mut().for_each(|idx| {
                    *idx = unsafe { *reverse_idx_map.get_unchecked(*idx as usize) };
                });
            });

            // set sorted to `false` as we descending sorted the left key.
            Ok(((left, right), false))
        },
        _ => s_left.hash_join_inner(s_right, validate, join_nulls),
    }
}

#[cfg(not(feature = "performant"))]
pub(crate) fn sort_or_hash_left(
    s_left: &Series,
    s_right: &Series,
    _verbose: bool,
    validate: JoinValidation,
    join_nulls: bool,
) -> PolarsResult<LeftJoinIds> {
    s_left.hash_join_left(s_right, validate, join_nulls)
}

#[cfg(feature = "performant")]
pub(crate) fn sort_or_hash_left(
    s_left: &Series,
    s_right: &Series,
    verbose: bool,
    validate: JoinValidation,
    join_nulls: bool,
) -> PolarsResult<LeftJoinIds> {
    if validate.needs_checks() {
        return s_left.hash_join_left(s_right, validate, join_nulls);
    }

    let size_factor_rhs = s_right.len() as f32 / s_left.len() as f32;
    let size_factor_acceptable = std::env::var("POLARS_JOIN_SORT_FACTOR")
        .map(|s| s.parse::<f32>().unwrap())
        .unwrap_or(1.0);
    let is_numeric = s_left.dtype().to_physical().is_numeric();

    let no_nulls = s_left.null_count() == 0 && s_right.null_count() == 0;

    match (s_left.is_sorted_flag(), s_right.is_sorted_flag(), no_nulls) {
        (IsSorted::Ascending, IsSorted::Ascending, true) if is_numeric => {
            if verbose {
                eprintln!("left join: keys are sorted: use sorted merge join");
            }
            let (left_idx, right_idx) = par_sorted_merge_left(s_left, s_right);
            Ok(to_left_join_ids(left_idx, right_idx))
        },
        (IsSorted::Ascending, _, true)
            if is_numeric && size_factor_rhs < size_factor_acceptable =>
        {
            if verbose {
                eprintln!("right key will be reverse sorted in left join operation.")
            }

            let sort_idx = s_right.arg_sort(SortOptions {
                descending: false,
                nulls_last: false,
                multithreaded: true,
                maintain_order: false,
            });
            let s_right = unsafe { s_right.take_unchecked(&sort_idx) };

            let ids = par_sorted_merge_left(s_left, &s_right);
            let reverse_idx_map = create_reverse_map_from_arg_sort(sort_idx);
            let (left, mut right) = ids;

            POOL.install(|| {
                right.par_iter_mut().for_each(|opt_idx| {
                    if !opt_idx.is_null_idx() {
                        *opt_idx =
                            unsafe { *reverse_idx_map.get_unchecked(opt_idx.idx() as usize) }
                                .into();
                    }
                });
            });

            Ok(to_left_join_ids(left, right))
        },
        // don't reverse sort a left join key yet. Have to figure out how to set sorted flag
        _ => s_left.hash_join_left(s_right, validate, join_nulls),
    }
}
