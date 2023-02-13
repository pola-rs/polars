#[cfg(feature = "performant")]
use polars_arrow::kernels::sorted_join;
#[cfg(feature = "performant")]
use polars_utils::flatten;

use super::*;
#[cfg(feature = "performant")]
use crate::utils::_split_offsets;

#[cfg(feature = "performant")]
fn par_sorted_merge_left_impl<T>(
    s_left: &ChunkedArray<T>,
    s_right: &ChunkedArray<T>,
) -> (Vec<IdxSize>, Vec<Option<IdxSize>>)
where
    T: PolarsNumericType,
{
    let offsets = _split_offsets(s_left.len(), POOL.current_num_threads());
    let s_left = s_left.rechunk();
    let s_right = s_right.rechunk();

    // we can unwrap because we should not have nulls
    let slice_left = s_left.cont_slice().unwrap();
    let slice_right = s_right.cont_slice().unwrap();

    let indexes = offsets
        .into_par_iter()
        .map(|(offset, len)| {
            let slice_left = &slice_left[offset..offset + len];
            sorted_join::left::join(slice_left, slice_right, offset as IdxSize)
        })
        .collect::<Vec<_>>();
    let lefts = indexes.iter().map(|t| &t.0).collect::<Vec<_>>();
    let rights = indexes.iter().map(|t| &t.1).collect::<Vec<_>>();

    (flatten(&lefts, None), flatten(&rights, None))
}

#[cfg(feature = "performant")]
pub(super) fn par_sorted_merge_left(
    s_left: &Series,
    s_right: &Series,
) -> (Vec<IdxSize>, Vec<Option<IdxSize>>) {
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
        }
        #[cfg(feature = "dtype-i16")]
        DataType::Int16 => {
            par_sorted_merge_left_impl(s_left.i16().unwrap(), s_right.i16().unwrap())
        }
        DataType::UInt32 => {
            par_sorted_merge_left_impl(s_left.u32().unwrap(), s_right.u32().unwrap())
        }
        DataType::Int32 => {
            par_sorted_merge_left_impl(s_left.i32().unwrap(), s_right.i32().unwrap())
        }
        DataType::UInt64 => {
            par_sorted_merge_left_impl(s_left.u64().unwrap(), s_right.u64().unwrap())
        }
        DataType::Int64 => {
            par_sorted_merge_left_impl(s_left.i64().unwrap(), s_right.i64().unwrap())
        }
        DataType::Float32 => {
            par_sorted_merge_left_impl(s_left.f32().unwrap(), s_right.f32().unwrap())
        }
        DataType::Float64 => {
            par_sorted_merge_left_impl(s_left.f64().unwrap(), s_right.f64().unwrap())
        }
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

    let indexes = offsets
        .into_par_iter()
        .map(|(offset, len)| {
            let slice_left = &slice_left[offset..offset + len];
            sorted_join::inner::join(slice_left, slice_right, offset as IdxSize)
        })
        .collect::<Vec<_>>();
    let lefts = indexes.iter().map(|t| &t.0).collect::<Vec<_>>();
    let rights = indexes.iter().map(|t| &t.1).collect::<Vec<_>>();

    (flatten(&lefts, None), flatten(&rights, None))
}

#[cfg(feature = "performant")]
pub(super) fn par_sorted_merge_inner(
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
        }
        #[cfg(feature = "dtype-i16")]
        DataType::Int16 => {
            par_sorted_merge_inner_impl(s_left.i16().unwrap(), s_right.i16().unwrap())
        }
        DataType::UInt32 => {
            par_sorted_merge_inner_impl(s_left.u32().unwrap(), s_right.u32().unwrap())
        }
        DataType::Int32 => {
            par_sorted_merge_inner_impl(s_left.i32().unwrap(), s_right.i32().unwrap())
        }
        DataType::UInt64 => {
            par_sorted_merge_inner_impl(s_left.u64().unwrap(), s_right.u64().unwrap())
        }
        DataType::Int64 => {
            par_sorted_merge_inner_impl(s_left.i64().unwrap(), s_right.i64().unwrap())
        }
        DataType::Float32 => {
            par_sorted_merge_inner_impl(s_left.f32().unwrap(), s_right.f32().unwrap())
        }
        DataType::Float64 => {
            par_sorted_merge_inner_impl(s_left.f64().unwrap(), s_right.f64().unwrap())
        }
        _ => unreachable!(),
    }
}

#[cfg(feature = "performant")]
fn to_left_join_ids(left_idx: Vec<IdxSize>, right_idx: Vec<Option<IdxSize>>) -> LeftJoinIds {
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
    let arr = arg_sort.chunks.pop().unwrap();
    primitive_to_vec::<IdxSize>(arr).unwrap()
}

#[cfg(not(feature = "performant"))]
pub fn _sort_or_hash_inner(
    s_left: &Series,
    s_right: &Series,
    _verbose: bool,
) -> ((Vec<IdxSize>, Vec<IdxSize>), bool) {
    s_left.hash_join_inner(s_right)
}

#[cfg(feature = "performant")]
pub fn _sort_or_hash_inner(
    s_left: &Series,
    s_right: &Series,
    verbose: bool,
) -> ((Vec<IdxSize>, Vec<IdxSize>), bool) {
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
    match (s_left.is_sorted_flag(), s_right.is_sorted_flag()) {
        (IsSorted::Ascending, IsSorted::Ascending) if is_numeric => {
            if verbose {
                eprintln!("inner join: keys are sorted: use sorted merge join");
            }
            (par_sorted_merge_inner(s_left, s_right), true)
        }
        (IsSorted::Ascending, _) if is_numeric && size_factor_rhs < size_factor_acceptable => {
            if verbose {
                eprintln!("right key will be reverse sorted in inner join operation.")
            }

            let sort_idx = s_right.arg_sort(SortOptions {
                descending: false,
                nulls_last: false,
                multithreaded: true,
            });
            let s_right = unsafe { s_right.take_unchecked(&sort_idx).unwrap() };
            let ids = par_sorted_merge_inner(s_left, &s_right);
            let reverse_idx_map = create_reverse_map_from_arg_sort(sort_idx);

            let (left, mut right) = ids;

            POOL.install(|| {
                right.par_iter_mut().for_each(|idx| {
                    *idx = unsafe { *reverse_idx_map.get_unchecked(*idx as usize) };
                });
            });

            ((left, right), true)
        }
        (_, IsSorted::Ascending) if is_numeric && size_factor_lhs < size_factor_acceptable => {
            if verbose {
                eprintln!("left key will be reverse sorted in inner join operation.")
            }

            let sort_idx = s_left.arg_sort(SortOptions {
                descending: false,
                nulls_last: false,
                multithreaded: true,
            });
            let s_left = unsafe { s_left.take_unchecked(&sort_idx).unwrap() };
            let ids = par_sorted_merge_inner(&s_left, s_right);
            let reverse_idx_map = create_reverse_map_from_arg_sort(sort_idx);

            let (mut left, right) = ids;

            POOL.install(|| {
                left.par_iter_mut().for_each(|idx| {
                    *idx = unsafe { *reverse_idx_map.get_unchecked(*idx as usize) };
                });
            });

            // set sorted to `false` as we reverse sorted the left key.
            ((left, right), false)
        }
        _ => s_left.hash_join_inner(s_right),
    }
}

#[cfg(not(feature = "performant"))]
pub(super) fn sort_or_hash_left(s_left: &Series, s_right: &Series, _verbose: bool) -> LeftJoinIds {
    s_left.hash_join_left(s_right)
}

#[cfg(feature = "performant")]
pub(super) fn sort_or_hash_left(s_left: &Series, s_right: &Series, verbose: bool) -> LeftJoinIds {
    let size_factor_rhs = s_right.len() as f32 / s_left.len() as f32;
    let size_factor_acceptable = std::env::var("POLARS_JOIN_SORT_FACTOR")
        .map(|s| s.parse::<f32>().unwrap())
        .unwrap_or(1.0);
    let is_numeric = s_left.dtype().to_physical().is_numeric();

    match (s_left.is_sorted_flag(), s_right.is_sorted_flag()) {
        (IsSorted::Ascending, IsSorted::Ascending) if is_numeric => {
            if verbose {
                eprintln!("left join: keys are sorted: use sorted merge join");
            }
            let (left_idx, right_idx) = par_sorted_merge_left(s_left, s_right);
            to_left_join_ids(left_idx, right_idx)
        }
        (IsSorted::Ascending, _) if is_numeric && size_factor_rhs < size_factor_acceptable => {
            if verbose {
                eprintln!("right key will be reverse sorted in left join operation.")
            }

            let sort_idx = s_right.arg_sort(SortOptions {
                descending: false,
                nulls_last: false,
                multithreaded: true,
            });
            let s_right = unsafe { s_right.take_unchecked(&sort_idx).unwrap() };

            let ids = par_sorted_merge_left(s_left, &s_right);
            let reverse_idx_map = create_reverse_map_from_arg_sort(sort_idx);
            let (left, mut right) = ids;

            POOL.install(|| {
                right.par_iter_mut().for_each(|opt_idx| {
                    *opt_idx =
                        opt_idx.map(|idx| unsafe { *reverse_idx_map.get_unchecked(idx as usize) });
                });
            });

            to_left_join_ids(left, right)
        }
        // don't reverse sort a left join key yet. Have to figure out how to set sorted flag
        _ => s_left.hash_join_left(s_right),
    }
}
