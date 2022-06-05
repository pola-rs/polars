use super::*;
use crate::utils::split_offsets;
use polars_arrow::kernels::sorted_join;
use polars_utils::flatten;

pub(super) fn use_sort_merge(s_left: &Series, s_right: &Series) -> bool {
    use IsSorted::*;
    let out = match (s_left.is_sorted(), s_right.is_sorted()) {
        (Ascending, Ascending) => s_left.null_count() == 0 && s_right.null_count() == 0,
        _ => false,
    };
    if out && std::env::var("POLARS_VERBOSE").is_ok() {
        eprintln!("keys are sorted: use sorted merge join")
    }
    out
}

fn par_sorted_merge_left_impl<T>(
    s_left: &ChunkedArray<T>,
    s_right: &ChunkedArray<T>,
) -> (Vec<IdxSize>, Vec<Option<IdxSize>>)
where
    T: PolarsNumericType,
{
    let offsets = split_offsets(s_left.len(), POOL.current_num_threads());
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

pub(super) fn par_sorted_merge_left(
    s_left: &Series,
    s_right: &Series,
) -> (Vec<IdxSize>, Vec<Option<IdxSize>>) {
    debug_assert_eq!(s_left.dtype(), s_right.dtype());
    if s_left.bit_repr_is_large() {
        let left = s_left.bit_repr_large();
        let right = s_right.bit_repr_large();

        par_sorted_merge_left_impl(&left, &right)
    } else {
        let left = s_left.bit_repr_small();
        let right = s_right.bit_repr_small();

        par_sorted_merge_left_impl(&left, &right)
    }
}
fn par_sorted_merge_inner_impl<T>(
    s_left: &ChunkedArray<T>,
    s_right: &ChunkedArray<T>,
) -> (Vec<IdxSize>, Vec<IdxSize>)
where
    T: PolarsNumericType,
{
    let offsets = split_offsets(s_left.len(), POOL.current_num_threads());
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

pub(super) fn par_sorted_merge_inner(
    s_left: &Series,
    s_right: &Series,
) -> (Vec<IdxSize>, Vec<IdxSize>) {
    debug_assert_eq!(s_left.dtype(), s_right.dtype());
    if s_left.bit_repr_is_large() {
        let left = s_left.bit_repr_large();
        let right = s_right.bit_repr_large();

        par_sorted_merge_inner_impl(&left, &right)
    } else {
        let left = s_left.bit_repr_small();
        let right = s_right.bit_repr_small();

        par_sorted_merge_inner_impl(&left, &right)
    }
}
