#[cfg(feature = "performant")]
use polars_arrow::kernels::sorted_join;
#[cfg(feature = "performant")]
use polars_utils::flatten;

use super::*;
#[cfg(feature = "performant")]
use crate::utils::_split_offsets;

pub(super) fn use_sort_merge(s_left: &Series, s_right: &Series) -> bool {
    // only use for numeric data for now
    use IsSorted::*;
    let out = match (s_left.is_sorted(), s_right.is_sorted()) {
        (Ascending, Ascending) => {
            s_left.null_count() == 0
                && s_right.null_count() == 0
                && s_left.dtype().to_physical().is_numeric()
        }
        _ => false,
    };
    if out && std::env::var("POLARS_VERBOSE").is_ok() {
        eprintln!("keys are sorted: use sorted merge join")
    }
    out
}

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
        _ => unreachable!(),
    }
}
