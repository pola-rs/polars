use polars_utils::float::IsFloat;
use polars_utils::ord::compare_fn_nan_max;

/// used a lot, ensure there is a single impl
pub fn sort_slice_ascending<T: IsFloat + PartialOrd>(v: &mut [T]) {
    v.sort_unstable_by(|a, b| compare_fn_nan_max(a, b))
}
pub fn sort_slice_descending<T: IsFloat + PartialOrd>(v: &mut [T]) {
    v.sort_unstable_by(|a, b| compare_fn_nan_max(b, a))
}
