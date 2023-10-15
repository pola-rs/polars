use arrow::legacy::kernels::rolling::compare_fn_nan_max;

use crate::prelude::*;

/// used a lot, ensure there is a single impl
pub fn sort_slice_ascending<T: IsFloat + PartialOrd>(v: &mut [T]) {
    v.sort_unstable_by(|a, b| compare_fn_nan_max(a, b))
}
pub fn sort_slice_descending<T: IsFloat + PartialOrd>(v: &mut [T]) {
    v.sort_unstable_by(|a, b| compare_fn_nan_max(b, a))
}
