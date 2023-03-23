use std::cmp::Ordering;

use either::Either;
use polars_arrow::kernels::rolling::compare_fn_nan_max;
use polars_core::downcast_as_macro_arg_physical;
use polars_core::prelude::sort::{sort_slice_ascending, sort_slice_descending};
use polars_core::prelude::*;

#[repr(transparent)]
struct Compare<T>(T);

impl<T: PartialOrd + IsFloat> PartialEq for Compare<T> {
    fn eq(&self, other: &Self) -> bool {
        matches!(self.cmp(other), Ordering::Equal)
    }
}

impl<T: PartialOrd + IsFloat> PartialOrd for Compare<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(compare_fn_nan_max(&self.0, &other.0))
    }
}

impl<T: PartialOrd + IsFloat> Eq for Compare<T> {}

impl<T: PartialOrd + IsFloat> Ord for Compare<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Safety:
        // we always return Some
        unsafe { self.partial_cmp(other).unwrap_unchecked() }
    }
}

fn arg_partition<T: IsFloat + PartialOrd>(v: &mut [T], k: usize, descending: bool) -> &[T] {
    let (lower, _el, upper) = v.select_nth_unstable_by(k, |a, b| compare_fn_nan_max(a, b));
    if descending {
        sort_slice_ascending(lower);
        lower
    } else {
        sort_slice_descending(upper);
        upper
    }
}

fn top_k_impl<T>(ca: &ChunkedArray<T>, k: usize, descending: bool) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkSort<T>,
{
    if k >= ca.len() {
        return ca.sort(!descending);
    }

    // descending is opposite from sort as top-k returns largest
    let k = if descending {
        std::cmp::min(k, ca.len())
    } else {
        ca.len().saturating_sub(k + 1)
    };

    match ca.to_vec_null_aware() {
        Either::Left(mut v) => {
            let values = arg_partition(&mut v, k, descending);
            ChunkedArray::from_slice(ca.name(), values)
        }
        Either::Right(mut v) => {
            let values = arg_partition(&mut v, k, descending);
            let mut out = ChunkedArray::from_iter(values.iter().copied());
            out.rename(ca.name());
            out
        }
    }
}

pub fn top_k(s: &Series, k: usize, descending: bool) -> PolarsResult<Series> {
    if s.is_empty() {
        return Ok(s.clone());
    }
    let dtype = s.dtype();

    let s = s.to_physical_repr();

    macro_rules! dispatch {
        ($ca:expr) => {{
            top_k_impl($ca, k, descending).into_series()
        }};
    }

    downcast_as_macro_arg_physical!(&s, dispatch).cast(dtype)
}
