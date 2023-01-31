use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;

use polars_arrow::kernels::rolling::compare_fn_nan_max;
use polars_core::downcast_as_macro_arg_physical;
use polars_core::export::num::NumCast;
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

fn top_k_impl<T>(
    ca: &ChunkedArray<T>,
    k: usize,
    mult_order: T::Native,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
{
    // mult_order should be -1 / +1 to determine the order of the heap
    let k = std::cmp::min(k, ca.len());

    let mut heap = BinaryHeap::with_capacity(ca.len());

    for arr in ca.downcast_iter() {
        for v in arr {
            heap.push(v.map(|v| Compare(*v * mult_order)));
        }
    }
    let mut out: ChunkedArray<_> = (0..k)
        .map(|_| {
            heap.pop()
                .unwrap()
                .map(|compare_struct| compare_struct.0 * mult_order)
        })
        .collect();
    out.rename(ca.name());
    Ok(out)
}

pub fn top_k(s: &Series, k: usize, reverse: bool) -> PolarsResult<Series> {
    if s.is_empty() {
        return Ok(s.clone());
    }
    let dtype = s.dtype();

    let s = s.to_physical_repr();

    macro_rules! dispatch {
        ($ca:expr) => {{
            let mult_order = if reverse { -1 } else { 1 };
            top_k_impl($ca, k, NumCast::from(mult_order).unwrap()).map(|ca| ca.into_series())
        }};
    }

    downcast_as_macro_arg_physical!(&s, dispatch).and_then(|s| s.cast(dtype))
}
