use std::cmp::Ordering;

use polars_arrow::kernels::rolling::compare_fn_nan_max;
use polars_arrow::prelude::*;
use polars_core::downcast_as_macro_arg_physical;
use polars_core::export::num::NumCast;
use polars_core::prelude::*;

fn search_sorted_ca<T>(
    ca: &ChunkedArray<T>,
    search_value: T::Native,
) -> std::result::Result<IdxSize, IdxSize>
where
    T: PolarsNumericType,
    T::Native: PartialOrd + IsFloat + NumCast,
{
    let taker = ca.take_rand();

    let mut size = ca.len() as IdxSize;
    let mut left = 0 as IdxSize;
    let mut right = size;
    while left < right {
        let mid = left + size / 2;

        // SAFETY: the call is made safe by the following invariants:
        // - `mid >= 0`
        // - `mid < size`: `mid` is limited by `[left; right)` bound.
        let cmp = match unsafe { taker.get_unchecked(mid as usize) } {
            None => Ordering::Less,
            Some(value) => compare_fn_nan_max(&value, &search_value),
        };

        // The reason why we use if/else control flow rather than match
        // is because match reorders comparison operations, which is perf sensitive.
        // This is x86 asm for u8: https://rust.godbolt.org/z/8Y8Pra.
        if cmp == Ordering::Less {
            left = mid + 1;
        } else if cmp == Ordering::Greater {
            right = mid;
        } else {
            return Ok(mid);
        }

        size = right - left;
    }
    Err(left)
}

pub fn search_sorted(s: &Series, search_value: &AnyValue) -> PolarsResult<IdxSize> {
    if s.dtype().is_logical() {
        let search_dtype: DataType = search_value.into();
        if &search_dtype != s.dtype() {
            return Err(PolarsError::ComputeError(
                format!(
                    "Cannot search a series of dtype: {} with a value of dtype: {}",
                    s.dtype(),
                    search_dtype
                )
                .into(),
            ));
        }
    }
    let s = s.to_physical_repr();

    macro_rules! dispatch {
        ($ca:expr) => {
            match search_sorted_ca($ca, search_value.extract().unwrap()) {
                Ok(idx) => Ok(idx),
                Err(idx) => Ok(idx),
            }
        };
    }

    downcast_as_macro_arg_physical!(s, dispatch)
}
