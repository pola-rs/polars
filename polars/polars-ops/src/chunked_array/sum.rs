use arrow::types::NativeType;
use polars_core::export::num::{NumCast, ToPrimitive};
use polars_utils::unwrap::UnwrapUncheckedRelease;

pub(super) fn sum_slice<T, S>(values: &[T]) -> S
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Sum,
{
    values
        .iter()
        .copied()
        .map(|t| unsafe {
            let s: S = NumCast::from(t).unwrap_unchecked_release();
            s
        })
        .sum()
}
