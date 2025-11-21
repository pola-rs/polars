use arrow::types::NativeType;
use num_traits::{NumCast, ToPrimitive};

pub(super) fn sum_slice<T, S>(values: &[T]) -> S
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Sum,
{
    values
        .iter()
        .copied()
        .map(|t| unsafe {
            let s: S = NumCast::from(t).unwrap_unchecked();
            s
        })
        .sum()
}
