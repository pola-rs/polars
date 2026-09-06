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

/// The sum of `count` copies of `value`, added up in the order a buffer holding them would be.
///
/// This is what a values buffer that repeats a single value sums to, without it being written out.
pub(super) fn sum_repeated<T, S>(value: T, count: usize) -> S
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Sum,
{
    (0..count)
        .map(|_| unsafe {
            // SAFETY: as `sum_slice`, this is the cast the element type is summed through.
            let s: S = NumCast::from(value).unwrap_unchecked();
            s
        })
        .sum()
}
