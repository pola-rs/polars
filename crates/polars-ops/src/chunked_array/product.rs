use arrow::types::NativeType;
use num_traits::{NumCast, ToPrimitive};

pub(super) fn product_slice<T, S>(values: &[T]) -> S
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Product,
{
    values
        .iter()
        .copied()
        .map(|t| unsafe {
            let s: S = NumCast::from(t).unwrap_unchecked();
            s
        })
        .product()
}
