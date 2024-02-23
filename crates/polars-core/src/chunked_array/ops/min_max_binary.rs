use crate::prelude::*;
use crate::series::arithmetic::coerce_lhs_rhs;

fn min_binary<T>(left: &ChunkedArray<T>, right: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: PartialOrd,
{
    let op = |l: T::Native, r: T::Native| {
        if l < r {
            l
        } else {
            r
        }
    };
    arity::binary_elementwise_values(left, right, op)
}

fn max_binary<T>(left: &ChunkedArray<T>, right: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: PartialOrd,
{
    let op = |l: T::Native, r: T::Native| {
        if l > r {
            l
        } else {
            r
        }
    };
    arity::binary_elementwise_values(left, right, op)
}

pub(crate) fn min_max_binary_series(
    left: &Series,
    right: &Series,
    min: bool,
) -> PolarsResult<Series> {
    if left.dtype().to_physical().is_numeric()
        && left.null_count() == 0
        && right.null_count() == 0
        && left.len() == right.len()
    {
        let (lhs, rhs) = coerce_lhs_rhs(left, right)?;
        let logical = lhs.dtype();
        let lhs = lhs.to_physical_repr();
        let rhs = rhs.to_physical_repr();

        with_match_physical_numeric_polars_type!(lhs.dtype(), |$T| {
        let a: &ChunkedArray<$T> = lhs.as_ref().as_ref().as_ref();
        let b: &ChunkedArray<$T> = rhs.as_ref().as_ref().as_ref();

        if min {
            min_binary(a, b).into_series().cast(logical)
        } else {
            max_binary(a, b).into_series().cast(logical)
            }
        })
    } else {
        let mask = if min {
            left.lt(right)? & left.is_not_null() | right.is_null()
        } else {
            left.gt(right)? & left.is_not_null() | right.is_null()
        };
        left.zip_with(&mask, right)
    }
}
