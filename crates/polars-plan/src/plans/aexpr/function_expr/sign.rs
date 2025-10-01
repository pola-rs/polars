use num_traits::{One, Zero};
use polars_core::with_match_physical_numeric_polars_type;

use super::*;

pub(super) fn sign(s: &Column) -> PolarsResult<Column> {
    let s = s.as_materialized_series();
    let dtype = s.dtype();
    match dtype {
        _ if dtype.is_primitive_numeric() => with_match_physical_numeric_polars_type!(dtype, |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref();
            Ok(sign_impl(ca))
        }),
        DataType::Decimal(_, scale) => {
            let ca = s.decimal()?;

            let p_one = 10i128.pow(*scale as u32);
            let n_one = -p_one;

            let out = ca
                .physical()
                .apply_values(|x| {
                    if x < 0 {
                        n_one
                    } else if x > 0 {
                        p_one
                    } else {
                        0
                    }
                })
                .into_column();
            unsafe { out.from_physical_unchecked(dtype) }
        },
        _ => polars_bail!(opq = sign, dtype),
    }
}

fn sign_impl<T>(ca: &ChunkedArray<T>) -> Column
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoColumn,
{
    ca.apply_values(|x| {
        if x < T::Native::zero() {
            T::Native::zero() - T::Native::one()
        } else if x > T::Native::zero() {
            T::Native::one()
        } else {
            // Returning x here ensures we return NaN for NaN input, and
            // maintain the sign for signed zeroes (although we don't really
            // care about the latter).
            x
        }
    })
    .into_column()
}
