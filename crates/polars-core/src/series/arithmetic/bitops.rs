use std::borrow::Cow;

use polars_error::PolarsResult;

use super::{polars_bail, BooleanChunked, ChunkedArray, DataType, IntoSeries, Series};

macro_rules! impl_bitop {
    ($(($trait:ident, $f:ident))+) => {
        $(
        impl std::ops::$trait for &Series {
            type Output = PolarsResult<Series>;
            #[inline(never)]
            fn $f(self, rhs: Self) -> Self::Output {
                use DataType as DT;
                match self.dtype() {
                    DT::Boolean => {
                        let lhs: &BooleanChunked = self.as_ref().as_ref().as_ref();
                        let rhs = lhs.unpack_series_matching_type(rhs)?;
                        Ok(lhs.$f(rhs).into_series())
                    },
                    dt if dt.is_integer() => {
                        let rhs = if rhs.len() == 1 {
                            Cow::Owned(rhs.cast(self.dtype())?)
                        } else {
                            Cow::Borrowed(rhs)
                        };

                        with_match_physical_integer_polars_type!(dt, |$T| {
                            let lhs: &ChunkedArray<$T> = self.as_ref().as_ref().as_ref();
                            let rhs = lhs.unpack_series_matching_type(&rhs)?;
                            Ok(lhs.$f(&rhs).into_series())
                        })
                    },
                    _ => polars_bail!(opq = $f, self.dtype()),
                }
            }
        }
        impl std::ops::$trait for Series {
            type Output = PolarsResult<Series>;
            #[inline(always)]
            fn $f(self, rhs: Self) -> Self::Output {
                <&Series as std::ops::$trait>::$f(&self, &rhs)
            }
        }
        impl std::ops::$trait<&Series> for Series {
            type Output = PolarsResult<Series>;
            #[inline(always)]
            fn $f(self, rhs: &Series) -> Self::Output {
                <&Series as std::ops::$trait>::$f(&self, rhs)
            }
        }
        impl std::ops::$trait<Series> for &Series {
            type Output = PolarsResult<Series>;
            #[inline(always)]
            fn $f(self, rhs: Series) -> Self::Output {
                <&Series as std::ops::$trait>::$f(self, &rhs)
            }
        }
        )+
    };
}

impl_bitop! {
    (BitAnd, bitand)
    (BitOr, bitor)
    (BitXor, bitxor)
}
