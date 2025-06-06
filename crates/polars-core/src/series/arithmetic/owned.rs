use super::*;
#[cfg(feature = "performant")]
use crate::utils::align_chunks_binary_owned_series;

#[cfg(feature = "performant")]
pub fn coerce_lhs_rhs_owned(lhs: Series, rhs: Series) -> PolarsResult<(Series, Series)> {
    let dtype = try_get_supertype(lhs.dtype(), rhs.dtype())?;
    let left = if lhs.dtype() == &dtype {
        lhs
    } else {
        lhs.cast(&dtype)?
    };
    let right = if rhs.dtype() == &dtype {
        rhs
    } else {
        rhs.cast(&dtype)?
    };
    Ok((left, right))
}

fn is_eligible(lhs: &DataType, rhs: &DataType) -> bool {
    !lhs.is_logical()
        && lhs.to_physical().is_primitive_numeric()
        && rhs.to_physical().is_primitive_numeric()
}

#[cfg(feature = "performant")]
fn apply_operation_mut<T, F>(mut lhs: Series, mut rhs: Series, op: F) -> Series
where
    T: PolarsNumericType,
    F: Fn(ChunkedArray<T>, ChunkedArray<T>) -> ChunkedArray<T> + Copy,
{
    let lhs_ca: &mut ChunkedArray<T> = lhs._get_inner_mut().as_mut();
    let rhs_ca: &mut ChunkedArray<T> = rhs._get_inner_mut().as_mut();

    let lhs = std::mem::take(lhs_ca);
    let rhs = std::mem::take(rhs_ca);

    op(lhs, rhs).into_series()
}

macro_rules! impl_operation {
    ($operation:ident, $method:ident, $function:expr) => {
        impl $operation for Series {
            type Output = PolarsResult<Series>;

            fn $method(self, rhs: Self) -> Self::Output {
                #[cfg(feature = "performant")]
                {
                    // only physical numeric values take the mutable path
                    if is_eligible(self.dtype(), rhs.dtype()) {
                        let (lhs, rhs) = coerce_lhs_rhs_owned(self, rhs).unwrap();
                        let (lhs, rhs) = align_chunks_binary_owned_series(lhs, rhs);
                        use DataType::*;
                        Ok(match lhs.dtype() {
                            #[cfg(feature = "dtype-i8")]
                            Int8 => apply_operation_mut::<Int8Type, _>(lhs, rhs, $function),
                            #[cfg(feature = "dtype-i16")]
                            Int16 => apply_operation_mut::<Int16Type, _>(lhs, rhs, $function),
                            Int32 => apply_operation_mut::<Int32Type, _>(lhs, rhs, $function),
                            Int64 => apply_operation_mut::<Int64Type, _>(lhs, rhs, $function),
                            #[cfg(feature = "dtype-i128")]
                            Int128 => apply_operation_mut::<Int128Type, _>(lhs, rhs, $function),
                            #[cfg(feature = "dtype-u8")]
                            UInt8 => apply_operation_mut::<UInt8Type, _>(lhs, rhs, $function),
                            #[cfg(feature = "dtype-u16")]
                            UInt16 => apply_operation_mut::<UInt16Type, _>(lhs, rhs, $function),
                            UInt32 => apply_operation_mut::<UInt32Type, _>(lhs, rhs, $function),
                            UInt64 => apply_operation_mut::<UInt64Type, _>(lhs, rhs, $function),
                            Float32 => apply_operation_mut::<Float32Type, _>(lhs, rhs, $function),
                            Float64 => apply_operation_mut::<Float64Type, _>(lhs, rhs, $function),
                            _ => unreachable!(),
                        })
                    } else {
                        (&self).$method(&rhs)
                    }
                }
                #[cfg(not(feature = "performant"))]
                {
                    (&self).$method(&rhs)
                }
            }
        }
    };
}

impl_operation!(Add, add, |a, b| a.add(b));
impl_operation!(Sub, sub, |a, b| a.sub(b));
impl_operation!(Mul, mul, |a, b| a.mul(b));
impl_operation!(Div, div, |a, b| a.div(b));

impl Series {
    pub fn try_add_owned(self, other: Self) -> PolarsResult<Self> {
        if is_eligible(self.dtype(), other.dtype()) {
            self + other
        } else {
            std::ops::Add::add(&self, &other)
        }
    }

    pub fn try_sub_owned(self, other: Self) -> PolarsResult<Self> {
        if is_eligible(self.dtype(), other.dtype()) {
            self - other
        } else {
            std::ops::Sub::sub(&self, &other)
        }
    }

    pub fn try_mul_owned(self, other: Self) -> PolarsResult<Self> {
        if is_eligible(self.dtype(), other.dtype()) {
            self * other
        } else {
            std::ops::Mul::mul(&self, &other)
        }
    }
}
