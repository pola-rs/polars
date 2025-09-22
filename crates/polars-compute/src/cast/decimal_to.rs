use arrow::array::*;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use num_traits::{AsPrimitive, Float, NumCast};
use polars_error::PolarsResult;

use crate::decimal::{dec128_fits, dec128_rescale, dec128_to_f64, dec128_to_i128};

/// Returns a [`PrimitiveArray<i128>`] with the cast values. Values become null on overflow.
pub fn decimal_to_decimal(
    from: &PrimitiveArray<i128>,
    to_precision: usize,
    to_scale: usize,
) -> PrimitiveArray<i128> {
    let (from_precision, from_scale) =
        if let ArrowDataType::Decimal(p, s) = from.dtype().to_logical_type() {
            (*p, *s)
        } else {
            panic!("internal error: i128 is always a decimal")
        };

    if to_scale == from_scale {
        if to_precision >= from_precision {
            // Increasing precision is always allowed.
            return from
                .clone()
                .to(ArrowDataType::Decimal(to_precision, to_scale));
        } else {
            let it = from
                .iter()
                .map(|opt_x| opt_x.copied().filter(|x| dec128_fits(*x, to_precision)));
            return PrimitiveArray::<i128>::from_trusted_len_iter(it)
                .to(ArrowDataType::Decimal(to_precision, to_scale));
        }
    }

    let it = from
        .iter()
        .map(|opt_x| dec128_rescale(*(opt_x?), from_scale, to_precision, to_scale));
    PrimitiveArray::<i128>::from_trusted_len_iter(it)
        .to(ArrowDataType::Decimal(to_precision, to_scale))
}

pub(super) fn decimal_to_decimal_dyn(
    from: &dyn Array,
    to_precision: usize,
    to_scale: usize,
) -> PolarsResult<Box<dyn Array>> {
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(decimal_to_decimal(from, to_precision, to_scale)))
}

/// Returns a [`PrimitiveArray<i128>`] with the cast values. Values are `None` on overflow
pub fn decimal_to_float<T>(from: &PrimitiveArray<i128>) -> PrimitiveArray<T>
where
    T: NativeType + Float,
    f64: AsPrimitive<T>,
{
    let (_, from_scale) = if let ArrowDataType::Decimal(p, s) = from.dtype().to_logical_type() {
        (*p, *s)
    } else {
        unreachable!()
    };

    let it = from
        .iter()
        .map(|opt_x| Some(dec128_to_f64(*(opt_x?), from_scale).as_()));
    PrimitiveArray::<T>::from_trusted_len_iter(it)
}

pub(super) fn decimal_to_float_dyn<T>(from: &dyn Array) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType + Float,
    f64: AsPrimitive<T>,
{
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(decimal_to_float::<T>(from)))
}

/// Returns a [`PrimitiveArray<i128>`] with the cast values. Values are `None` on overflow
pub fn decimal_to_integer<T>(from: &PrimitiveArray<i128>) -> PrimitiveArray<T>
where
    T: NativeType + NumCast,
{
    let (_, from_scale) = if let ArrowDataType::Decimal(p, s) = from.dtype().to_logical_type() {
        (*p, *s)
    } else {
        unreachable!()
    };

    let it = from
        .iter()
        .map(|opt_x| T::from(dec128_to_i128(*(opt_x?), from_scale)));
    PrimitiveArray::<T>::from_trusted_len_iter(it)
}

pub(super) fn decimal_to_integer_dyn<T>(from: &dyn Array) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType + NumCast,
{
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(decimal_to_integer::<T>(from)))
}

/// Returns a [`Utf8Array`] where every element is the utf8 representation of the decimal.
#[cfg(feature = "dtype-decimal")]
pub(super) fn decimal_to_utf8view(from: &PrimitiveArray<i128>) -> Utf8ViewArray {
    use crate::decimal::DecimalFmtBuffer;

    let (_, from_scale) = if let ArrowDataType::Decimal(p, s) = from.dtype().to_logical_type() {
        (*p, *s)
    } else {
        unreachable!()
    };

    let mut mutable = MutableBinaryViewArray::with_capacity(from.len());
    let mut fmt_buf = DecimalFmtBuffer::new();
    for &x in from.values().iter() {
        mutable.push_value_ignore_validity(fmt_buf.format_dec128(x, from_scale, false))
    }

    mutable.freeze().with_validity(from.validity().cloned())
}

#[cfg(feature = "dtype-decimal")]
pub(super) fn decimal_to_utf8view_dyn(from: &dyn Array) -> Utf8ViewArray {
    let from = from.as_any().downcast_ref().unwrap();
    decimal_to_utf8view(from)
}
