use num_traits::{AsPrimitive, Float, NumCast};
use polars_error::PolarsResult;

use crate::array::*;
use crate::datatypes::DataType;
use crate::types::NativeType;

#[inline]
fn decimal_to_decimal_impl<F: Fn(i128) -> Option<i128>>(
    from: &PrimitiveArray<i128>,
    op: F,
    to_precision: usize,
    to_scale: usize,
) -> PrimitiveArray<i128> {
    let min_for_precision = 9_i128
        .saturating_pow(1 + to_precision as u32)
        .saturating_neg();
    let max_for_precision = 9_i128.saturating_pow(1 + to_precision as u32);

    let values = from.iter().map(|x| {
        x.and_then(|x| {
            op(*x).and_then(|x| {
                if x > max_for_precision || x < min_for_precision {
                    None
                } else {
                    Some(x)
                }
            })
        })
    });
    PrimitiveArray::<i128>::from_trusted_len_iter(values)
        .to(DataType::Decimal(to_precision, to_scale))
}

/// Returns a [`PrimitiveArray<i128>`] with the casted values. Values are `None` on overflow
pub fn decimal_to_decimal(
    from: &PrimitiveArray<i128>,
    to_precision: usize,
    to_scale: usize,
) -> PrimitiveArray<i128> {
    let (from_precision, from_scale) =
        if let DataType::Decimal(p, s) = from.data_type().to_logical_type() {
            (*p, *s)
        } else {
            panic!("internal error: i128 is always a decimal")
        };

    if to_scale == from_scale && to_precision >= from_precision {
        // fast path
        return from.clone().to(DataType::Decimal(to_precision, to_scale));
    }
    // todo: other fast paths include increasing scale and precision by so that
    // a number will never overflow (validity is preserved)

    if from_scale > to_scale {
        let factor = 10_i128.pow((from_scale - to_scale) as u32);
        decimal_to_decimal_impl(
            from,
            |x: i128| x.checked_div(factor),
            to_precision,
            to_scale,
        )
    } else {
        let factor = 10_i128.pow((to_scale - from_scale) as u32);
        decimal_to_decimal_impl(
            from,
            |x: i128| x.checked_mul(factor),
            to_precision,
            to_scale,
        )
    }
}

pub(super) fn decimal_to_decimal_dyn(
    from: &dyn Array,
    to_precision: usize,
    to_scale: usize,
) -> PolarsResult<Box<dyn Array>> {
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(decimal_to_decimal(from, to_precision, to_scale)))
}

/// Returns a [`PrimitiveArray<i128>`] with the casted values. Values are `None` on overflow
pub fn decimal_to_float<T>(from: &PrimitiveArray<i128>) -> PrimitiveArray<T>
where
    T: NativeType + Float,
    f64: AsPrimitive<T>,
{
    let (_, from_scale) = if let DataType::Decimal(p, s) = from.data_type().to_logical_type() {
        (*p, *s)
    } else {
        panic!("internal error: i128 is always a decimal")
    };

    let div = 10_f64.powi(from_scale as i32);
    let values = from
        .values()
        .iter()
        .map(|x| (*x as f64 / div).as_())
        .collect();

    PrimitiveArray::<T>::new(T::PRIMITIVE.into(), values, from.validity().cloned())
}

pub(super) fn decimal_to_float_dyn<T>(from: &dyn Array) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType + Float,
    f64: AsPrimitive<T>,
{
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(decimal_to_float::<T>(from)))
}

/// Returns a [`PrimitiveArray<i128>`] with the casted values. Values are `None` on overflow
pub fn decimal_to_integer<T>(from: &PrimitiveArray<i128>) -> PrimitiveArray<T>
where
    T: NativeType + NumCast,
{
    let (_, from_scale) = if let DataType::Decimal(p, s) = from.data_type().to_logical_type() {
        (*p, *s)
    } else {
        panic!("internal error: i128 is always a decimal")
    };

    let factor = 10_i128.pow(from_scale as u32);
    let values = from.iter().map(|x| x.and_then(|x| T::from(*x / factor)));

    PrimitiveArray::from_trusted_len_iter(values)
}

pub(super) fn decimal_to_integer_dyn<T>(from: &dyn Array) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType + NumCast,
{
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(decimal_to_integer::<T>(from)))
}
