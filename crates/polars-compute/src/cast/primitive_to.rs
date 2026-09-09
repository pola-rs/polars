//! Casting the primitive arrays of `polars-array`, which every fixed-width type is held by.

use arrow::types::NativeType;
use num_traits::AsPrimitive;
#[cfg(feature = "dtype-decimal")]
use num_traits::{Float, NumCast, ToPrimitive};
use polars_array::{
    PlBinaryViewArray, PlBinaryViewArrayBuilder, PlBitmap, PlBooleanArray, PlPrimitiveArray,
    PlUtf8ViewArray, StaticArrayBuilder,
};
use polars_utils::float16::pf16;

use super::{map_values, map_values_fallible, mask_where};
use crate::comparisons::PlTotalEqKernel;
#[cfg(feature = "dtype-decimal")]
use crate::decimal::{
    dec128_fits, dec128_rescale, dec128_to_f64, dec128_to_i128, f64_to_dec128, i128_to_dec128,
};

/// The text a number is written as, which is how a cast to a string writes it.
pub trait SerPrimitive {
    fn write(f: &mut Vec<u8>, val: Self) -> usize
    where
        Self: Sized;
}

macro_rules! impl_ser_primitive {
    ($ptype:ident) => {
        impl SerPrimitive for $ptype {
            fn write(f: &mut Vec<u8>, val: Self) -> usize
            where
                Self: Sized,
            {
                let mut buffer = itoa::Buffer::new();
                let value = buffer.format(val);
                f.extend_from_slice(value.as_bytes());
                value.len()
            }
        }
    };
}

impl_ser_primitive!(i8);
impl_ser_primitive!(i16);
impl_ser_primitive!(i32);
impl_ser_primitive!(i64);
impl_ser_primitive!(i128);
impl_ser_primitive!(u8);
impl_ser_primitive!(u16);
impl_ser_primitive!(u32);
impl_ser_primitive!(u64);
impl_ser_primitive!(u128);

impl SerPrimitive for pf16 {
    fn write(f: &mut Vec<u8>, val: Self) -> usize
    where
        Self: Sized,
    {
        f32::write(f, AsPrimitive::<f32>::as_(val))
    }
}

impl SerPrimitive for f32 {
    fn write(f: &mut Vec<u8>, val: Self) -> usize
    where
        Self: Sized,
    {
        let mut buffer = zmij::Buffer::new();
        let value = buffer.format(val);
        f.extend_from_slice(value.as_bytes());
        value.len()
    }
}

impl SerPrimitive for f64 {
    fn write(f: &mut Vec<u8>, val: Self) -> usize
    where
        Self: Sized,
    {
        let mut buffer = zmij::Buffer::new();
        let value = buffer.format(val);
        f.extend_from_slice(value.as_bytes());
        value.len()
    }
}

/// Casts the values of `from` to `O`, leaving a null where a value does not fit.
pub fn numeric_to_numeric<I, O>(from: &PlPrimitiveArray<I>, wrapped: bool) -> PlPrimitiveArray<O>
where
    I: NativeType + num_traits::NumCast + num_traits::AsPrimitive<O>,
    O: NativeType + num_traits::NumCast,
{
    // A wrapping cast answers for every value, so the mask is the one the array came with.
    if wrapped {
        return map_values(from, num_traits::AsPrimitive::<O>::as_);
    }

    numeric_to_numeric_checked(from)
}

/// Casts the values of `from` to `O`, leaving a null where a value does not fit.
pub fn numeric_to_numeric_checked<I, O>(from: &PlPrimitiveArray<I>) -> PlPrimitiveArray<O>
where
    I: NativeType + num_traits::NumCast,
    O: NativeType + num_traits::NumCast,
{
    map_values_fallible(from, num_traits::cast::cast::<I, O>)
}

/// Casts every set bit to one and every unset bit to zero, which is how a boolean reads as a number.
pub fn boolean_to_primitive<T>(from: &PlBooleanArray) -> PlPrimitiveArray<T>
where
    T: NativeType + num_traits::One,
{
    let value_of = |set: bool| if set { T::one() } else { T::default() };
    let values = match from.scalar_value_ignore_validity() {
        Some(value) => PlPrimitiveArray::new_scalar(value_of(value), from.len()),
        None => {
            let out: Vec<T> = from.flat_values().unwrap().iter().map(value_of).collect();
            PlPrimitiveArray::from_vec(out)
        },
    };
    values.with_validity(from.validity().map(PlBitmap::from))
}

/// Reads every value other than zero as `true`, which is how a number reads as a boolean.
pub fn primitive_to_boolean<T>(from: &PlPrimitiveArray<T>) -> PlBooleanArray
where
    T: NativeType,
    PlPrimitiveArray<T>: PlTotalEqKernel<Scalar = T>,
{
    // The comparison kernel answers over the representation the values are in, so a chunk that
    // repeats one value is compared once and the answer repeats in turn.
    let values = from.tot_ne_kernel_broadcast(&T::default());
    PlBooleanArray::from_pl_bitmap(values).with_validity(from.validity().map(PlBitmap::from))
}

/// Writes every value as the text it is written as, which is how a number reads as a string.
pub fn primitive_to_binview<T: NativeType + SerPrimitive>(
    from: &PlPrimitiveArray<T>,
) -> PlBinaryViewArray {
    let mut scratch = vec![];
    let write = |value: T, scratch: &mut Vec<u8>| {
        scratch.clear();
        T::write(scratch, value);
    };

    // The one value every element of a scalar chunk reads is written once, and the views repeat it.
    if let Some(value) = from.scalar_value_ignore_validity() {
        write(value, &mut scratch);
        return PlBinaryViewArray::new_scalar(&scratch, from.len())
            .with_validity(from.validity().map(PlBitmap::from));
    }

    let values = from.flat_values().unwrap();
    let mut builder = PlBinaryViewArrayBuilder::with_capacity(values.len());
    for &value in values.iter() {
        write(value, &mut scratch);
        builder.push_value(&scratch);
    }
    builder
        .freeze()
        .with_validity(from.validity().map(PlBitmap::from))
}

/// [`primitive_to_binview`], whose text is UTF-8 because the text of a number is.
pub fn primitive_to_utf8view<T: NativeType + SerPrimitive>(
    from: &PlPrimitiveArray<T>,
) -> PlUtf8ViewArray {
    // SAFETY: a number is written as ASCII, which is valid UTF-8.
    unsafe { PlUtf8ViewArray::from_binview_unchecked(primitive_to_binview(from)) }
}

/// Scales the values of `from` by `to_multiple / from_multiple`, which is how a time unit changes.
pub fn rescale_time(
    from: &PlPrimitiveArray<i64>,
    from_multiple: i64,
    to_multiple: i64,
) -> PlPrimitiveArray<i64> {
    if from_multiple == to_multiple {
        return from.clone();
    }

    if from_multiple >= to_multiple {
        let factor = from_multiple / to_multiple;
        map_values(from, move |value| value / factor)
    } else {
        let factor = to_multiple / from_multiple;
        map_values_fallible(from, move |value| value.checked_mul(factor))
    }
}

/// Reads the elapsed time a value holds as the day it falls in.
pub fn timestamp_to_date(
    from: &PlPrimitiveArray<i64>,
    timestamps_per_day: i64,
) -> PlPrimitiveArray<i32> {
    map_values(from, move |value| (value / timestamps_per_day) as i32)
}

/// Unsets the mask wherever a value names no time of day, which is how an `i64` reads as a time.
pub fn int64_to_time(from: &PlPrimitiveArray<i64>) -> PlPrimitiveArray<i64> {
    const NANOSECONDS_IN_DAY: i64 = 86_400_000_000_000;
    mask_where(from, |value| (0..NANOSECONDS_IN_DAY).contains(&value))
}

/// Reads an integer as a decimal of `to_scale`, leaving a null where a value does not fit.
#[cfg(feature = "dtype-decimal")]
pub fn integer_to_decimal<T: NativeType + ToPrimitive>(
    from: &PlPrimitiveArray<T>,
    to_precision: usize,
    to_scale: usize,
) -> PlPrimitiveArray<i128> {
    map_values_fallible(from, move |value| {
        i128_to_dec128(value.to_i128()?, to_precision, to_scale)
    })
}

/// Reads a float as a decimal of `to_scale`, leaving a null where a value does not fit.
#[cfg(feature = "dtype-decimal")]
pub fn float_to_decimal<T: NativeType + Float + AsPrimitive<f64>>(
    from: &PlPrimitiveArray<T>,
    to_precision: usize,
    to_scale: usize,
) -> PlPrimitiveArray<i128> {
    map_values_fallible(from, move |value| {
        f64_to_dec128(value.as_(), to_precision, to_scale)
    })
}

/// Drops the scale of a decimal, which is how it reads as an integer.
#[cfg(feature = "dtype-decimal")]
pub fn decimal_to_integer<T>(
    from: &PlPrimitiveArray<i128>,
    from_scale: usize,
) -> PlPrimitiveArray<T>
where
    T: NativeType + NumCast,
{
    map_values_fallible(from, move |value| {
        T::from(dec128_to_i128(value, from_scale))
    })
}

/// Reads a decimal as the closest float to it.
#[cfg(feature = "dtype-decimal")]
pub fn decimal_to_float<T>(from: &PlPrimitiveArray<i128>, from_scale: usize) -> PlPrimitiveArray<T>
where
    T: NativeType + Float,
    f64: AsPrimitive<T>,
{
    map_values(from, move |value| dec128_to_f64(value, from_scale).as_())
}

/// Rescales a decimal, leaving a null where a value no longer fits the precision.
#[cfg(feature = "dtype-decimal")]
pub fn decimal_to_decimal(
    from: &PlPrimitiveArray<i128>,
    from_precision: usize,
    from_scale: usize,
    to_precision: usize,
    to_scale: usize,
) -> PlPrimitiveArray<i128> {
    if from_scale == to_scale {
        // Widening the precision keeps every value, so the array itself is the answer.
        if to_precision >= from_precision {
            return from.clone();
        }
        return mask_where(from, move |value| dec128_fits(value, to_precision));
    }

    map_values_fallible(from, move |value| {
        dec128_rescale(value, from_scale, to_precision, to_scale)
    })
}

/// Writes a decimal as the text it is written as, which is how it reads as a string.
#[cfg(feature = "dtype-decimal")]
pub fn decimal_to_utf8view(from: &PlPrimitiveArray<i128>, from_scale: usize) -> PlUtf8ViewArray {
    use crate::decimal::DecimalFmtBuffer;

    let mut fmt_buf = DecimalFmtBuffer::new();

    // The one value every element of a scalar chunk reads is written once, and the views repeat it.
    let binview = if let Some(value) = from.scalar_value_ignore_validity() {
        PlBinaryViewArray::new_scalar(
            fmt_buf
                .format_dec128(value, from_scale, false, false)
                .as_bytes(),
            from.len(),
        )
    } else {
        let values = from.flat_values().unwrap();
        let mut builder = PlBinaryViewArrayBuilder::with_capacity(values.len());
        for &value in values.iter() {
            builder.push_value(
                fmt_buf
                    .format_dec128(value, from_scale, false, false)
                    .as_bytes(),
            );
        }
        builder.freeze()
    };

    // SAFETY: a decimal is written as ASCII, which is valid UTF-8.
    unsafe {
        PlUtf8ViewArray::from_binview_unchecked(
            binview.with_validity(from.validity().map(PlBitmap::from)),
        )
    }
}
