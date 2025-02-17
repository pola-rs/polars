use std::hash::Hash;

use arrow::array::*;
use arrow::bitmap::Bitmap;
use arrow::compute::arity::unary;
use arrow::datatypes::{ArrowDataType, TimeUnit};
use arrow::offset::{Offset, Offsets};
use arrow::types::{f16, NativeType};
use num_traits::{AsPrimitive, Float, ToPrimitive};
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;

use super::temporal::*;
use super::CastOptionsImpl;

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
impl_ser_primitive!(u8);
impl_ser_primitive!(u16);
impl_ser_primitive!(u32);
impl_ser_primitive!(u64);

impl SerPrimitive for f32 {
    fn write(f: &mut Vec<u8>, val: Self) -> usize
    where
        Self: Sized,
    {
        let mut buffer = ryu::Buffer::new();
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
        let mut buffer = ryu::Buffer::new();
        let value = buffer.format(val);
        f.extend_from_slice(value.as_bytes());
        value.len()
    }
}

fn primitive_to_values_and_offsets<T: NativeType + SerPrimitive, O: Offset>(
    from: &PrimitiveArray<T>,
) -> (Vec<u8>, Offsets<O>) {
    let mut values: Vec<u8> = Vec::with_capacity(from.len());
    let mut offsets: Vec<O> = Vec::with_capacity(from.len() + 1);
    offsets.push(O::default());

    let mut offset: usize = 0;

    unsafe {
        for &x in from.values().iter() {
            let len = T::write(&mut values, x);

            offset += len;
            offsets.push(O::from_as_usize(offset));
        }
        values.set_len(offset);
        values.shrink_to_fit();
        // SAFETY: offsets _are_ monotonically increasing
        let offsets = Offsets::new_unchecked(offsets);

        (values, offsets)
    }
}

/// Returns a [`BooleanArray`] where every element is different from zero.
/// Validity is preserved.
pub fn primitive_to_boolean<T: NativeType>(
    from: &PrimitiveArray<T>,
    to_type: ArrowDataType,
) -> BooleanArray {
    let iter = from.values().iter().map(|v| *v != T::default());
    let values = Bitmap::from_trusted_len_iter(iter);

    BooleanArray::new(to_type, values, from.validity().cloned())
}

pub(super) fn primitive_to_boolean_dyn<T>(
    from: &dyn Array,
    to_type: ArrowDataType,
) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType,
{
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(primitive_to_boolean::<T>(from, to_type)))
}

/// Returns a [`Utf8Array`] where every element is the utf8 representation of the number.
pub(super) fn primitive_to_utf8<T: NativeType + SerPrimitive, O: Offset>(
    from: &PrimitiveArray<T>,
) -> Utf8Array<O> {
    let (values, offsets) = primitive_to_values_and_offsets(from);
    unsafe {
        Utf8Array::<O>::new_unchecked(
            Utf8Array::<O>::default_dtype(),
            offsets.into(),
            values.into(),
            from.validity().cloned(),
        )
    }
}

pub(super) fn primitive_to_utf8_dyn<T, O>(from: &dyn Array) -> PolarsResult<Box<dyn Array>>
where
    O: Offset,
    T: NativeType + SerPrimitive,
{
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(primitive_to_utf8::<T, O>(from)))
}

pub(super) fn primitive_to_primitive_dyn<I, O>(
    from: &dyn Array,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn Array>>
where
    I: NativeType + num_traits::NumCast + num_traits::AsPrimitive<O>,
    O: NativeType + num_traits::NumCast,
{
    let from = from.as_any().downcast_ref::<PrimitiveArray<I>>().unwrap();
    if options.wrapped {
        Ok(Box::new(primitive_as_primitive::<I, O>(from, to_type)))
    } else {
        Ok(Box::new(primitive_to_primitive::<I, O>(from, to_type)))
    }
}

/// Cast [`PrimitiveArray`] to a [`PrimitiveArray`] of another physical type via numeric conversion.
pub fn primitive_to_primitive<I, O>(
    from: &PrimitiveArray<I>,
    to_type: &ArrowDataType,
) -> PrimitiveArray<O>
where
    I: NativeType + num_traits::NumCast,
    O: NativeType + num_traits::NumCast,
{
    let iter = from
        .iter()
        .map(|v| v.and_then(|x| num_traits::cast::cast::<I, O>(*x)));
    PrimitiveArray::<O>::from_trusted_len_iter(iter).to(to_type.clone())
}

/// Returns a [`PrimitiveArray<i128>`] with the cast values. Values are `None` on overflow
pub fn integer_to_decimal<T: NativeType + AsPrimitive<i128>>(
    from: &PrimitiveArray<T>,
    to_precision: usize,
    to_scale: usize,
) -> PrimitiveArray<i128> {
    let multiplier = 10_i128.pow(to_scale as u32);

    let min_for_precision = 9_i128
        .saturating_pow(1 + to_precision as u32)
        .saturating_neg();
    let max_for_precision = 9_i128.saturating_pow(1 + to_precision as u32);

    let values = from.iter().map(|x| {
        x.and_then(|x| {
            x.as_().checked_mul(multiplier).and_then(|x| {
                if x > max_for_precision || x < min_for_precision {
                    None
                } else {
                    Some(x)
                }
            })
        })
    });

    PrimitiveArray::<i128>::from_trusted_len_iter(values)
        .to(ArrowDataType::Decimal(to_precision, to_scale))
}

pub(super) fn integer_to_decimal_dyn<T>(
    from: &dyn Array,
    precision: usize,
    scale: usize,
) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType + AsPrimitive<i128>,
{
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(integer_to_decimal::<T>(from, precision, scale)))
}

/// Returns a [`PrimitiveArray<i128>`] with the cast values. Values are `None` on overflow
pub fn float_to_decimal<T>(
    from: &PrimitiveArray<T>,
    to_precision: usize,
    to_scale: usize,
) -> PrimitiveArray<i128>
where
    T: NativeType + Float + ToPrimitive,
    f64: AsPrimitive<T>,
{
    // 1.2 => 12
    let multiplier: T = (10_f64).powi(to_scale as i32).as_();

    let min_for_precision = 9_i128
        .saturating_pow(1 + to_precision as u32)
        .saturating_neg();
    let max_for_precision = 9_i128.saturating_pow(1 + to_precision as u32);

    let values = from.iter().map(|x| {
        x.and_then(|x| {
            let x = (*x * multiplier).to_i128()?;
            if x > max_for_precision || x < min_for_precision {
                None
            } else {
                Some(x)
            }
        })
    });

    PrimitiveArray::<i128>::from_trusted_len_iter(values)
        .to(ArrowDataType::Decimal(to_precision, to_scale))
}

pub(super) fn float_to_decimal_dyn<T>(
    from: &dyn Array,
    precision: usize,
    scale: usize,
) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType + Float + ToPrimitive,
    f64: AsPrimitive<T>,
{
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(float_to_decimal::<T>(from, precision, scale)))
}

/// Cast [`PrimitiveArray`] as a [`PrimitiveArray`]
/// Same as `number as to_number_type` in rust
pub fn primitive_as_primitive<I, O>(
    from: &PrimitiveArray<I>,
    to_type: &ArrowDataType,
) -> PrimitiveArray<O>
where
    I: NativeType + num_traits::AsPrimitive<O>,
    O: NativeType,
{
    unary(from, num_traits::AsPrimitive::<O>::as_, to_type.clone())
}

/// Cast [`PrimitiveArray`] to a [`PrimitiveArray`] of the same physical type.
/// This is O(1).
pub fn primitive_to_same_primitive<T>(
    from: &PrimitiveArray<T>,
    to_type: &ArrowDataType,
) -> PrimitiveArray<T>
where
    T: NativeType,
{
    PrimitiveArray::<T>::new(
        to_type.clone(),
        from.values().clone(),
        from.validity().cloned(),
    )
}

/// Cast [`PrimitiveArray`] to a [`PrimitiveArray`] of the same physical type.
/// This is O(1).
pub(super) fn primitive_to_same_primitive_dyn<T>(
    from: &dyn Array,
    to_type: &ArrowDataType,
) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType,
{
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(primitive_to_same_primitive::<T>(from, to_type)))
}

pub(super) fn primitive_to_dictionary_dyn<T: NativeType + Eq + Hash, K: DictionaryKey>(
    from: &dyn Array,
) -> PolarsResult<Box<dyn Array>> {
    let from = from.as_any().downcast_ref().unwrap();
    primitive_to_dictionary::<T, K>(from).map(|x| Box::new(x) as Box<dyn Array>)
}

/// Cast [`PrimitiveArray`] to [`DictionaryArray`]. Also known as packing.
/// # Errors
/// This function errors if the maximum key is smaller than the number of distinct elements
/// in the array.
pub fn primitive_to_dictionary<T: NativeType + Eq + Hash, K: DictionaryKey>(
    from: &PrimitiveArray<T>,
) -> PolarsResult<DictionaryArray<K>> {
    let iter = from.iter().map(|x| x.copied());
    let mut array = MutableDictionaryArray::<K, _>::try_empty(MutablePrimitiveArray::<T>::from(
        from.dtype().clone(),
    ))?;
    array.reserve(from.len());
    array.try_extend(iter)?;

    Ok(array.into())
}

/// # Safety
///
/// `dtype` should be valid for primitive.
pub unsafe fn primitive_map_is_valid<T: NativeType>(
    from: &PrimitiveArray<T>,
    f: impl Fn(T) -> bool,
    dtype: ArrowDataType,
) -> PrimitiveArray<T> {
    let values = from.values().clone();

    let validity: Bitmap = values.iter().map(|&v| f(v)).collect();

    let validity = if validity.unset_bits() > 0 {
        let new_validity = match from.validity() {
            None => validity,
            Some(v) => v & &validity,
        };

        Some(new_validity)
    } else {
        from.validity().cloned()
    };

    // SAFETY:
    // - Validity did not change length
    // - dtype should be valid
    unsafe { PrimitiveArray::new_unchecked(dtype, values, validity) }
}

/// Conversion of `Int32` to `Time32(TimeUnit::Second)`
pub fn int32_to_time32s(from: &PrimitiveArray<i32>) -> PrimitiveArray<i32> {
    // SAFETY: Time32(TimeUnit::Second) is valid for Int32
    unsafe {
        primitive_map_is_valid(
            from,
            |v| (0..SECONDS_IN_DAY as i32).contains(&v),
            ArrowDataType::Time32(TimeUnit::Second),
        )
    }
}

/// Conversion of `Int32` to `Time32(TimeUnit::Millisecond)`
pub fn int32_to_time32ms(from: &PrimitiveArray<i32>) -> PrimitiveArray<i32> {
    // SAFETY: Time32(TimeUnit::Millisecond) is valid for Int32
    unsafe {
        primitive_map_is_valid(
            from,
            |v| (0..MILLISECONDS_IN_DAY as i32).contains(&v),
            ArrowDataType::Time32(TimeUnit::Millisecond),
        )
    }
}

/// Conversion of `Int64` to `Time32(TimeUnit::Microsecond)`
pub fn int64_to_time64us(from: &PrimitiveArray<i64>) -> PrimitiveArray<i64> {
    // SAFETY: Time64(TimeUnit::Microsecond) is valid for Int64
    unsafe {
        primitive_map_is_valid(
            from,
            |v| (0..MICROSECONDS_IN_DAY).contains(&v),
            ArrowDataType::Time32(TimeUnit::Microsecond),
        )
    }
}

/// Conversion of `Int64` to `Time32(TimeUnit::Nanosecond)`
pub fn int64_to_time64ns(from: &PrimitiveArray<i64>) -> PrimitiveArray<i64> {
    // SAFETY: Time64(TimeUnit::Nanosecond) is valid for Int64
    unsafe {
        primitive_map_is_valid(
            from,
            |v| (0..NANOSECONDS_IN_DAY).contains(&v),
            ArrowDataType::Time64(TimeUnit::Nanosecond),
        )
    }
}

/// Conversion of dates
pub fn date32_to_date64(from: &PrimitiveArray<i32>) -> PrimitiveArray<i64> {
    unary(
        from,
        |x| x as i64 * MILLISECONDS_IN_DAY,
        ArrowDataType::Date64,
    )
}

/// Conversion of dates
pub fn date64_to_date32(from: &PrimitiveArray<i64>) -> PrimitiveArray<i32> {
    unary(
        from,
        |x| (x / MILLISECONDS_IN_DAY) as i32,
        ArrowDataType::Date32,
    )
}

/// Conversion of times
pub fn time32s_to_time32ms(from: &PrimitiveArray<i32>) -> PrimitiveArray<i32> {
    unary(
        from,
        |x| x * 1000,
        ArrowDataType::Time32(TimeUnit::Millisecond),
    )
}

/// Conversion of times
pub fn time32ms_to_time32s(from: &PrimitiveArray<i32>) -> PrimitiveArray<i32> {
    unary(from, |x| x / 1000, ArrowDataType::Time32(TimeUnit::Second))
}

/// Conversion of times
pub fn time64us_to_time64ns(from: &PrimitiveArray<i64>) -> PrimitiveArray<i64> {
    unary(
        from,
        |x| x * 1000,
        ArrowDataType::Time64(TimeUnit::Nanosecond),
    )
}

/// Conversion of times
pub fn time64ns_to_time64us(from: &PrimitiveArray<i64>) -> PrimitiveArray<i64> {
    unary(
        from,
        |x| x / 1000,
        ArrowDataType::Time64(TimeUnit::Microsecond),
    )
}

/// Conversion of timestamp
pub fn timestamp_to_date64(from: &PrimitiveArray<i64>, from_unit: TimeUnit) -> PrimitiveArray<i64> {
    let from_size = time_unit_multiple(from_unit);
    let to_size = MILLISECONDS;
    let to_type = ArrowDataType::Date64;

    // Scale time_array by (to_size / from_size) using a
    // single integer operation, but need to avoid integer
    // math rounding down to zero

    match to_size.cmp(&from_size) {
        std::cmp::Ordering::Less => unary(from, |x| (x / (from_size / to_size)), to_type),
        std::cmp::Ordering::Equal => primitive_to_same_primitive(from, &to_type),
        std::cmp::Ordering::Greater => unary(from, |x| (x * (to_size / from_size)), to_type),
    }
}

/// Conversion of timestamp
pub fn timestamp_to_date32(from: &PrimitiveArray<i64>, from_unit: TimeUnit) -> PrimitiveArray<i32> {
    let from_size = time_unit_multiple(from_unit) * SECONDS_IN_DAY;
    unary(from, |x| (x / from_size) as i32, ArrowDataType::Date32)
}

/// Conversion of time
pub fn time32_to_time64(
    from: &PrimitiveArray<i32>,
    from_unit: TimeUnit,
    to_unit: TimeUnit,
) -> PrimitiveArray<i64> {
    let from_size = time_unit_multiple(from_unit);
    let to_size = time_unit_multiple(to_unit);
    let divisor = to_size / from_size;
    unary(
        from,
        |x| (x as i64 * divisor),
        ArrowDataType::Time64(to_unit),
    )
}

/// Conversion of time
pub fn time64_to_time32(
    from: &PrimitiveArray<i64>,
    from_unit: TimeUnit,
    to_unit: TimeUnit,
) -> PrimitiveArray<i32> {
    let from_size = time_unit_multiple(from_unit);
    let to_size = time_unit_multiple(to_unit);
    let divisor = from_size / to_size;
    unary(
        from,
        |x| (x / divisor) as i32,
        ArrowDataType::Time32(to_unit),
    )
}

/// Conversion of timestamp
pub fn timestamp_to_timestamp(
    from: &PrimitiveArray<i64>,
    from_unit: TimeUnit,
    to_unit: TimeUnit,
    tz: &Option<PlSmallStr>,
) -> PrimitiveArray<i64> {
    let from_size = time_unit_multiple(from_unit);
    let to_size = time_unit_multiple(to_unit);
    let to_type = ArrowDataType::Timestamp(to_unit, tz.clone());
    // we either divide or multiply, depending on size of each unit
    if from_size >= to_size {
        unary(from, |x| (x / (from_size / to_size)), to_type)
    } else {
        unary(from, |x| (x * (to_size / from_size)), to_type)
    }
}

/// Casts f16 into f32
pub fn f16_to_f32(from: &PrimitiveArray<f16>) -> PrimitiveArray<f32> {
    unary(from, |x| x.to_f32(), ArrowDataType::Float32)
}

/// Returns a [`Utf8Array`] where every element is the utf8 representation of the number.
pub(super) fn primitive_to_binview<T: NativeType + SerPrimitive>(
    from: &PrimitiveArray<T>,
) -> BinaryViewArray {
    let mut mutable = MutableBinaryViewArray::with_capacity(from.len());

    let mut scratch = vec![];
    for &x in from.values().iter() {
        unsafe { scratch.set_len(0) };
        T::write(&mut scratch, x);
        mutable.push_value_ignore_validity(&scratch)
    }

    mutable.freeze().with_validity(from.validity().cloned())
}

pub(super) fn primitive_to_binview_dyn<T>(from: &dyn Array) -> BinaryViewArray
where
    T: NativeType + SerPrimitive,
{
    let from = from.as_any().downcast_ref().unwrap();
    primitive_to_binview::<T>(from)
}
