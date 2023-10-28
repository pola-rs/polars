use chrono::Datelike;
use polars_error::PolarsResult;

use super::CastOptions;
use crate::array::*;
use crate::datatypes::{DataType, TimeUnit};
use crate::offset::Offset;
use crate::temporal_conversions::{
    utf8_to_naive_timestamp as utf8_to_naive_timestamp_, utf8_to_timestamp as utf8_to_timestamp_,
    EPOCH_DAYS_FROM_CE,
};
use crate::types::NativeType;

const RFC3339: &str = "%Y-%m-%dT%H:%M:%S%.f%:z";

/// Casts a [`Utf8Array`] to a [`PrimitiveArray`], making any uncastable value a Null.
pub fn utf8_to_primitive<O: Offset, T>(from: &Utf8Array<O>, to: &DataType) -> PrimitiveArray<T>
where
    T: NativeType + lexical_core::FromLexical,
{
    let iter = from
        .iter()
        .map(|x| x.and_then::<T, _>(|x| lexical_core::parse(x.as_bytes()).ok()));

    PrimitiveArray::<T>::from_trusted_len_iter(iter).to(to.clone())
}

/// Casts a [`Utf8Array`] to a [`PrimitiveArray`] at best-effort using `lexical_core::parse_partial`, making any uncastable value as zero.
pub fn partial_utf8_to_primitive<O: Offset, T>(
    from: &Utf8Array<O>,
    to: &DataType,
) -> PrimitiveArray<T>
where
    T: NativeType + lexical_core::FromLexical,
{
    let iter = from.iter().map(|x| {
        x.and_then::<T, _>(|x| lexical_core::parse_partial(x.as_bytes()).ok().map(|x| x.0))
    });

    PrimitiveArray::<T>::from_trusted_len_iter(iter).to(to.clone())
}

pub(super) fn utf8_to_primitive_dyn<O: Offset, T>(
    from: &dyn Array,
    to: &DataType,
    options: CastOptions,
) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType + lexical_core::FromLexical,
{
    let from = from.as_any().downcast_ref().unwrap();
    if options.partial {
        Ok(Box::new(partial_utf8_to_primitive::<O, T>(from, to)))
    } else {
        Ok(Box::new(utf8_to_primitive::<O, T>(from, to)))
    }
}

/// Casts a [`Utf8Array`] to a Date32 primitive, making any uncastable value a Null.
pub fn utf8_to_date32<O: Offset>(from: &Utf8Array<O>) -> PrimitiveArray<i32> {
    let iter = from.iter().map(|x| {
        x.and_then(|x| {
            x.parse::<chrono::NaiveDate>()
                .ok()
                .map(|x| x.num_days_from_ce() - EPOCH_DAYS_FROM_CE)
        })
    });
    PrimitiveArray::<i32>::from_trusted_len_iter(iter).to(DataType::Date32)
}

pub(super) fn utf8_to_date32_dyn<O: Offset>(from: &dyn Array) -> PolarsResult<Box<dyn Array>> {
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(utf8_to_date32::<O>(from)))
}

/// Casts a [`Utf8Array`] to a Date64 primitive, making any uncastable value a Null.
pub fn utf8_to_date64<O: Offset>(from: &Utf8Array<O>) -> PrimitiveArray<i64> {
    let iter = from.iter().map(|x| {
        x.and_then(|x| {
            x.parse::<chrono::NaiveDate>()
                .ok()
                .map(|x| (x.num_days_from_ce() - EPOCH_DAYS_FROM_CE) as i64 * 86400000)
        })
    });
    PrimitiveArray::from_trusted_len_iter(iter).to(DataType::Date64)
}

pub(super) fn utf8_to_date64_dyn<O: Offset>(from: &dyn Array) -> PolarsResult<Box<dyn Array>> {
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(utf8_to_date64::<O>(from)))
}

pub(super) fn utf8_to_dictionary_dyn<O: Offset, K: DictionaryKey>(
    from: &dyn Array,
) -> PolarsResult<Box<dyn Array>> {
    let values = from.as_any().downcast_ref().unwrap();
    utf8_to_dictionary::<O, K>(values).map(|x| Box::new(x) as Box<dyn Array>)
}

/// Cast [`Utf8Array`] to [`DictionaryArray`], also known as packing.
/// # Errors
/// This function errors if the maximum key is smaller than the number of distinct elements
/// in the array.
pub fn utf8_to_dictionary<O: Offset, K: DictionaryKey>(
    from: &Utf8Array<O>,
) -> PolarsResult<DictionaryArray<K>> {
    let mut array = MutableDictionaryArray::<K, MutableUtf8Array<O>>::new();
    array.try_extend(from.iter())?;

    Ok(array.into())
}

pub(super) fn utf8_to_naive_timestamp_dyn<O: Offset>(
    from: &dyn Array,
    time_unit: TimeUnit,
) -> PolarsResult<Box<dyn Array>> {
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(utf8_to_naive_timestamp::<O>(from, time_unit)))
}

/// [`crate::temporal_conversions::utf8_to_timestamp`] applied for RFC3339 formatting
pub fn utf8_to_naive_timestamp<O: Offset>(
    from: &Utf8Array<O>,
    time_unit: TimeUnit,
) -> PrimitiveArray<i64> {
    utf8_to_naive_timestamp_(from, RFC3339, time_unit)
}

pub(super) fn utf8_to_timestamp_dyn<O: Offset>(
    from: &dyn Array,
    timezone: String,
    time_unit: TimeUnit,
) -> PolarsResult<Box<dyn Array>> {
    let from = from.as_any().downcast_ref().unwrap();
    utf8_to_timestamp::<O>(from, timezone, time_unit)
        .map(Box::new)
        .map(|x| x as Box<dyn Array>)
}

/// [`crate::temporal_conversions::utf8_to_timestamp`] applied for RFC3339 formatting
pub fn utf8_to_timestamp<O: Offset>(
    from: &Utf8Array<O>,
    timezone: String,
    time_unit: TimeUnit,
) -> PolarsResult<PrimitiveArray<i64>> {
    utf8_to_timestamp_(from, RFC3339, timezone, time_unit)
}

/// Conversion of utf8
pub fn utf8_to_large_utf8(from: &Utf8Array<i32>) -> Utf8Array<i64> {
    let data_type = Utf8Array::<i64>::default_data_type();
    let validity = from.validity().cloned();
    let values = from.values().clone();

    let offsets = from.offsets().into();
    // Safety: sound because `values` fulfills the same invariants as `from.values()`
    unsafe { Utf8Array::<i64>::new_unchecked(data_type, offsets, values, validity) }
}

/// Conversion of utf8
pub fn utf8_large_to_utf8(from: &Utf8Array<i64>) -> PolarsResult<Utf8Array<i32>> {
    let data_type = Utf8Array::<i32>::default_data_type();
    let validity = from.validity().cloned();
    let values = from.values().clone();
    let offsets = from.offsets().try_into()?;

    // Safety: sound because `values` fulfills the same invariants as `from.values()`
    Ok(unsafe { Utf8Array::<i32>::new_unchecked(data_type, offsets, values, validity) })
}

/// Conversion to binary
pub fn utf8_to_binary<O: Offset>(from: &Utf8Array<O>, to_data_type: DataType) -> BinaryArray<O> {
    // Safety: erasure of an invariant is always safe
    unsafe {
        BinaryArray::<O>::new(
            to_data_type,
            from.offsets().clone(),
            from.values().clone(),
            from.validity().cloned(),
        )
    }
}
