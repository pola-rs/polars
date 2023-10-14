use polars_error::PolarsResult;

use super::CastOptions;
use crate::array::*;
use crate::datatypes::DataType;
use crate::offset::{Offset, Offsets};
use crate::types::NativeType;

/// Conversion of binary
pub fn binary_to_large_binary(from: &BinaryArray<i32>, to_data_type: DataType) -> BinaryArray<i64> {
    let values = from.values().clone();
    BinaryArray::<i64>::new(
        to_data_type,
        from.offsets().into(),
        values,
        from.validity().cloned(),
    )
}

/// Conversion of binary
pub fn binary_large_to_binary(
    from: &BinaryArray<i64>,
    to_data_type: DataType,
) -> PolarsResult<BinaryArray<i32>> {
    let values = from.values().clone();
    let offsets = from.offsets().try_into()?;
    Ok(BinaryArray::<i32>::new(
        to_data_type,
        offsets,
        values,
        from.validity().cloned(),
    ))
}

/// Conversion to utf8
pub fn binary_to_utf8<O: Offset>(
    from: &BinaryArray<O>,
    to_data_type: DataType,
) -> PolarsResult<Utf8Array<O>> {
    Utf8Array::<O>::try_new(
        to_data_type,
        from.offsets().clone(),
        from.values().clone(),
        from.validity().cloned(),
    )
}

/// Conversion to utf8
/// # Errors
/// This function errors if the values are not valid utf8
pub fn binary_to_large_utf8(
    from: &BinaryArray<i32>,
    to_data_type: DataType,
) -> PolarsResult<Utf8Array<i64>> {
    let values = from.values().clone();
    let offsets = from.offsets().into();

    Utf8Array::<i64>::try_new(to_data_type, offsets, values, from.validity().cloned())
}

/// Casts a [`BinaryArray`] to a [`PrimitiveArray`] at best-effort using `lexical_core::parse_partial`, making any uncastable value as zero.
pub fn partial_binary_to_primitive<O: Offset, T>(
    from: &BinaryArray<O>,
    to: &DataType,
) -> PrimitiveArray<T>
where
    T: NativeType + lexical_core::FromLexical,
{
    let iter = from
        .iter()
        .map(|x| x.and_then::<T, _>(|x| lexical_core::parse_partial(x).ok().map(|x| x.0)));

    PrimitiveArray::<T>::from_trusted_len_iter(iter).to(to.clone())
}

/// Casts a [`BinaryArray`] to a [`PrimitiveArray`], making any uncastable value a Null.
pub fn binary_to_primitive<O: Offset, T>(from: &BinaryArray<O>, to: &DataType) -> PrimitiveArray<T>
where
    T: NativeType + lexical_core::FromLexical,
{
    let iter = from
        .iter()
        .map(|x| x.and_then::<T, _>(|x| lexical_core::parse(x).ok()));

    PrimitiveArray::<T>::from_trusted_len_iter(iter).to(to.clone())
}

pub(super) fn binary_to_primitive_dyn<O: Offset, T>(
    from: &dyn Array,
    to: &DataType,
    options: CastOptions,
) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType + lexical_core::FromLexical,
{
    let from = from.as_any().downcast_ref().unwrap();
    if options.partial {
        Ok(Box::new(partial_binary_to_primitive::<O, T>(from, to)))
    } else {
        Ok(Box::new(binary_to_primitive::<O, T>(from, to)))
    }
}

/// Cast [`BinaryArray`] to [`DictionaryArray`], also known as packing.
/// # Errors
/// This function errors if the maximum key is smaller than the number of distinct elements
/// in the array.
pub fn binary_to_dictionary<O: Offset, K: DictionaryKey>(
    from: &BinaryArray<O>,
) -> PolarsResult<DictionaryArray<K>> {
    let mut array = MutableDictionaryArray::<K, MutableBinaryArray<O>>::new();
    array.try_extend(from.iter())?;

    Ok(array.into())
}

pub(super) fn binary_to_dictionary_dyn<O: Offset, K: DictionaryKey>(
    from: &dyn Array,
) -> PolarsResult<Box<dyn Array>> {
    let values = from.as_any().downcast_ref().unwrap();
    binary_to_dictionary::<O, K>(values).map(|x| Box::new(x) as Box<dyn Array>)
}

fn fixed_size_to_offsets<O: Offset>(values_len: usize, fixed_size: usize) -> Offsets<O> {
    let offsets = (0..(values_len + 1))
        .step_by(fixed_size)
        .map(|v| O::from_as_usize(v))
        .collect();
    // Safety
    // * every element is `>= 0`
    // * element at position `i` is >= than element at position `i-1`.
    unsafe { Offsets::new_unchecked(offsets) }
}

/// Conversion of `FixedSizeBinary` to `Binary`.
pub fn fixed_size_binary_binary<O: Offset>(
    from: &FixedSizeBinaryArray,
    to_data_type: DataType,
) -> BinaryArray<O> {
    let values = from.values().clone();
    let offsets = fixed_size_to_offsets(values.len(), from.size());
    BinaryArray::<O>::new(
        to_data_type,
        offsets.into(),
        values,
        from.validity().cloned(),
    )
}

/// Conversion of binary
pub fn binary_to_list<O: Offset>(from: &BinaryArray<O>, to_data_type: DataType) -> ListArray<O> {
    let values = from.values().clone();
    let values = PrimitiveArray::new(DataType::UInt8, values, None);
    ListArray::<O>::new(
        to_data_type,
        from.offsets().clone(),
        values.boxed(),
        from.validity().cloned(),
    )
}
