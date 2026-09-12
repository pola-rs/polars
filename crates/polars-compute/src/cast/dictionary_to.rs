//! Packing the Arrow dictionaries, whose elements no array of `polars-array` holds.

use arrow::array::*;
use arrow::datatypes::ArrowDataType;
use arrow::match_integer_type;
use arrow::types::NativeType;
use polars_array::arrow::{export, import};
use polars_error::{PolarsResult, polars_bail, polars_ensure};

use super::arrow_kernels::{binview_to_arrow_large_binary, utf8view_to_arrow_large_utf8};

/// Packs `array` into the dictionary `to_type` names, which no array of `polars-array` holds.
pub fn cast_to_dictionary(
    array: &dyn Array,
    to_type: &ArrowDataType,
) -> PolarsResult<Box<dyn Array>> {
    let ArrowDataType::Dictionary(index_type, value_type, ordered) = to_type else {
        polars_bail!(InvalidOperation: "casting to {to_type:?} is not a dictionary");
    };

    // An array that is a dictionary already is packed: what a cast off one changes is the width of
    // its keys, which is what a consumer that reads them as a signed integer needs.
    if let ArrowDataType::Dictionary(from_index_type, _, _) = array.dtype() {
        return match_integer_type!(from_index_type, |$F| {
            recast_dictionary::<$F>(downcast(array), index_type, value_type, to_type)
        });
    }

    match_integer_type!(index_type, |$K| {
        pack_dictionary::<$K>(array, value_type, *ordered)
    })
}

/// Reads the keys of a dictionary as keys of another width, which leaves its values as they are.
fn recast_dictionary<F: DictionaryKey + num_traits::NumCast>(
    array: &DictionaryArray<F>,
    to_index_type: &arrow::datatypes::IntegerType,
    to_value_type: &ArrowDataType,
    to_type: &ArrowDataType,
) -> PolarsResult<Box<dyn Array>> {
    let values = cast_dictionary_values(array.values().as_ref(), to_value_type)?;
    let keys = import::primitive_from_arrow(array.keys());
    let to_key_type: ArrowDataType = (*to_index_type).into();

    match_integer_type!(to_index_type, |$T| {
        let cast_keys = super::numeric_to_numeric_checked::<F, $T>(&keys);

        // A key that does not fit the target width reads as null, which names no value at all.
        polars_ensure!(
            cast_keys.null_count() == array.keys().null_count(),
            ComputeError: "overflow"
        );

        let cast_keys = export::primitive_to_arrow_primitive(&cast_keys).to(to_key_type.clone());

        // SAFETY: the keys were read off the ones of `array`, which are in bounds of its values.
        unsafe { DictionaryArray::<$T>::try_new_unchecked(to_type.clone(), cast_keys, values.clone()) }
            .map(|array| array.boxed())
    })
}

/// Packs the values of `array`, cast to `value_type`, into a dictionary keyed by `K`.
fn pack_dictionary<K: DictionaryKey>(
    array: &dyn Array,
    value_type: &ArrowDataType,
    ordered: bool,
) -> PolarsResult<Box<dyn Array>> {
    let array = cast_dictionary_values(array, value_type)?;
    let array = array.as_ref();

    match value_type.to_storage() {
        ArrowDataType::Int8 => primitive_to_dictionary::<i8, K>(array, ordered),
        ArrowDataType::Int16 => primitive_to_dictionary::<i16, K>(array, ordered),
        ArrowDataType::Int32 | ArrowDataType::Date32 => {
            primitive_to_dictionary::<i32, K>(array, ordered)
        },
        ArrowDataType::Int64 | ArrowDataType::Time64(_) | ArrowDataType::Timestamp(_, _) => {
            primitive_to_dictionary::<i64, K>(array, ordered)
        },
        ArrowDataType::UInt8 => primitive_to_dictionary::<u8, K>(array, ordered),
        ArrowDataType::UInt16 => primitive_to_dictionary::<u16, K>(array, ordered),
        ArrowDataType::UInt32 => primitive_to_dictionary::<u32, K>(array, ordered),
        ArrowDataType::UInt64 => primitive_to_dictionary::<u64, K>(array, ordered),
        ArrowDataType::BinaryView => {
            let array: &BinaryViewArray = downcast(array);
            let mut dictionary =
                MutableDictionaryArray::<K, MutableBinaryViewArray<[u8]>>::new(ordered);
            dictionary.reserve(array.len());
            dictionary.try_extend(array.iter())?;
            Ok(DictionaryArray::<K>::from(dictionary).boxed())
        },
        ArrowDataType::Utf8View => {
            let array: &Utf8ViewArray = downcast(array);
            let mut dictionary =
                MutableDictionaryArray::<K, MutableBinaryViewArray<str>>::new(ordered);
            dictionary.reserve(array.len());
            dictionary.try_extend(array.iter())?;
            Ok(DictionaryArray::<K>::from(dictionary).boxed())
        },
        ArrowDataType::LargeUtf8 => {
            let array: &Utf8Array<i64> = downcast(array);
            let mut dictionary =
                MutableDictionaryArray::<K, MutableUtf8Array<i64>>::empty_with_value_dtype(
                    array.dtype().clone(),
                    ordered,
                );
            dictionary.reserve(array.len());
            dictionary.try_extend(array.iter())?;
            Ok(DictionaryArray::<K>::from(dictionary).boxed())
        },
        _ => polars_bail!(ComputeError:
            "unsupported output type for dictionary packing: {value_type:?}"
        ),
    }
}

/// Casts the values a dictionary is packed out of, read as the Arrow type they are written as.
fn cast_dictionary_values(
    array: &dyn Array,
    to_type: &ArrowDataType,
) -> PolarsResult<Box<dyn Array>> {
    use ArrowDataType as A;

    if array.dtype() == to_type {
        return Ok(array.to_boxed());
    }

    match (array.dtype(), to_type) {
        (A::Utf8View, A::LargeUtf8) => Ok(utf8view_to_arrow_large_utf8(downcast(array)).boxed()),
        (A::BinaryView, A::LargeBinary) => {
            Ok(binview_to_arrow_large_binary(downcast(array)).boxed())
        },
        (from_type, to_type) => {
            polars_bail!(InvalidOperation: "casting from {from_type:?} to {to_type:?} not supported")
        },
    }
}

/// Packs the values of a primitive array into a dictionary keyed by `K`, also known as packing.
fn primitive_to_dictionary<T: NativeType + std::hash::Hash + Eq, K: DictionaryKey>(
    array: &dyn Array,
    ordered: bool,
) -> PolarsResult<Box<dyn Array>> {
    let array: &PrimitiveArray<T> = downcast(array);

    let mut dictionary = MutableDictionaryArray::<K, _>::try_empty(
        MutablePrimitiveArray::<T>::from(array.dtype().clone()),
        ordered,
    )?;
    dictionary.reserve(array.len());
    dictionary.try_extend(array.iter().map(|value| value.copied()))?;

    Ok(DictionaryArray::<K>::from(dictionary).boxed())
}

#[inline]
fn downcast<A: Array + 'static>(array: &dyn Array) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the data type of an arrow array names the array it downcasts to")
}
