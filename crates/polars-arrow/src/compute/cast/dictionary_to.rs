use polars_error::{polars_bail, PolarsResult};

use super::{primitive_as_primitive, primitive_to_primitive, CastOptionsImpl};
use crate::array::{Array, DictionaryArray, DictionaryKey};
use crate::compute::cast::cast;
use crate::datatypes::ArrowDataType;
use crate::match_integer_type;

macro_rules! key_cast {
    ($keys:expr, $values:expr, $array:expr, $to_keys_type:expr, $to_type:ty, $to_datatype:expr) => {{
        let cast_keys = primitive_to_primitive::<_, $to_type>($keys, $to_keys_type);

        // Failure to cast keys (because they don't fit in the
        // target type) results in NULL values;
        if cast_keys.null_count() > $keys.null_count() {
            polars_bail!(ComputeError: "overflow")
        }
        // SAFETY: this is safe because given a type `T` that fits in a `usize`, casting it to type `P` either overflows or also fits in a `usize`
        unsafe {
             DictionaryArray::try_new_unchecked($to_datatype, cast_keys, $values.clone())
        }
            .map(|x| x.boxed())
    }};
}

/// Casts a [`DictionaryArray`] to a new [`DictionaryArray`] by keeping the
/// keys and casting the values to `values_type`.
/// # Errors
/// This function errors if the values are not castable to `values_type`
pub fn dictionary_to_dictionary_values<K: DictionaryKey>(
    from: &DictionaryArray<K>,
    values_type: &ArrowDataType,
) -> PolarsResult<DictionaryArray<K>> {
    let keys = from.keys();
    let values = from.values();
    let length = values.len();

    let values = cast(values.as_ref(), values_type, CastOptionsImpl::default())?;

    assert_eq!(values.len(), length); // this is guaranteed by `cast`
    unsafe {
        DictionaryArray::try_new_unchecked(from.data_type().clone(), keys.clone(), values.clone())
    }
}

/// Similar to dictionary_to_dictionary_values, but overflowing cast is wrapped
pub fn wrapping_dictionary_to_dictionary_values<K: DictionaryKey>(
    from: &DictionaryArray<K>,
    values_type: &ArrowDataType,
) -> PolarsResult<DictionaryArray<K>> {
    let keys = from.keys();
    let values = from.values();
    let length = values.len();

    let values = cast(
        values.as_ref(),
        values_type,
        CastOptionsImpl {
            wrapped: true,
            partial: false,
        },
    )?;
    assert_eq!(values.len(), length); // this is guaranteed by `cast`
    unsafe {
        DictionaryArray::try_new_unchecked(from.data_type().clone(), keys.clone(), values.clone())
    }
}

/// Casts a [`DictionaryArray`] to a new [`DictionaryArray`] backed by a
/// different physical type of the keys, while keeping the values equal.
/// # Errors
/// Errors if any of the old keys' values is larger than the maximum value
/// supported by the new physical type.
pub fn dictionary_to_dictionary_keys<K1, K2>(
    from: &DictionaryArray<K1>,
) -> PolarsResult<DictionaryArray<K2>>
where
    K1: DictionaryKey + num_traits::NumCast,
    K2: DictionaryKey + num_traits::NumCast,
{
    let keys = from.keys();
    let values = from.values();
    let is_ordered = from.is_ordered();

    let casted_keys = primitive_to_primitive::<K1, K2>(keys, &K2::PRIMITIVE.into());

    if casted_keys.null_count() > keys.null_count() {
        polars_bail!(ComputeError: "overflow")
    } else {
        let data_type = ArrowDataType::Dictionary(
            K2::KEY_TYPE,
            Box::new(values.data_type().clone()),
            is_ordered,
        );
        // SAFETY: this is safe because given a type `T` that fits in a `usize`, casting it to type `P` either overflows or also fits in a `usize`
        unsafe { DictionaryArray::try_new_unchecked(data_type, casted_keys, values.clone()) }
    }
}

/// Similar to dictionary_to_dictionary_keys, but overflowing cast is wrapped
pub fn wrapping_dictionary_to_dictionary_keys<K1, K2>(
    from: &DictionaryArray<K1>,
) -> PolarsResult<DictionaryArray<K2>>
where
    K1: DictionaryKey + num_traits::AsPrimitive<K2>,
    K2: DictionaryKey,
{
    let keys = from.keys();
    let values = from.values();
    let is_ordered = from.is_ordered();

    let casted_keys = primitive_as_primitive::<K1, K2>(keys, &K2::PRIMITIVE.into());

    if casted_keys.null_count() > keys.null_count() {
        polars_bail!(ComputeError: "overflow")
    } else {
        let data_type = ArrowDataType::Dictionary(
            K2::KEY_TYPE,
            Box::new(values.data_type().clone()),
            is_ordered,
        );
        // some of the values may not fit in `usize` and thus this needs to be checked
        DictionaryArray::try_new(data_type, casted_keys, values.clone())
    }
}

pub(super) fn dictionary_cast_dyn<K: DictionaryKey + num_traits::NumCast>(
    array: &dyn Array,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn Array>> {
    let array = array.as_any().downcast_ref::<DictionaryArray<K>>().unwrap();
    let keys = array.keys();
    let values = array.values();

    match to_type {
        ArrowDataType::Dictionary(to_keys_type, to_values_type, _) => {
            let values = cast(values.as_ref(), to_values_type, options)?;

            // create the appropriate array type
            let to_key_type = (*to_keys_type).into();

            // SAFETY:
            // we return an error on overflow so the integers remain within bounds
            match_integer_type!(to_keys_type, |$T| {
                key_cast!(keys, values, array, &to_key_type, $T, to_type.clone())
            })
        },
        _ => unimplemented!(),
    }
}
