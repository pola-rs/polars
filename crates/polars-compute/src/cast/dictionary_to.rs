use arrow::array::{Array, DictionaryArray, DictionaryKey};
use arrow::datatypes::ArrowDataType;
use arrow::match_integer_type;
use polars_error::{PolarsResult, polars_bail};

use super::{CastOptionsImpl, cast, primitive_to_primitive};

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
