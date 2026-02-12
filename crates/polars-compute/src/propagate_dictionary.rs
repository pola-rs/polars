use arrow::array::{Array, BinaryViewArray, PrimitiveArray, Utf8ViewArray};
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType::UInt32;

/// Propagate the nulls from the dictionary values into the keys and remove those nulls from the
/// values.
pub fn propagate_dictionary_value_nulls(
    keys: &PrimitiveArray<u32>,
    values: &Utf8ViewArray,
) -> (PrimitiveArray<u32>, Utf8ViewArray) {
    let Some(values_validity) = values.validity() else {
        return (keys.clone(), values.clone().with_validity(None));
    };
    if values_validity.unset_bits() == 0 {
        return (keys.clone(), values.clone().with_validity(None));
    }

    let num_values = values.len();

    // Create a map from the old indices to indices with nulls filtered out
    let mut offset = 0;
    let new_idx_map: Vec<u32> = (0..num_values)
        .map(|i| {
            let is_valid = unsafe { values_validity.get_bit_unchecked(i) };
            offset += usize::from(!is_valid);
            if is_valid { (i - offset) as u32 } else { 0 }
        })
        .collect();

    let keys = match keys.validity() {
        None => {
            let values = keys
                .values()
                .iter()
                .map(|&k| unsafe {
                    // SAFETY: Arrow invariant that all keys are in range of values
                    *new_idx_map.get_unchecked(k as usize)
                })
                .collect();
            let validity = Bitmap::from_iter(keys.values().iter().map(|&k| unsafe {
                // SAFETY: Arrow invariant that all keys are in range of values
                values_validity.get_bit_unchecked(k as usize)
            }));

            PrimitiveArray::new(UInt32, values, Some(validity))
        },
        Some(keys_validity) => {
            let values = keys
                .values()
                .iter()
                .map(|&k| {
                    // deal with nulls in keys
                    let idx = (k as usize).min(num_values);
                    // SAFETY: Arrow invariant that all keys are in range of values
                    *unsafe { new_idx_map.get_unchecked(idx) }
                })
                .collect();
            let propagated_validity = Bitmap::from_iter(keys.values().iter().map(|&k| {
                // deal with nulls in keys
                let idx = (k as usize).min(num_values);
                // SAFETY: Arrow invariant that all keys are in range of values
                unsafe { values_validity.get_bit_unchecked(idx) }
            }));

            let validity = &propagated_validity & keys_validity;
            PrimitiveArray::new(UInt32, values, Some(validity))
        },
    };

    // Filter only handles binary
    let values = values.to_binview();

    // Filter out the null values
    let values = crate::filter::filter_with_bitmap(&values, values_validity);
    let values = values.as_any().downcast_ref::<BinaryViewArray>().unwrap();
    let values = unsafe { values.to_utf8view_unchecked() };

    // Explicitly set the values validity to none.
    assert_eq!(values.null_count(), 0);
    let values = values.with_validity(None);

    (keys, values)
}
