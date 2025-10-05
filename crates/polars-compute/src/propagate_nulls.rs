use arrow::array::{Array, FixedSizeListArray, ListArray, StructArray};
use arrow::bitmap::BitmapBuilder;
use arrow::bitmap::bitmask::BitMask;
use arrow::types::Offset;

/// Propagate nulls down to masked-out values in lower nesting levels.
pub fn propagate_nulls(arr: &dyn Array) -> Option<Box<dyn Array>> {
    let arr = arr.as_any();
    if let Some(arr) = arr.downcast_ref::<ListArray<i32>>() {
        return propagate_nulls_list(arr).map(|arr| Box::new(arr) as _);
    }
    if let Some(arr) = arr.downcast_ref::<ListArray<i64>>() {
        return propagate_nulls_list(arr).map(|arr| Box::new(arr) as _);
    }
    if let Some(arr) = arr.downcast_ref::<FixedSizeListArray>() {
        return propagate_nulls_fsl(arr).map(|arr| Box::new(arr) as _);
    }
    if let Some(arr) = arr.downcast_ref::<StructArray>() {
        return propagate_nulls_struct(arr).map(|arr| Box::new(arr) as _);
    }

    None
}

pub fn propagate_nulls_list<O: Offset>(arr: &ListArray<O>) -> Option<ListArray<O>> {
    let Some(validity) = arr.validity() else {
        return propagate_nulls(arr.values().as_ref()).map(|values| {
            ListArray::new(arr.dtype().clone(), arr.offsets().clone(), values, None)
        });
    };

    let mut last_idx = 0;
    let old_child_validity = arr.values().validity();
    let mut new_child_validity = BitmapBuilder::new();

    let mut new_values = None;

    // Find the first element that does not have propagated nulls.
    let null_mask = !validity;
    for i in null_mask.true_idx_iter() {
        last_idx = i;
        let (start, end) = arr.offsets().start_end(i);
        if end == start {
            continue;
        }

        if old_child_validity.is_none_or(|v| {
            BitMask::from_bitmap(v)
                .sliced(start, end - start)
                .set_bits()
                > 0
        }) {
            new_child_validity.subslice_extend_from_opt_validity(old_child_validity, 0, start);
            new_child_validity.extend_constant(end - start, false);
            break;
        }
    }

    if !new_child_validity.is_empty() {
        // If nulls need to be propagated, create a new validity mask for the child array.
        let null_mask = null_mask.sliced(last_idx + 1, arr.len() - last_idx - 1);

        for i in null_mask.true_idx_iter() {
            let i = i + last_idx + 1;
            let (start, end) = arr.offsets().start_end(i);
            if end == start {
                continue;
            }

            new_child_validity.subslice_extend_from_opt_validity(
                old_child_validity,
                new_child_validity.len(),
                start - new_child_validity.len(),
            );
            new_child_validity.extend_constant(end - start, false);
        }

        new_child_validity.subslice_extend_from_opt_validity(
            old_child_validity,
            new_child_validity.len(),
            arr.values().len() - new_child_validity.len(),
        );

        let new_child_validity = new_child_validity.freeze();
        new_values = Some(arr.values().with_validity(Some(new_child_validity)));
    }

    let Some(values) = new_values
        .as_ref()
        .and_then(|v| propagate_nulls(v.as_ref()))
        .or(new_values)
    else {
        // Nothing was changed. Return the original array.
        return None;
    };

    Some(ListArray::new(
        arr.dtype().clone(),
        arr.offsets().clone(),
        values,
        Some(validity.clone()),
    ))
}

pub fn propagate_nulls_fsl(arr: &FixedSizeListArray) -> Option<FixedSizeListArray> {
    let Some(validity) = arr.validity() else {
        return propagate_nulls(arr.values().as_ref())
            .map(|values| FixedSizeListArray::new(arr.dtype().clone(), arr.len(), values, None));
    };

    if arr.size() == 0 || validity.unset_bits() == 0 {
        return None;
    }

    let start_point = match arr.values().validity() {
        None => Some(validity.leading_ones()),
        Some(old_child_validity) => {
            // Find the first element that does not have propagated nulls.
            let null_mask = !validity;
            null_mask.true_idx_iter().find(|i| {
                BitMask::from_bitmap(old_child_validity)
                    .sliced(i * arr.size(), arr.size())
                    .set_bits()
                    > 0
            })
        },
    };

    let mut new_values = None;
    if let Some(start_point) = start_point {
        // Nulls need to be propagated, create a new validity mask.
        let mut new_child_validity = BitmapBuilder::with_capacity(arr.size() * arr.len());

        let mut validity = validity.clone();
        validity.slice(start_point, validity.len() - start_point);
        match arr.values().validity() {
            None => {
                new_child_validity.extend_constant(start_point * arr.size(), true);

                while !validity.is_empty() {
                    let num_zeroes = validity.take_leading_zeros();
                    new_child_validity.extend_constant(num_zeroes * arr.size(), false);

                    let num_ones = validity.take_leading_ones();
                    new_child_validity.extend_constant(num_ones * arr.size(), true);
                }
            },

            Some(old_child_validity) => {
                new_child_validity.subslice_extend_from_bitmap(
                    old_child_validity,
                    0,
                    start_point * arr.size(),
                );
                while !validity.is_empty() {
                    let num_zeroes = validity.take_leading_zeros();
                    new_child_validity.extend_constant(num_zeroes * arr.size(), false);

                    let num_ones = validity.take_leading_ones();
                    new_child_validity.subslice_extend_from_bitmap(
                        old_child_validity,
                        new_child_validity.len(),
                        num_ones * arr.size(),
                    );
                }
            },
        }

        let new_child_validity = new_child_validity.freeze();
        new_values = Some(arr.values().with_validity(Some(new_child_validity)));
    }

    let Some(values) = new_values
        .as_ref()
        .and_then(|v| propagate_nulls(v.as_ref()))
        .or(new_values)
    else {
        // Nothing was changed. Return the original array.
        return None;
    };

    // The child array was changed.
    Some(FixedSizeListArray::new(
        arr.dtype().clone(),
        arr.len(),
        values,
        Some(validity.clone()),
    ))
}

pub fn propagate_nulls_struct(arr: &StructArray) -> Option<StructArray> {
    let Some(validity) = arr.validity() else {
        let mut new_values = Vec::new();
        for (i, field_array) in arr.values().iter().enumerate() {
            if let Some(field_array) = propagate_nulls(field_array.as_ref()) {
                new_values.reserve(arr.values().len());
                new_values.extend(arr.values()[..i].iter().cloned());
                new_values.push(field_array);
                break;
            }
        }

        if new_values.is_empty() {
            return None;
        }

        new_values.extend(arr.values()[new_values.len()..].iter().map(|field_array| {
            propagate_nulls(field_array.as_ref()).unwrap_or_else(|| field_array.to_boxed())
        }));
        return Some(StructArray::new(
            arr.dtype().clone(),
            arr.len(),
            new_values,
            None,
        ));
    };

    if arr.values().is_empty() || validity.unset_bits() == 0 {
        return None;
    }

    let mut new_values = Vec::new();
    for (i, field_array) in arr.values().iter().enumerate() {
        let new_field_array = match field_array.validity() {
            None => Some(field_array.with_validity(Some(validity.clone()))),
            Some(v) => Some(field_array.with_validity(Some(v & validity))),
        };

        let Some(new_field_array) = new_field_array
            .as_ref()
            .and_then(|v| propagate_nulls(v.as_ref()))
            .or(new_field_array)
        else {
            // Nothing was changed. Return the original array.
            continue;
        };

        new_values.reserve(arr.values().len());
        new_values.extend(arr.values()[..i].iter().cloned());
        new_values.push(new_field_array);
        break;
    }

    if new_values.is_empty() {
        return None;
    }

    new_values.extend(arr.values()[new_values.len()..].iter().map(|field_array| {
        let new_field_array = match field_array.validity() {
            None => Some(field_array.with_validity(Some(validity.clone()))),
            Some(v) if v.num_intersections_with(validity) == validity.set_bits() => None,
            Some(v) => Some(field_array.with_validity(Some(v & validity))),
        };

        new_field_array
            .as_ref()
            .and_then(|v| propagate_nulls(v.as_ref()))
            .or(new_field_array)
            .unwrap_or_else(|| field_array.clone())
    }));

    Some(StructArray::new(
        arr.dtype().clone(),
        arr.len(),
        new_values,
        Some(validity.clone()),
    ))
}

#[cfg(test)]
mod tests {
    use arrow::array::proptest::array;
    use proptest::proptest;

    use crate::propagate_nulls::propagate_nulls;

    proptest! {
        #[test]
        fn test_proptest(array in array(0..100)) {
            if let Some(p_arr) = propagate_nulls(array.as_ref()) {
                proptest::prop_assert_eq!(array, p_arr);
            }
        }
    }
}
