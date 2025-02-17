use super::*;

/// # Safety
/// All preconditions in [`super::horizontal_flatten_unchecked`]
pub(super) unsafe fn horizontal_flatten_unchecked(
    arrays: &[StructArray],
    widths: &[usize],
    output_height: usize,
) -> StructArray {
    // For StructArrays, we perform the flatten operation individually for every field in the struct
    // as well as on the outer validity. We then construct the result array from the individual
    // result parts.

    let dtype = arrays[0].dtype();

    let field_arrays: Vec<&[Box<dyn Array>]> = arrays
        .iter()
        .inspect(|x| debug_assert_eq!(x.dtype(), dtype))
        .map(|x| x.values())
        .collect::<Vec<_>>();

    let n_fields = field_arrays[0].len();

    let mut scratch = Vec::with_capacity(field_arrays.len());
    // Safety: We can take by index as all struct arrays have the same columns names in the same
    // order.
    // Note: `field_arrays` can be empty for 0-field structs.
    let field_arrays = (0..n_fields)
        .map(|i| {
            scratch.clear();
            scratch.extend(field_arrays.iter().map(|v| v[i].clone()));

            super::horizontal_flatten_unchecked(&scratch, widths, output_height)
        })
        .collect::<Vec<_>>();

    let validity = if arrays.iter().any(|x| x.validity().is_some()) {
        let max_height = output_height * widths.iter().fold(0usize, |a, b| a.max(*b));
        let mut shared_validity = None;

        // We need to create BooleanArrays from the Bitmaps for dispatch.
        let validities: Vec<BooleanArray> = arrays
            .iter()
            .map(|x| {
                x.validity().cloned().unwrap_or_else(|| {
                    if shared_validity.is_none() {
                        shared_validity = Some(Bitmap::new_with_value(true, max_height))
                    };
                    // We have to slice to exact length to pass an assertion.
                    shared_validity.clone().unwrap().sliced(0, x.len())
                })
            })
            .map(|x| BooleanArray::from_inner_unchecked(ArrowDataType::Boolean, x, None))
            .collect::<Vec<_>>();

        Some(
            super::horizontal_flatten_unchecked_impl_generic::<BooleanArray>(
                &validities,
                widths,
                output_height,
                &ArrowDataType::Boolean,
            )
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .values()
            .clone(),
        )
    } else {
        None
    };

    StructArray::new(
        dtype.clone(),
        if n_fields == 0 {
            output_height * widths.iter().copied().sum::<usize>()
        } else {
            debug_assert_eq!(
                field_arrays[0].len(),
                output_height * widths.iter().copied().sum::<usize>()
            );

            field_arrays[0].len()
        },
        field_arrays,
        validity,
    )
}
