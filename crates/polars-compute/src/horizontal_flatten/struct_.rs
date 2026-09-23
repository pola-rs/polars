use polars_array::{PlArray, PlBitmap, PlBooleanArray, PlStructArray};

use super::horizontal_flatten;
use crate::nesting::downcast;

/// Lays struct arrays out end to end, a field at a time.
pub(super) fn flatten_structs(
    arrays: &[Box<dyn PlArray>],
    widths: &[usize],
    output_height: usize,
    out_len: usize,
) -> PlStructArray {
    let mut field = Vec::with_capacity(arrays.len());
    let fields: Vec<Box<dyn PlArray>> = (0..downcast::<PlStructArray>(&*arrays[0]).num_fields())
        .map(|i| {
            field.clear();
            field.extend(
                arrays
                    .iter()
                    .map(|array| downcast::<PlStructArray>(&**array).field(i).to_boxed()),
            );
            horizontal_flatten(&field, widths, output_height)
        })
        .collect();

    let validity = arrays
        .iter()
        .any(|array| array.validity().is_some())
        .then(|| {
            let masks: Vec<Box<dyn PlArray>> = arrays
                .iter()
                .map(|array| {
                    let mask = match array.validity() {
                        None => PlBooleanArray::new_scalar(true, array.len()),
                        Some(validity) => PlBooleanArray::from_pl_bitmap(validity.into()),
                    };
                    Box::new(mask) as Box<dyn PlArray>
                })
                .collect();

            let flattened = horizontal_flatten(&masks, widths, output_height);
            downcast::<PlBooleanArray>(&*flattened)
                .values()
                .to_flat_or_scalar()
        });

    PlStructArray::new(fields, out_len, None).with_validity(validity.map(PlBitmap::from_bitmap))
}
