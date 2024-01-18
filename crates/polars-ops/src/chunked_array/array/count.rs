use arrow::array::{Array, BooleanArray};
use arrow::bitmap::utils::count_zeros;
use arrow::bitmap::Bitmap;
use arrow::legacy::utils::CustomIterTools;
use polars_core::prelude::arity::unary_mut_with_options;

use super::*;

#[cfg(feature = "array_count")]
pub fn array_count_matches(ca: &ArrayChunked, value: AnyValue) -> PolarsResult<Series> {
    let value = Series::new("", [value]);

    let ca = ca.apply_to_inner(&|s| {
        ChunkCompare::<&Series>::equal_missing(&s, &value).map(|ca| ca.into_series())
    })?;
    let out = count_boolean_bits(&ca);
    Ok(out.into_series())
}

pub(super) fn count_boolean_bits(ca: &ArrayChunked) -> IdxCa {
    unary_mut_with_options(ca, |arr| {
        let inner_arr = arr.values();
        let mask = inner_arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(mask.null_count(), 0);
        let out = count_bits_set(mask.values(), arr.len(), arr.size());
        IdxArr::from_data_default(out.into(), arr.validity().cloned())
    })
}

fn count_bits_set(values: &Bitmap, len: usize, width: usize) -> Vec<IdxSize> {
    // Fast path where all bits are either set or unset.
    if values.unset_bits() == values.len() {
        return vec![0 as IdxSize; len];
    } else if values.unset_bits() == 0 {
        return vec![width as IdxSize; len];
    }

    let (bits, bitmap_offset, _) = values.as_slice();

    (0..len)
        .map(|i| {
            let set_ones = width - count_zeros(bits, bitmap_offset + i * width, width);
            set_ones as IdxSize
        })
        .collect_trusted()
}
