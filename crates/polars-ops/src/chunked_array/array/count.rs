use arrow::bitmap::Bitmap;
use arrow::bitmap::utils::count_zeros;
use arrow::legacy::utils::CustomIterTools;
use polars_core::prelude::arity::unary_mut_with_options_flat;

use super::*;

#[cfg(feature = "array_count")]
pub fn array_count_matches(ca: &ArrayChunked, value: AnyValue) -> PolarsResult<Series> {
    let value = Series::new(PlSmallStr::EMPTY, [value]);

    let ca = ca.apply_to_inner(&|s| {
        ChunkCompareEq::<&Series>::equal_missing(&s, &value).map(|ca| ca.into_series())
    })?;
    let out = count_boolean_bits(&ca);
    Ok(out.into_series())
}

pub(super) fn count_boolean_bits(ca: &ArrayChunked) -> IdxCa {
    unary_mut_with_options_flat(ca, |arr| {
        let mask = arr
            .values()
            .as_any()
            .downcast_ref::<PlBooleanArray>()
            .unwrap();
        assert_eq!(mask.null_count(), 0);
        // TODO(polars-array-scalar): the bits are counted over a flat bitmap, so a scalar values
        // buffer is written out here rather than the one bit it stands for being counted once.
        let mask = mask.to_flat();
        let out = count_bits_set(mask.values(), arr.len(), arr.width());
        // One count per element, and the mask of the array holds one bit per element as well.
        PlPrimitiveArray::from_vec(out).with_validity(arr.validity().cloned())
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
