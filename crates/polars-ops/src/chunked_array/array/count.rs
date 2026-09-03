use arrow::bitmap::utils::count_zeros;
use arrow::legacy::utils::CustomIterTools;
use polars_core::prelude::arity::unary_elementwise_mut_with_options_flat;

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
    unary_elementwise_mut_with_options_flat(ca, |arr| {
        let mask = arr
            .values()
            .as_any()
            .downcast_ref::<PlBooleanArray>()
            .unwrap();
        assert_eq!(mask.null_count(), 0);
        let out = count_bits_set(mask.values(), arr.len(), arr.width());
        // One count per element, and the mask of the array holds one bit per element as well.
        PlPrimitiveArray::from_vec(out).with_validity(arr.validity().cloned())
    })
}

fn count_bits_set(values: PlBitmapRef<'_>, len: usize, width: usize) -> Vec<IdxSize> {
    // Fast path where all bits are either set or unset, which is every scalar values buffer: the
    // one bit it holds stands for every value, and settles every count without being written out.
    if values.unset_bits() == values.len() {
        return vec![0 as IdxSize; len];
    } else if values.unset_bits() == 0 {
        return vec![width as IdxSize; len];
    }

    // A scalar mask is all set or all unset, so what is left here already holds one bit per
    // value: this borrows it rather than writing anything out.
    let values = values.to_flat();
    let (bits, bitmap_offset, _) = values.as_slice();

    (0..len)
        .map(|i| {
            let set_ones = width - count_zeros(bits, bitmap_offset + i * width, width);
            set_ones as IdxSize
        })
        .collect_trusted()
}
