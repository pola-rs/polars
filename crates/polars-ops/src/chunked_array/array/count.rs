use polars_arrow::bitmap::utils::count_zeros;
use polars_arrow::legacy::utils::CustomIterTools;

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
    let chunks = ca.downcast_iter().map(|arr| {
        let mask = arr
            .values()
            .as_any()
            .downcast_ref::<PlBooleanArray>()
            .unwrap();
        assert_eq!(mask.null_count(), 0);
        let validity = arr.validity().map(PlBitmap::from);

        if arr.values_are_scalar() {
            let [count] = count_bits_set(mask.values(), 1, arr.width())[..] else {
                unreachable!("one element was counted over")
            };
            return PlPrimitiveArray::new_scalar(count, arr.len()).with_validity(validity);
        }

        let out = count_bits_set(mask.values(), arr.len(), arr.width());
        PlPrimitiveArray::from_vec(out).with_validity(validity)
    });
    IdxCa::from_chunk_iter(ca.name().clone(), chunks)
}

fn count_bits_set(values: PlBitmapRef<'_>, len: usize, width: usize) -> Vec<IdxSize> {
    if values.unset_bits() == values.len() {
        return vec![0 as IdxSize; len];
    } else if values.unset_bits() == 0 {
        return vec![width as IdxSize; len];
    }

    let values = values.to_flat();
    let (bits, bitmap_offset, _) = values.as_slice();

    (0..len)
        .map(|i| {
            let set_ones = width - count_zeros(bits, bitmap_offset + i * width, width);
            set_ones as IdxSize
        })
        .collect_trusted()
}
