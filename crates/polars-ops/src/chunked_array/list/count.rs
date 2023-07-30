use arrow::array::{Array, BooleanArray};
use arrow::bitmap::utils::count_zeros;
use arrow::bitmap::Bitmap;
use polars_arrow::utils::CustomIterTools;

use super::*;

fn count_bits_set_by_offsets(values: &Bitmap, offset: &[i64]) -> Vec<IdxSize> {
    let (bits, bitmap_offset, _) = values.as_slice();

    let mut running_offset = offset[0];

    (offset[1..])
        .iter()
        .map(|end| {
            let current_offset = running_offset;
            running_offset = *end;

            let len = (end - current_offset) as usize;

            let set_ones = len - count_zeros(bits, bitmap_offset + current_offset as usize, len);
            set_ones as IdxSize
        })
        .collect_trusted()
}

#[cfg(feature = "list_count")]
pub fn list_count_match(ca: &ListChunked, value: AnyValue) -> PolarsResult<Series> {
    let value = Series::new("", [value]);

    let ca = ca.apply_to_inner(&|s| {
        ChunkCompare::<&Series>::equal_missing(&s, &value).map(|ca| ca.into_series())
    })?;
    let out = count_boolean_bits(&ca);
    Ok(out.into_series())
}

pub(super) fn count_boolean_bits(ca: &ListChunked) -> IdxCa {
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let inner_arr = arr.values();
            let mask = inner_arr.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert_eq!(mask.null_count(), 0);
            let out = count_bits_set_by_offsets(mask.values(), arr.offsets().as_slice());
            Box::new(IdxArr::from_data_default(
                out.into(),
                arr.validity().cloned(),
            )) as ArrayRef
        })
        .collect();
    unsafe { IdxCa::from_chunks(ca.name(), chunks) }
}
