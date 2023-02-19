use arrow::array::{Array, BooleanArray};
use arrow::bitmap::utils::count_zeros;
use arrow::bitmap::Bitmap;
use polars_arrow::utils::CustomIterTools;
use polars_core::utils::NoNull;

use super::*;

fn count_bits_set_by_offsets(values: &Bitmap, offset: &[i64]) -> IdxCa {
    let (bits, bitmap_offset, _) = values.as_slice();

    let mut running_offset = offset[0];

    let ca: NoNull<IdxCa> = (offset[1..])
        .iter()
        .map(|end| {
            let current_offset = running_offset;
            running_offset = *end;

            let len = (end - current_offset) as usize;

            let set_ones = len - count_zeros(bits, bitmap_offset + current_offset as usize, len);
            set_ones as IdxSize
        })
        .collect_trusted();

    ca.into_inner()
}

#[cfg(feature = "list_count")]
pub fn list_count_match(ca: &ListChunked, value: AnyValue) -> PolarsResult<Series> {
    let value = Series::new("", [value]);

    let ca = ca.apply_to_inner(&|s| {
        ChunkCompare::<&Series>::equal(&s, &value).map(|ca| ca.into_series())
    })?;
    let out = count_boolean_bits(&ca);
    Ok(out.into_series())
}

pub(super) fn count_boolean_bits(ca: &ListChunked) -> IdxCa {
    assert_eq!(ca.chunks().len(), 1);
    let arr = ca.downcast_iter().next().unwrap();
    let inner_arr = arr.values();
    let mask = inner_arr.as_any().downcast_ref::<BooleanArray>().unwrap();
    assert_eq!(mask.null_count(), 0);
    let mut out = count_bits_set_by_offsets(mask.values(), arr.offsets().as_slice());
    out.rename(ca.name());
    out
}
