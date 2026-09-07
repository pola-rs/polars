use arrow::bitmap::Bitmap;
use arrow::bitmap::utils::count_zeros;
use arrow::legacy::utils::CustomIterTools;

use super::*;

fn count_bits_set_by_offsets(values: &Bitmap, offset: &[u64]) -> Vec<IdxSize> {
    // Fast path where all bits are either set or unset.
    if values.unset_bits() == values.len() {
        return vec![0 as IdxSize; offset.len() - 1];
    } else if values.unset_bits() == 0 {
        let mut start = offset[0];
        let v = (offset[1..])
            .iter()
            .map(|end| {
                let current_offset = start;
                start = *end;
                (end - current_offset) as IdxSize
            })
            .collect_trusted();
        return v;
    }

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
pub fn list_count_matches(ca: &ListChunked, value: AnyValue) -> PolarsResult<Series> {
    let value = Series::new(PlSmallStr::EMPTY, [value]);

    let ca = ca.apply_to_inner(&|s| {
        ChunkCompareEq::<&Series>::equal_missing(&s, &value).map(|ca| ca.into_series())
    })?;
    let out = count_boolean_bits(&ca);
    Ok(out.into_series())
}

pub(super) fn count_boolean_bits(ca: &ListChunked) -> IdxCa {
    let chunks = ca.downcast_iter().map(|arr| {
        let mask = arr
            .values()
            .as_any()
            .downcast_ref::<PlBooleanArray>()
            .unwrap();
        assert_eq!(mask.null_count(), 0);
        let validity = arr.validity().map(PlBitmap::from);

        // Every element covers the one range, so they all count the same: it is counted once over
        // that range rather than the lists being laid end to end first.
        if let Some(range) = arr.scalar_offsets() {
            return PlPrimitiveArray::new_scalar(count_set_over(mask, range), arr.len())
                .with_validity(validity);
        }

        let offsets = arr
            .flat_offsets()
            .expect("the elements cover ranges of their own");
        let out = match mask.scalar_values() {
            // The bits repeat a single bit, so a list counts either its whole length or nothing.
            Some(bit) => offsets
                .windows(2)
                .map(|window| match bit {
                    true => (window[1] - window[0]) as IdxSize,
                    false => 0,
                })
                .collect(),
            None => count_bits_set_by_offsets(
                mask.flat_values().expect("the bits are not repeated"),
                offsets,
            ),
        };

        // One count per element, and the mask of the array holds one bit per element as well.
        PlPrimitiveArray::from_vec(out).with_validity(validity)
    });
    IdxCa::from_chunk_iter(ca.name().clone(), chunks)
}

/// The number of set bits of `mask` over `range`, reading a scalar bitmap as the one bit it is.
fn count_set_over(mask: &PlBooleanArray, range: std::ops::Range<usize>) -> IdxSize {
    match mask.scalar_values() {
        Some(true) => range.len() as IdxSize,
        Some(false) => 0,
        None => mask
            .flat_values()
            .expect("the bits are not repeated")
            .clone()
            .sliced(range.start, range.len())
            .set_bits() as IdxSize,
    }
}
