use arrow::array::MutableBinaryViewArray;
use arrow::bitmap::Bitmap;

use super::decode_plain_generic;
use crate::parquet::error::ParquetResult;

pub fn decode(
    num_expected_values: usize,
    values: &[u8],
    target: &mut MutableBinaryViewArray<[u8]>,

    mask: &Bitmap,

    verify_utf8: bool,
) -> ParquetResult<()> {
    if mask.unset_bits() == 0 {
        return super::required::decode(
            num_expected_values,
            values,
            Some(mask.len()),
            target,
            verify_utf8,
        );
    }

    let mut mask_iter = mask.iter();
    decode_plain_generic(
        values,
        target,
        mask.set_bits(),
        || Some((true, mask_iter.next()?)),
        verify_utf8,
    )
}
