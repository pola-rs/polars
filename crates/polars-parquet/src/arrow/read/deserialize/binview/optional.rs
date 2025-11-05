use arrow::array::MutableBinaryViewArray;
use arrow::bitmap::Bitmap;

use super::decode_plain_generic;
use crate::parquet::error::ParquetResult;

pub fn decode(
    num_expected_values: usize,
    values: &[u8],
    target: &mut MutableBinaryViewArray<[u8]>,
    page_validity: &Bitmap,

    verify_utf8: bool,
) -> ParquetResult<()> {
    if page_validity.unset_bits() == 0 {
        return super::required::decode(
            num_expected_values,
            values,
            Some(page_validity.len()),
            target,
            verify_utf8,
        );
    }

    let mut validity_iter = page_validity.iter();
    decode_plain_generic(
        values,
        target,
        page_validity.len(),
        || Some((validity_iter.next()?, true)),
        verify_utf8,
    )
}
