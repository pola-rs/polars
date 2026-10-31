use polars_arrow::array::MutableBinaryViewArray;
use polars_arrow::bitmap::Bitmap;

use super::decode_plain_generic;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::SharedPageBuffer;

pub fn decode(
    num_expected_values: usize,
    values: &[u8],
    page_buffer: Option<SharedPageBuffer<'_>>,
    target: &mut MutableBinaryViewArray<[u8]>,

    mask: &Bitmap,

    verify_utf8: bool,
) -> ParquetResult<()> {
    if mask.unset_bits() == 0 {
        return super::required::decode(
            num_expected_values,
            values,
            page_buffer,
            Some(mask.len()),
            target,
            verify_utf8,
        );
    }

    let mut mask_iter = mask.iter();
    let page_buffer = page_buffer.filter(|_| mask.unset_bits() <= mask.len() / 4);
    decode_plain_generic(
        values,
        page_buffer,
        target,
        mask.set_bits(),
        || Some((true, mask_iter.next()?)),
        verify_utf8,
    )
}
