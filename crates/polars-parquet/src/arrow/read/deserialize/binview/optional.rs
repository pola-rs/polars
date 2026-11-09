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
    page_validity: &Bitmap,

    verify_utf8: bool,
) -> ParquetResult<()> {
    if page_validity.unset_bits() == 0 {
        return super::required::decode(
            num_expected_values,
            values,
            page_buffer,
            Some(page_validity.len()),
            target,
            verify_utf8,
        );
    }

    let mut validity_iter = page_validity.iter();
    decode_plain_generic(
        values,
        page_buffer,
        target,
        page_validity.len(),
        || Some((validity_iter.next()?, true)),
        verify_utf8,
    )
}
