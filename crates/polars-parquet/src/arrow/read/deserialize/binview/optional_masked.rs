use polars_arrow::array::MutableBinaryViewArray;
use polars_arrow::bitmap::Bitmap;

use crate::parquet::error::ParquetResult;
use crate::parquet::page::SharedPageBuffer;
use crate::read::deserialize::binview::decode_plain_generic;

pub fn decode(
    num_expected_values: usize,
    values: &[u8],
    page_buffer: Option<SharedPageBuffer<'_>>,
    target: &mut MutableBinaryViewArray<[u8]>,

    page_validity: &Bitmap,
    mask: &Bitmap,

    verify_utf8: bool,
) -> ParquetResult<()> {
    assert_eq!(page_validity.len(), mask.len());

    if mask.unset_bits() == 0 {
        return super::optional::decode(
            num_expected_values,
            values,
            page_buffer,
            target,
            page_validity,
            verify_utf8,
        );
    }
    if page_validity.unset_bits() == 0 {
        return super::required_masked::decode(
            num_expected_values,
            values,
            page_buffer,
            target,
            mask,
            verify_utf8,
        );
    }

    let mut validity_iter = page_validity.iter();
    let mut mask_iter = mask.iter();
    let page_buffer = page_buffer.filter(|_| mask.unset_bits() <= mask.len() / 4);
    decode_plain_generic(
        values,
        page_buffer,
        target,
        mask.set_bits(),
        || Some((validity_iter.next()?, mask_iter.next()?)),
        verify_utf8,
    )
}
