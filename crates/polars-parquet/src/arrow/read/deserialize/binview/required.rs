use arrow::array::MutableBinaryViewArray;

use super::decode_plain_generic;
use crate::parquet::error::ParquetResult;

pub fn decode(
    num_expected_values: usize,
    values: &[u8],
    limit: Option<usize>,
    target: &mut MutableBinaryViewArray<[u8]>,

    verify_utf8: bool,
) -> ParquetResult<()> {
    let limit = limit.unwrap_or(num_expected_values);

    let mut idx = 0;
    decode_plain_generic(
        values,
        target,
        limit,
        || {
            if idx >= limit {
                return None;
            }

            idx += 1;

            Some((true, true))
        },
        verify_utf8,
    )
}
