use super::BinaryIter;
use crate::parquet::error::ParquetResult;

pub fn decode_plain(
    values: &[u8],
    max_num_values: usize,
    target: &mut Vec<u8>,
    offsets: &mut Vec<i64>,
) -> ParquetResult<()> {
    assert!(target.is_empty());
    assert!(offsets.is_empty());

    offsets.reserve(max_num_values + 1);
    offsets.push(0);

    // First, get the summed length so we can reserve once.
    let mut total_buffer_size = 0;
    let mut offset = 0;
    offsets.extend(BinaryIter::new(values, max_num_values).map(|value| {
        total_buffer_size += value.len();
        offset += value.len() as i64;
        offset
    }));

    // Second, fill in all the data.
    target.reserve(total_buffer_size);
    for value in BinaryIter::new(values, max_num_values) {
        target.extend_from_slice(value);
    }

    Ok(())
}
