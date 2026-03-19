use polars_parquet::parquet::error::{ParquetError, ParquetResult};

#[derive(Debug)]
pub struct FixedLenByteArrayPageDict {
    values: Vec<u8>,
    size: usize,
}

impl FixedLenByteArrayPageDict {
    pub fn new(values: Vec<u8>, size: usize) -> Self {
        Self { values, size }
    }

    #[inline]
    pub fn value(&self, index: usize) -> ParquetResult<&[u8]> {
        self.values
            .get(index * self.size..(index + 1) * self.size)
            .ok_or_else(|| {
                ParquetError::OutOfSpec(
                    "The data page has an index larger than the dictionary page values".to_string(),
                )
            })
    }
}

pub fn read(
    buf: &[u8],
    size: usize,
    num_values: usize,
) -> ParquetResult<FixedLenByteArrayPageDict> {
    let length = size.saturating_mul(num_values);
    let values = buf.get(..length).ok_or_else(|| ParquetError::OutOfSpec("Fixed sized binary declares a number of values times size larger than the page buffer".to_string()))?.to_vec();

    Ok(FixedLenByteArrayPageDict::new(values, size))
}
