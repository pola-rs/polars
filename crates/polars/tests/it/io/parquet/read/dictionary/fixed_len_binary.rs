use polars_parquet::parquet::error::{Error, Result};

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
    pub fn value(&self, index: usize) -> Result<&[u8]> {
        self.values
            .get(index * self.size..(index + 1) * self.size)
            .ok_or_else(|| {
                Error::OutOfSpec(
                    "The data page has an index larger than the dictionary page values".to_string(),
                )
            })
    }
}

pub fn read(buf: &[u8], size: usize, num_values: usize) -> Result<FixedLenByteArrayPageDict> {
    let length = size.saturating_mul(num_values);
    let values = buf.get(..length).ok_or_else(|| Error::OutOfSpec("Fixed sized binary declares a number of values times size larger than the page buffer".to_string()))?.to_vec();

    Ok(FixedLenByteArrayPageDict::new(values, size))
}
