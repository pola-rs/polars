use polars_parquet::parquet::encoding::get_length;
use polars_parquet::parquet::error::ParquetError;

#[derive(Debug)]
pub struct BinaryPageDict {
    values: Vec<Vec<u8>>,
}

impl BinaryPageDict {
    pub fn new(values: Vec<Vec<u8>>) -> Self {
        Self { values }
    }

    #[inline]
    pub fn value(&self, index: usize) -> Result<&[u8], ParquetError> {
        self.values
            .get(index)
            .map(|x| x.as_ref())
            .ok_or_else(|| ParquetError::OutOfSpec("invalid index".to_string()))
    }
}

fn read_plain(bytes: &[u8], length: usize) -> Result<Vec<Vec<u8>>, ParquetError> {
    let mut bytes = bytes;
    let mut values = Vec::new();

    for _ in 0..length {
        let slot_length = get_length(bytes).unwrap();
        bytes = &bytes[4..];

        if slot_length > bytes.len() {
            return Err(ParquetError::OutOfSpec(
                "The string on a dictionary page has a length that is out of bounds".to_string(),
            ));
        }
        let (result, remaining) = bytes.split_at(slot_length);

        values.push(result.to_vec());
        bytes = remaining;
    }

    Ok(values)
}

pub fn read(buf: &[u8], num_values: usize) -> Result<BinaryPageDict, ParquetError> {
    let values = read_plain(buf, num_values)?;
    Ok(BinaryPageDict::new(values))
}
