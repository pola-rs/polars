use super::super::delta_bitpacked;
use crate::parquet::error::ParquetError;

/// Decodes [Delta-length byte array](https://github.com/apache/parquet-format/blob/master/Encodings.md#delta-length-byte-array-delta_length_byte_array--6)
/// lengths and values.
/// # Implementation
/// This struct does not allocate on the heap.
#[derive(Debug)]
pub struct Decoder<'a> {
    pub(crate) lengths: delta_bitpacked::Decoder<'a>,
    pub(crate) values: &'a [u8],
    pub(crate) offset: usize,
}

impl<'a> Decoder<'a> {
    pub fn try_new(values: &'a [u8]) -> Result<Self, ParquetError> {
        let (lengths, values) = delta_bitpacked::Decoder::try_new(values)?;
        Ok(Self {
            lengths,
            values,
            offset: 0,
        })
    }

    /// Returns the slice of concatenated values.
    pub fn lengths(&mut self) -> &mut delta_bitpacked::Decoder<'a> {
        &mut self.lengths
    }

    /// Returns the slice of concatenated values.
    pub fn values(&self) -> &'a [u8] {
        self.values
    }

    pub fn len(&self) -> usize {
        self.lengths.len()
    }
}

impl<'a> Iterator for Decoder<'a> {
    type Item = Result<&'a [u8], ParquetError>;

    fn next(&mut self) -> Option<Self::Item> {
        let length = self.lengths.next()?;
        Some(match length {
            Ok(length) => {
                let length = length as usize;
                let value = &self.values[self.offset..self.offset + length];
                self.offset += length;
                Ok(value)
            },
            Err(error) => Err(error),
        })
    }
}
