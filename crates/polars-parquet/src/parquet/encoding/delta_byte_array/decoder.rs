use super::super::delta_bitpacked;
use crate::parquet::error::ParquetResult;

/// Decodes according to [Delta strings](https://github.com/apache/parquet-format/blob/master/Encodings.md#delta-strings-delta_byte_array--7),
/// prefixes, lengths and values
/// # Implementation
/// This struct does not allocate on the heap.
#[derive(Debug)]
pub struct Decoder<'a> {
    pub(crate) prefix_lengths: delta_bitpacked::Decoder<'a>,
    pub(crate) suffix_lengths: delta_bitpacked::Decoder<'a>,
    pub(crate) values: &'a [u8],
}

impl<'a> Decoder<'a> {
    pub fn try_new(values: &'a [u8]) -> ParquetResult<Self> {
        let (prefix_lengths, values) = delta_bitpacked::Decoder::try_new(values)?;
        let (suffix_lengths, values) = delta_bitpacked::Decoder::try_new(values)?;

        Ok(Self {
            prefix_lengths,
            suffix_lengths,
            values,
        })
    }

    pub fn values(&self) -> &'a [u8] {
        self.values
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(self.prefix_lengths.len(), self.suffix_lengths.len());
        self.prefix_lengths.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bla() -> ParquetResult<()> {
        // VALIDATED from spark==3.1.1
        let data = &[
            128, 1, 4, 2, 0, 0, 0, 0, 0, 0, 128, 1, 4, 2, 10, 0, 0, 0, 0, 0, 72, 101, 108, 108,
            111, 87, 111, 114, 108, 100,
            // extra bytes are not from spark, but they should be ignored by the decoder
            // because they are beyond the sum of all lengths.
            1, 2, 3,
        ];

        let decoder = Decoder::try_new(data)?;
        assert_eq!(decoder.prefix_lengths.collect::<Vec<i64>>()?, [0, 0]);
        assert_eq!(decoder.suffix_lengths.collect::<Vec<i64>>()?, [5, 5]);
        assert_eq!(decoder.values, b"HelloWorld\x01\x02\x03");

        Ok(())
    }

    #[test]
    fn test_with_prefix() -> ParquetResult<()> {
        // VALIDATED from spark==3.1.1
        let data = &[
            128, 1, 4, 2, 0, 6, 0, 0, 0, 0, 128, 1, 4, 2, 10, 4, 0, 0, 0, 0, 72, 101, 108, 108,
            111, 105, 99, 111, 112, 116, 101, 114,
            // extra bytes are not from spark, but they should be ignored by the decoder
            // because they are beyond the sum of all lengths.
            1, 2, 3,
        ];

        // "Hello", then "Helicopter" as the first 3 bytes of "Hello" and "icopter".
        let decoder = Decoder::try_new(data)?;
        assert_eq!(decoder.prefix_lengths.collect::<Vec<i64>>()?, [0, 3]);
        assert_eq!(decoder.suffix_lengths.collect::<Vec<i64>>()?, [5, 7]);
        assert_eq!(decoder.values, b"Helloicopter\x01\x02\x03");

        Ok(())
    }
}
