use super::super::delta_bitpacked;
use crate::parquet::encoding::delta_bitpacked::SumGatherer;
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

    pub(crate) offset: usize,
    pub(crate) last: Vec<u8>,
}

impl<'a> Decoder<'a> {
    pub fn try_new(values: &'a [u8]) -> ParquetResult<Self> {
        let (prefix_lengths, values) = delta_bitpacked::Decoder::try_new(values)?;
        let (suffix_lengths, values) = delta_bitpacked::Decoder::try_new(values)?;

        Ok(Self {
            prefix_lengths,
            suffix_lengths,
            values,

            offset: 0,
            last: Vec::with_capacity(32),
        })
    }

    pub fn values(&self) -> &'a [u8] {
        self.values
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(self.prefix_lengths.len(), self.suffix_lengths.len());
        self.prefix_lengths.len()
    }

    pub fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        let mut prefix_sum = 0usize;
        self.prefix_lengths
            .gather_n_into(&mut prefix_sum, n, &mut SumGatherer(0))?;
        let mut suffix_sum = 0usize;
        self.suffix_lengths
            .gather_n_into(&mut suffix_sum, n, &mut SumGatherer(0))?;
        self.offset += prefix_sum + suffix_sum;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl<'a> Iterator for Decoder<'a> {
        type Item = ParquetResult<Vec<u8>>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.len() == 0 {
                return None;
            }

            let mut prefix_length = vec![];
            let mut suffix_length = vec![];
            if let Err(e) = self.prefix_lengths.collect_n(&mut prefix_length, 1) {
                return Some(Err(e));
            }
            if let Err(e) = self.suffix_lengths.collect_n(&mut suffix_length, 1) {
                return Some(Err(e));
            }
            let prefix_length = prefix_length[0];
            let suffix_length = suffix_length[0];

            let prefix_length = prefix_length as usize;
            let suffix_length = suffix_length as usize;

            let mut value = Vec::with_capacity(prefix_length + suffix_length);

            value.extend_from_slice(&self.last[..prefix_length]);
            value.extend_from_slice(&self.values[self.offset..self.offset + suffix_length]);

            self.last.clear();
            self.last.extend_from_slice(&value);

            self.offset += suffix_length;

            Some(Ok(value))
        }
    }

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
        let values = decoder.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(values, vec![b"Hello".to_vec(), b"World".to_vec()]);

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

        let decoder = Decoder::try_new(data)?;
        let prefixes = decoder.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(prefixes, vec![b"Hello".to_vec(), b"Helicopter".to_vec()]);

        Ok(())
    }
}
