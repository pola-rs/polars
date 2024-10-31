use super::super::delta_bitpacked;
use crate::parquet::encoding::delta_bitpacked::SumGatherer;
use crate::parquet::error::ParquetResult;

/// Decodes [Delta-length byte array](https://github.com/apache/parquet-format/blob/master/Encodings.md#delta-length-byte-array-delta_length_byte_array--6)
/// lengths and values.
/// # Implementation
/// This struct does not allocate on the heap.
#[derive(Debug)]
pub(crate) struct Decoder<'a> {
    pub(crate) lengths: delta_bitpacked::Decoder<'a>,
    pub(crate) values: &'a [u8],
    pub(crate) offset: usize,
}

impl<'a> Decoder<'a> {
    pub fn try_new(values: &'a [u8]) -> ParquetResult<Self> {
        let (lengths, values) = delta_bitpacked::Decoder::try_new(values)?;
        Ok(Self {
            lengths,
            values,
            offset: 0,
        })
    }

    pub(crate) fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        let mut sum = 0usize;
        self.lengths
            .gather_n_into(&mut sum, n, &mut SumGatherer(0))?;
        self.offset += sum;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.lengths.len()
    }
}

#[cfg(test)]
impl<'a> Iterator for Decoder<'a> {
    type Item = ParquetResult<&'a [u8]>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.lengths.len() == 0 {
            return None;
        }

        let mut length = vec![];
        if let Err(e) = self.lengths.collect_n(&mut length, 1) {
            return Some(Err(e));
        }
        let length = length[0] as usize;
        let value = &self.values[self.offset..self.offset + length];
        self.offset += length;
        Some(Ok(value))
    }
}
