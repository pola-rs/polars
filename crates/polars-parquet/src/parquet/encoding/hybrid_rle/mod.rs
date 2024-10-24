// See https://github.com/apache/parquet-format/blob/master/Encodings.md#run-length-encoding--bit-packing-hybrid-rle--3
mod bitmap;
mod encoder;

pub use bitmap::{encode_bool as bitpacked_encode, BitmapIter};
pub use encoder::{encode, Encoder};
use polars_utils::slice::GetSaferUnchecked;

use super::{bitpacked, uleb128};
use crate::parquet::error::{ParquetError, ParquetResult};

/// A [`Iterator`] for Hybrid Run-Length Encoding
///
/// The hybrid here means that each second is prepended by a bit that differentiates between two
/// modes.
///
/// 1. Run-Length Encoding in the shape of `[Number of Values, Value]`
/// 2. Bitpacking in the shape of `[Value 1 in n bits, Value 2 in n bits, ...]`
///
/// Note, that this can iterate, but the set of `collect_*` and `translate_and_collect_*` methods
/// should be highly preferred as they are way more efficient and have better error handling.
#[derive(Debug, Clone)]
pub struct HybridRleDecoder<'a> {
    data: &'a [u8],
    num_bits: usize,
    num_values: usize,
}

pub struct HybridRleChunkIter<'a> {
    decoder: HybridRleDecoder<'a>,
}

#[derive(Debug)]
pub enum HybridRleChunk<'a> {
    Rle(u32, usize),
    Bitpacked(bitpacked::Decoder<'a, u32>),
}

impl<'a> Iterator for HybridRleChunkIter<'a> {
    type Item = ParquetResult<HybridRleChunk<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.decoder.next_chunk().transpose()
    }
}

impl<'a> HybridRleDecoder<'a> {
    /// Returns a new [`HybridRleDecoder`]
    pub fn new(data: &'a [u8], num_bits: u32, num_values: usize) -> Self {
        Self {
            data,
            num_bits: num_bits as usize,
            num_values,
        }
    }

    pub fn len(&self) -> usize {
        self.num_values
    }

    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    pub fn into_chunk_iter(self) -> HybridRleChunkIter<'a> {
        HybridRleChunkIter { decoder: self }
    }

    pub fn next_chunk(&mut self) -> ParquetResult<Option<HybridRleChunk<'a>>> {
        if self.len() == 0 {
            return Ok(None);
        }

        if self.num_bits == 0 {
            let num_values = self.num_values;
            self.num_values = 0;
            return Ok(Some(HybridRleChunk::Rle(0, num_values)));
        }

        if self.data.is_empty() {
            return Ok(None);
        }

        let (indicator, consumed) = uleb128::decode(self.data);
        self.data = unsafe { self.data.get_unchecked_release(consumed..) };

        Ok(Some(if indicator & 1 == 1 {
            // is bitpacking
            let bytes = (indicator as usize >> 1) * self.num_bits;
            let bytes = std::cmp::min(bytes, self.data.len());
            let Some((packed, remaining)) = self.data.split_at_checked(bytes) else {
                return Err(ParquetError::oos("Not enough bytes for bitpacked data"));
            };
            self.data = remaining;

            let length = std::cmp::min(packed.len() * 8 / self.num_bits, self.num_values);
            let decoder = bitpacked::Decoder::<u32>::try_new(packed, self.num_bits, length)?;

            self.num_values -= length;

            HybridRleChunk::Bitpacked(decoder)
        } else {
            // is rle
            let run_length = indicator as usize >> 1;
            // repeated-value := value that is repeated, using a fixed-width of round-up-to-next-byte(bit-width)
            let rle_bytes = self.num_bits.div_ceil(8);
            let Some((pack, remaining)) = self.data.split_at_checked(rle_bytes) else {
                return Err(ParquetError::oos("Not enough bytes for RLE encoded data"));
            };
            self.data = remaining;

            let mut bytes = [0u8; std::mem::size_of::<u32>()];
            pack.iter().zip(bytes.iter_mut()).for_each(|(src, dst)| {
                *dst = *src;
            });
            let value = u32::from_le_bytes(bytes);

            let length = std::cmp::min(run_length, self.num_values);

            self.num_values -= length;

            HybridRleChunk::Rle(value, length)
        }))
    }

    pub fn limit_to(&mut self, length: usize) {
        self.num_values = self.num_values.min(length);
    }

    pub fn collect(self) -> ParquetResult<Vec<u32>> {
        let mut target = Vec::with_capacity(self.len());

        for chunk in self.into_chunk_iter() {
            match chunk? {
                HybridRleChunk::Rle(value, size) => {
                    target.resize(target.len() + size, value);
                },
                HybridRleChunk::Bitpacked(decoder) => {
                    decoder.collect_into(&mut target);
                },
            }
        }

        Ok(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() -> ParquetResult<()> {
        let mut buffer = vec![];
        let num_bits = 10u32;

        let data = (0..1000).collect::<Vec<_>>();

        encode::<u32, _, _>(&mut buffer, data.iter().cloned(), num_bits).unwrap();

        let decoder = HybridRleDecoder::new(&buffer, num_bits, data.len());

        let result = decoder.collect()?;

        assert_eq!(result, data);
        Ok(())
    }

    #[test]
    fn pyarrow_integration() -> ParquetResult<()> {
        // data encoded from pyarrow representing (0..1000)
        let data = vec![
            127, 0, 4, 32, 192, 0, 4, 20, 96, 192, 1, 8, 36, 160, 192, 2, 12, 52, 224, 192, 3, 16,
            68, 32, 193, 4, 20, 84, 96, 193, 5, 24, 100, 160, 193, 6, 28, 116, 224, 193, 7, 32,
            132, 32, 194, 8, 36, 148, 96, 194, 9, 40, 164, 160, 194, 10, 44, 180, 224, 194, 11, 48,
            196, 32, 195, 12, 52, 212, 96, 195, 13, 56, 228, 160, 195, 14, 60, 244, 224, 195, 15,
            64, 4, 33, 196, 16, 68, 20, 97, 196, 17, 72, 36, 161, 196, 18, 76, 52, 225, 196, 19,
            80, 68, 33, 197, 20, 84, 84, 97, 197, 21, 88, 100, 161, 197, 22, 92, 116, 225, 197, 23,
            96, 132, 33, 198, 24, 100, 148, 97, 198, 25, 104, 164, 161, 198, 26, 108, 180, 225,
            198, 27, 112, 196, 33, 199, 28, 116, 212, 97, 199, 29, 120, 228, 161, 199, 30, 124,
            244, 225, 199, 31, 128, 4, 34, 200, 32, 132, 20, 98, 200, 33, 136, 36, 162, 200, 34,
            140, 52, 226, 200, 35, 144, 68, 34, 201, 36, 148, 84, 98, 201, 37, 152, 100, 162, 201,
            38, 156, 116, 226, 201, 39, 160, 132, 34, 202, 40, 164, 148, 98, 202, 41, 168, 164,
            162, 202, 42, 172, 180, 226, 202, 43, 176, 196, 34, 203, 44, 180, 212, 98, 203, 45,
            184, 228, 162, 203, 46, 188, 244, 226, 203, 47, 192, 4, 35, 204, 48, 196, 20, 99, 204,
            49, 200, 36, 163, 204, 50, 204, 52, 227, 204, 51, 208, 68, 35, 205, 52, 212, 84, 99,
            205, 53, 216, 100, 163, 205, 54, 220, 116, 227, 205, 55, 224, 132, 35, 206, 56, 228,
            148, 99, 206, 57, 232, 164, 163, 206, 58, 236, 180, 227, 206, 59, 240, 196, 35, 207,
            60, 244, 212, 99, 207, 61, 248, 228, 163, 207, 62, 252, 244, 227, 207, 63, 0, 5, 36,
            208, 64, 4, 21, 100, 208, 65, 8, 37, 164, 208, 66, 12, 53, 228, 208, 67, 16, 69, 36,
            209, 68, 20, 85, 100, 209, 69, 24, 101, 164, 209, 70, 28, 117, 228, 209, 71, 32, 133,
            36, 210, 72, 36, 149, 100, 210, 73, 40, 165, 164, 210, 74, 44, 181, 228, 210, 75, 48,
            197, 36, 211, 76, 52, 213, 100, 211, 77, 56, 229, 164, 211, 78, 60, 245, 228, 211, 79,
            64, 5, 37, 212, 80, 68, 21, 101, 212, 81, 72, 37, 165, 212, 82, 76, 53, 229, 212, 83,
            80, 69, 37, 213, 84, 84, 85, 101, 213, 85, 88, 101, 165, 213, 86, 92, 117, 229, 213,
            87, 96, 133, 37, 214, 88, 100, 149, 101, 214, 89, 104, 165, 165, 214, 90, 108, 181,
            229, 214, 91, 112, 197, 37, 215, 92, 116, 213, 101, 215, 93, 120, 229, 165, 215, 94,
            124, 245, 229, 215, 95, 128, 5, 38, 216, 96, 132, 21, 102, 216, 97, 136, 37, 166, 216,
            98, 140, 53, 230, 216, 99, 144, 69, 38, 217, 100, 148, 85, 102, 217, 101, 152, 101,
            166, 217, 102, 156, 117, 230, 217, 103, 160, 133, 38, 218, 104, 164, 149, 102, 218,
            105, 168, 165, 166, 218, 106, 172, 181, 230, 218, 107, 176, 197, 38, 219, 108, 180,
            213, 102, 219, 109, 184, 229, 166, 219, 110, 188, 245, 230, 219, 111, 192, 5, 39, 220,
            112, 196, 21, 103, 220, 113, 200, 37, 167, 220, 114, 204, 53, 231, 220, 115, 208, 69,
            39, 221, 116, 212, 85, 103, 221, 117, 216, 101, 167, 221, 118, 220, 117, 231, 221, 119,
            224, 133, 39, 222, 120, 228, 149, 103, 222, 121, 232, 165, 167, 222, 122, 236, 181,
            231, 222, 123, 240, 197, 39, 223, 124, 244, 213, 103, 223, 125, 125, 248, 229, 167,
            223, 126, 252, 245, 231, 223, 127, 0, 6, 40, 224, 128, 4, 22, 104, 224, 129, 8, 38,
            168, 224, 130, 12, 54, 232, 224, 131, 16, 70, 40, 225, 132, 20, 86, 104, 225, 133, 24,
            102, 168, 225, 134, 28, 118, 232, 225, 135, 32, 134, 40, 226, 136, 36, 150, 104, 226,
            137, 40, 166, 168, 226, 138, 44, 182, 232, 226, 139, 48, 198, 40, 227, 140, 52, 214,
            104, 227, 141, 56, 230, 168, 227, 142, 60, 246, 232, 227, 143, 64, 6, 41, 228, 144, 68,
            22, 105, 228, 145, 72, 38, 169, 228, 146, 76, 54, 233, 228, 147, 80, 70, 41, 229, 148,
            84, 86, 105, 229, 149, 88, 102, 169, 229, 150, 92, 118, 233, 229, 151, 96, 134, 41,
            230, 152, 100, 150, 105, 230, 153, 104, 166, 169, 230, 154, 108, 182, 233, 230, 155,
            112, 198, 41, 231, 156, 116, 214, 105, 231, 157, 120, 230, 169, 231, 158, 124, 246,
            233, 231, 159, 128, 6, 42, 232, 160, 132, 22, 106, 232, 161, 136, 38, 170, 232, 162,
            140, 54, 234, 232, 163, 144, 70, 42, 233, 164, 148, 86, 106, 233, 165, 152, 102, 170,
            233, 166, 156, 118, 234, 233, 167, 160, 134, 42, 234, 168, 164, 150, 106, 234, 169,
            168, 166, 170, 234, 170, 172, 182, 234, 234, 171, 176, 198, 42, 235, 172, 180, 214,
            106, 235, 173, 184, 230, 170, 235, 174, 188, 246, 234, 235, 175, 192, 6, 43, 236, 176,
            196, 22, 107, 236, 177, 200, 38, 171, 236, 178, 204, 54, 235, 236, 179, 208, 70, 43,
            237, 180, 212, 86, 107, 237, 181, 216, 102, 171, 237, 182, 220, 118, 235, 237, 183,
            224, 134, 43, 238, 184, 228, 150, 107, 238, 185, 232, 166, 171, 238, 186, 236, 182,
            235, 238, 187, 240, 198, 43, 239, 188, 244, 214, 107, 239, 189, 248, 230, 171, 239,
            190, 252, 246, 235, 239, 191, 0, 7, 44, 240, 192, 4, 23, 108, 240, 193, 8, 39, 172,
            240, 194, 12, 55, 236, 240, 195, 16, 71, 44, 241, 196, 20, 87, 108, 241, 197, 24, 103,
            172, 241, 198, 28, 119, 236, 241, 199, 32, 135, 44, 242, 200, 36, 151, 108, 242, 201,
            40, 167, 172, 242, 202, 44, 183, 236, 242, 203, 48, 199, 44, 243, 204, 52, 215, 108,
            243, 205, 56, 231, 172, 243, 206, 60, 247, 236, 243, 207, 64, 7, 45, 244, 208, 68, 23,
            109, 244, 209, 72, 39, 173, 244, 210, 76, 55, 237, 244, 211, 80, 71, 45, 245, 212, 84,
            87, 109, 245, 213, 88, 103, 173, 245, 214, 92, 119, 237, 245, 215, 96, 135, 45, 246,
            216, 100, 151, 109, 246, 217, 104, 167, 173, 246, 218, 108, 183, 237, 246, 219, 112,
            199, 45, 247, 220, 116, 215, 109, 247, 221, 120, 231, 173, 247, 222, 124, 247, 237,
            247, 223, 128, 7, 46, 248, 224, 132, 23, 110, 248, 225, 136, 39, 174, 248, 226, 140,
            55, 238, 248, 227, 144, 71, 46, 249, 228, 148, 87, 110, 249, 229, 152, 103, 174, 249,
            230, 156, 119, 238, 249, 231, 160, 135, 46, 250, 232, 164, 151, 110, 250, 233, 168,
            167, 174, 250, 234, 172, 183, 238, 250, 235, 176, 199, 46, 251, 236, 180, 215, 110,
            251, 237, 184, 231, 174, 251, 238, 188, 247, 238, 251, 239, 192, 7, 47, 252, 240, 196,
            23, 111, 252, 241, 200, 39, 175, 252, 242, 204, 55, 239, 252, 243, 208, 71, 47, 253,
            244, 212, 87, 111, 253, 245, 216, 103, 175, 253, 246, 220, 119, 239, 253, 247, 224,
            135, 47, 254, 248, 228, 151, 111, 254, 249,
        ];
        let num_bits = 10;

        let decoder = HybridRleDecoder::new(&data, num_bits, 1000);
        let result = decoder.collect()?;

        assert_eq!(result, (0..1000).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn small() -> ParquetResult<()> {
        let data = vec![3, 2];

        let num_bits = 3;

        let decoder = HybridRleDecoder::new(&data, num_bits, 1);

        let result = decoder.collect()?;

        assert_eq!(result, &[2]);
        Ok(())
    }

    #[test]
    fn zero_bit_width() -> ParquetResult<()> {
        let data = vec![3];

        let num_bits = 0;

        let decoder = HybridRleDecoder::new(&data, num_bits, 2);

        let result = decoder.collect()?;

        assert_eq!(result, &[0, 0]);
        Ok(())
    }

    #[test]
    fn empty_values() -> ParquetResult<()> {
        let data = [];

        let num_bits = 0;

        let decoder = HybridRleDecoder::new(&data, num_bits, 100);

        let result = decoder.collect()?;

        assert_eq!(result, vec![0; 100]);
        Ok(())
    }
}
