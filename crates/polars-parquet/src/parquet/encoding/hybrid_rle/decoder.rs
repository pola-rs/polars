use polars_utils::slice::GetSaferUnchecked;

use super::super::{ceil8, uleb128};
use super::HybridEncoded;

/// An [`Iterator`] of [`HybridEncoded`].
#[derive(Debug, Clone)]
pub struct Decoder<'a> {
    values: &'a [u8],
    num_bits: usize,
}

impl<'a> Decoder<'a> {
    /// Returns a new [`Decoder`]
    pub fn new(values: &'a [u8], num_bits: usize) -> Self {
        Self { values, num_bits }
    }

    /// Returns the number of bits being used by this decoder.
    #[inline]
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }
}

impl<'a> Iterator for Decoder<'a> {
    type Item = HybridEncoded<'a>;

    #[inline] // -18% improvement in bench
    fn next(&mut self) -> Option<Self::Item> {
        let (indicator, consumed) = uleb128::decode(self.values);
        self.values = unsafe { self.values.get_unchecked_release(consumed..) };

        // We want to early return if consumed == 0 OR num_bits == 0, so combine into a single branch.
        if (consumed * self.num_bits) == 0 {
            return None;
        }

        if indicator & 1 == 1 {
            // is bitpacking
            let bytes = (indicator as usize >> 1) * self.num_bits;
            let bytes = std::cmp::min(bytes, self.values.len());
            let (result, remaining) = self.values.split_at(bytes);
            self.values = remaining;
            Some(HybridEncoded::Bitpacked(result))
        } else {
            // is rle
            let run_length = indicator as usize >> 1;
            // repeated-value := value that is repeated, using a fixed-width of round-up-to-next-byte(bit-width)
            let rle_bytes = ceil8(self.num_bits);
            let (result, remaining) = self.values.split_at(rle_bytes);
            self.values = remaining;
            Some(HybridEncoded::Rle(result, run_length))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::bitpacked;
    use super::*;

    #[test]
    fn basics_1() {
        let bit_width = 1usize;
        let length = 5;
        let values = [
            2, 0, 0, 0, // length
            0b00000011, 0b00001011, // data
        ];

        let mut decoder = Decoder::new(&values[4..6], bit_width);

        let run = decoder.next().unwrap();

        if let HybridEncoded::Bitpacked(values) = run {
            assert_eq!(values, &[0b00001011]);
            let result = bitpacked::Decoder::<u32>::try_new(values, bit_width, length)
                .unwrap()
                .collect::<Vec<_>>();
            assert_eq!(result, &[1, 1, 0, 1, 0]);
        } else {
            panic!()
        };
    }

    #[test]
    fn basics_2() {
        // This test was validated by the result of what pyarrow3 outputs when
        // the bitmap is used.
        let bit_width = 1;
        let values = [
            3, 0, 0, 0, // length
            0b00000101, 0b11101011, 0b00000010, // data
        ];
        let expected = &[1, 1, 0, 1, 0, 1, 1, 1, 0, 1];

        let mut decoder = Decoder::new(&values[4..4 + 3], bit_width);

        let run = decoder.next().unwrap();

        if let HybridEncoded::Bitpacked(values) = run {
            assert_eq!(values, &[0b11101011, 0b00000010]);
            let result = bitpacked::Decoder::<u32>::try_new(values, bit_width, 10)
                .unwrap()
                .collect::<Vec<_>>();
            assert_eq!(result, expected);
        } else {
            panic!()
        };
    }

    #[test]
    fn basics_3() {
        let bit_width = 1;
        let length = 8;
        let values = [
            2, 0, 0, 0,          // length
            0b00010000, // data
            0b00000001,
        ];

        let mut decoder = Decoder::new(&values[4..4 + 2], bit_width);

        let run = decoder.next().unwrap();

        if let HybridEncoded::Rle(values, items) = run {
            assert_eq!(values, &[0b00000001]);
            assert_eq!(items, length);
        } else {
            panic!()
        };
    }
}
