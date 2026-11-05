use std::cmp::{max, min};

use polars_arrow::bitmap::MutableBitmap;
use polars_arrow::bitmap::bitmask::BitMask;

/// Bit array with a filter to speed up searching for set bits when sparse,
/// based on section 4.1 from Khayyat et al. 2015,
/// "Lightning Fast and Space Efficient Inequality Joins"
pub struct FilteredBitArray {
    bit_array: MutableBitmap,
    filter: MutableBitmap,
}

impl FilteredBitArray {
    const CHUNK_SIZE: usize = 1024;

    pub fn from_len_zeroed(len: usize) -> Self {
        Self {
            bit_array: MutableBitmap::from_len_zeroed(len),
            filter: MutableBitmap::from_len_zeroed(len.div_ceil(Self::CHUNK_SIZE)),
        }
    }

    pub unsafe fn set_bit_unchecked(&mut self, index: usize) {
        self.bit_array.set_unchecked(index, true);
        self.filter.set_unchecked(index / Self::CHUNK_SIZE, true);
    }

    pub fn on_set_bits_from(&self, start: usize, mut action: impl FnMut(usize)) {
        let len = self.bit_array.len();
        if start >= len {
            return;
        }
        let bits = BitMask::new(self.bit_array.as_slice(), 0, len);
        let start_chunk = start / Self::CHUNK_SIZE;
        let filter = BitMask::new(
            self.filter.as_slice(),
            start_chunk,
            self.filter.len() - start_chunk,
        );
        for chunk_idx in filter.true_idx_iter() {
            let chunk_idx = start_chunk + chunk_idx;
            let chunk_start = max(start, chunk_idx * Self::CHUNK_SIZE);
            let chunk_end = min((chunk_idx + 1) * Self::CHUNK_SIZE, len);
            let chunk = bits.sliced(chunk_start, chunk_end - chunk_start);
            chunk
                .true_idx_iter()
                .for_each(|i| action(chunk_start + i));
        }
    }
}
