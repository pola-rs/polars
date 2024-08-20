use std::cmp::min;

use arrow::bitmap::MutableBitmap;

/// Bit array with a filter to speed up searching for set bits,
/// based on section 4.1 in Khayyat et al.
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

    pub fn set_bit(&mut self, index: usize) {
        self.bit_array.set(index, true);
        self.filter.set(index / Self::CHUNK_SIZE, true);
    }

    pub fn on_set_bits_from<F>(&self, start: usize, mut action: F)
    where
        F: FnMut(usize),
    {
        let start_chunk = start / Self::CHUNK_SIZE;
        let mut chunk_offset = start % Self::CHUNK_SIZE;
        for chunk_idx in start_chunk..self.filter.len() {
            if self.filter.get(chunk_idx) {
                // There are some set bits in this chunk
                let start = chunk_idx * Self::CHUNK_SIZE + chunk_offset;
                let end = min((chunk_idx + 1) * Self::CHUNK_SIZE, self.bit_array.len());
                for bit_idx in start..end {
                    // SAFETY: `bit_idx` is always less than `self.bit_array.len()`
                    if unsafe { self.bit_array.get_unchecked(bit_idx) } {
                        action(bit_idx);
                    }
                }
            }
            chunk_offset = 0;
        }
    }
}
