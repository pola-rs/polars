use std::iter::Enumerate;

use arrow::array::BooleanArray;
use arrow::bitmap::utils::BitChunks;
pub mod concatenate;
pub mod ewm;
pub mod float;
pub mod list;
pub mod list_bytes_iter;
pub mod rolling;
pub mod set;
pub mod sort_partition;
#[cfg(feature = "performant")]
pub mod sorted_join;
#[cfg(feature = "strings")]
pub mod string;
pub mod take_agg;
#[cfg(feature = "timezones")]
mod time;
#[cfg(feature = "timezones")]
pub use time::replace_timezone;

/// Internal state of [SlicesIterator]
#[derive(Debug, PartialEq)]
enum State {
    // it is iterating over bits of a mask (`u64`, steps of size of 1 slot)
    Bits(u64),
    // it is iterating over chunks (steps of size of 64 slots)
    Chunks,
    // it is iterating over the remaining bits (steps of size of 1 slot)
    Remainder,
    // nothing more to iterate.
    Finish,
}

/// Forked and modified from arrow crate.
///
/// An iterator of `(usize, usize)` each representing an interval `[start,end[` whose
/// slots of a [BooleanArray] are true. Each interval corresponds to a contiguous region of memory to be
/// "taken" from an array to be filtered.
#[derive(Debug)]
struct MaskedSlicesIterator<'a> {
    iter: Enumerate<BitChunks<'a, u64>>,
    state: State,
    remainder_mask: u64,
    remainder_len: usize,
    chunk_len: usize,
    len: usize,
    start: usize,
    on_region: bool,
    current_chunk: usize,
    current_bit: usize,
    total_len: usize,
}

impl<'a> MaskedSlicesIterator<'a> {
    pub(crate) fn new(mask: &'a BooleanArray) -> Self {
        let chunks = mask.values().chunks::<u64>();

        let chunk_bits = 8 * std::mem::size_of::<u64>();
        let chunk_len = mask.len() / chunk_bits;
        let remainder_len = chunks.remainder_len();
        let remainder_mask = chunks.remainder();

        Self {
            iter: chunks.enumerate(),
            state: State::Chunks,
            remainder_len,
            chunk_len,
            remainder_mask,
            len: 0,
            start: 0,
            on_region: false,
            current_chunk: 0,
            current_bit: 0,
            total_len: mask.len(),
        }
    }

    #[inline]
    fn current_start(&self) -> usize {
        self.current_chunk * 64 + self.current_bit
    }

    #[inline]
    fn iterate_bits(&mut self, mask: u64, max: usize) -> Option<(usize, usize)> {
        while self.current_bit < max {
            if (mask & (1 << self.current_bit)) != 0 {
                if !self.on_region {
                    self.start = self.current_start();
                    self.on_region = true;
                }
                self.len += 1;
            } else if self.on_region {
                let result = (self.start, self.start + self.len);
                self.len = 0;
                self.on_region = false;
                self.current_bit += 1;
                return Some(result);
            }
            self.current_bit += 1;
        }
        self.current_bit = 0;
        None
    }

    /// iterates over chunks.
    #[inline]
    fn iterate_chunks(&mut self) -> Option<(usize, usize)> {
        while let Some((i, mask)) = self.iter.next() {
            self.current_chunk = i;
            if mask == 0 {
                if self.on_region {
                    let result = (self.start, self.start + self.len);
                    self.len = 0;
                    self.on_region = false;
                    return Some(result);
                }
            } else if mask == u64::MAX {
                // = !0u64
                if !self.on_region {
                    self.start = self.current_start();
                    self.on_region = true;
                }
                self.len += 64;
            } else {
                // there is a chunk that has a non-trivial mask => iterate over bits.
                self.state = State::Bits(mask);
                return None;
            }
        }
        // no more chunks => start iterating over the remainder
        self.current_chunk = self.chunk_len;
        self.state = State::Remainder;
        None
    }
}

impl<'a> Iterator for MaskedSlicesIterator<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self.state {
            State::Chunks => {
                match self.iterate_chunks() {
                    None => {
                        // iterating over chunks does not yield any new slice => continue to the next
                        self.current_bit = 0;
                        self.next()
                    }
                    other => other,
                }
            }
            State::Bits(mask) => {
                match self.iterate_bits(mask, 64) {
                    None => {
                        // iterating over bits does not yield any new slice => change back
                        // to chunks and continue to the next
                        self.state = State::Chunks;
                        self.next()
                    }
                    other => other,
                }
            }
            State::Remainder => match self.iterate_bits(self.remainder_mask, self.remainder_len) {
                None => {
                    self.state = State::Finish;
                    if self.on_region {
                        Some((self.start, self.start + self.len))
                    } else {
                        None
                    }
                }
                other => other,
            },
            State::Finish => None,
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
enum BinaryMaskedState {
    Start,
    // Last masks were false values
    LastFalse,
    // Last masks were true values
    LastTrue,
    Finish,
}

pub(crate) struct BinaryMaskedSliceIterator<'a> {
    slice_iter: MaskedSlicesIterator<'a>,
    filled: usize,
    low: usize,
    high: usize,
    state: BinaryMaskedState,
}

impl<'a> BinaryMaskedSliceIterator<'a> {
    pub(crate) fn new(mask: &'a BooleanArray) -> Self {
        Self {
            slice_iter: MaskedSlicesIterator::new(mask),
            filled: 0,
            low: 0,
            high: 0,
            state: BinaryMaskedState::Start,
        }
    }
}

impl<'a> Iterator for BinaryMaskedSliceIterator<'a> {
    type Item = (usize, usize, bool);

    fn next(&mut self) -> Option<Self::Item> {
        use BinaryMaskedState::*;

        match self.state {
            Start => {
                // first iteration
                if self.low == 0 && self.high == 0 {
                    match self.slice_iter.next() {
                        Some((low, high)) => {
                            self.low = low;
                            self.high = high;

                            if low > 0 {
                                // do another start iteration.
                                Some((0, low, false))
                            } else {
                                self.state = LastTrue;
                                self.filled = high;
                                Some((low, high, true))
                            }
                        }
                        None => {
                            self.state = Finish;
                            Some((self.filled, self.slice_iter.total_len, false))
                        }
                    }
                } else {
                    self.filled = self.high;
                    self.state = LastTrue;
                    Some((self.low, self.high, true))
                }
            }
            LastFalse => {
                self.state = LastTrue;
                self.filled = self.high;
                Some((self.low, self.high, true))
            }
            LastTrue => match self.slice_iter.next() {
                Some((low, high)) => {
                    self.low = low;
                    self.high = high;
                    self.state = LastFalse;
                    let last_filled = self.filled;
                    self.filled = low;
                    Some((last_filled, low, false))
                }
                None => {
                    self.state = Finish;
                    if self.filled != self.slice_iter.total_len {
                        Some((self.filled, self.slice_iter.total_len, false))
                    } else {
                        None
                    }
                }
            },
            Finish => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_binary_masked_slice_iter() {
        let mask = BooleanArray::from_slice(&[false, false, true, true, true, false, false]);

        let out = BinaryMaskedSliceIterator::new(&mask)
            .into_iter()
            .map(|(_, _, b)| b)
            .collect::<Vec<_>>();
        assert_eq!(out, &[false, true, false]);
        let out = BinaryMaskedSliceIterator::new(&mask)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[(0, 2, false), (2, 5, true), (5, 7, false)]);
        let mask = BooleanArray::from_slice(&[true, true, false, true]);
        let out = BinaryMaskedSliceIterator::new(&mask)
            .into_iter()
            .map(|(_, _, b)| b)
            .collect::<Vec<_>>();
        assert_eq!(out, &[true, false, true]);
        let mask = BooleanArray::from_slice(&[true, true, false, true, true]);
        let out = BinaryMaskedSliceIterator::new(&mask)
            .into_iter()
            .map(|(_, _, b)| b)
            .collect::<Vec<_>>();
        assert_eq!(out, &[true, false, true]);
    }

    #[test]
    fn test_binary_slice_mask_iter_with_false() {
        let mask = BooleanArray::from_slice(&[false, false]);
        let out = BinaryMaskedSliceIterator::new(&mask)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[(0, 2, false)]);
    }
}
