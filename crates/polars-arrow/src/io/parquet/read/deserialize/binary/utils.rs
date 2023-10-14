use super::super::utils::Pushable;
use crate::offset::{Offset, Offsets};

/// [`Pushable`] for variable length binary data.
#[derive(Debug)]
pub struct Binary<O: Offset> {
    pub offsets: Offsets<O>,
    pub values: Vec<u8>,
}

impl<O: Offset> Pushable<usize> for Offsets<O> {
    fn reserve(&mut self, additional: usize) {
        self.reserve(additional)
    }
    #[inline]
    fn len(&self) -> usize {
        self.len_proxy()
    }

    #[inline]
    fn push(&mut self, value: usize) {
        self.try_push(value).unwrap()
    }

    #[inline]
    fn push_null(&mut self) {
        self.extend_constant(1);
    }

    #[inline]
    fn extend_constant(&mut self, additional: usize, _: usize) {
        self.extend_constant(additional)
    }
}

impl<O: Offset> Binary<O> {
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            offsets: Offsets::with_capacity(capacity),
            values: Vec::with_capacity(capacity.min(100) * 24),
        }
    }

    #[inline]
    pub fn push(&mut self, v: &[u8]) {
        if self.offsets.len_proxy() == 100 && self.offsets.capacity() > 100 {
            let bytes_per_row = self.values.len() / 100 + 1;
            let bytes_estimate = bytes_per_row * self.offsets.capacity();
            if bytes_estimate > self.values.capacity() {
                self.values.reserve(bytes_estimate - self.values.capacity());
            }
        }

        self.values.extend(v);
        self.offsets.try_push(v.len()).unwrap()
    }

    #[inline]
    pub fn extend_constant(&mut self, additional: usize) {
        self.offsets.extend_constant(additional);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    #[inline]
    pub fn extend_lengths<I: Iterator<Item = usize>>(&mut self, lengths: I, values: &mut &[u8]) {
        let current_offset = *self.offsets.last();
        self.offsets.try_extend_from_lengths(lengths).unwrap();
        let new_offset = *self.offsets.last();
        let length = new_offset.to_usize() - current_offset.to_usize();
        let (consumed, remaining) = values.split_at(length);
        *values = remaining;
        self.values.extend_from_slice(consumed);
    }
}

impl<'a, O: Offset> Pushable<&'a [u8]> for Binary<O> {
    #[inline]
    fn reserve(&mut self, additional: usize) {
        let avg_len = self.values.len() / std::cmp::max(self.offsets.last().to_usize(), 1);
        self.values.reserve(additional * avg_len);
        self.offsets.reserve(additional);
    }
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn push_null(&mut self) {
        self.push(&[])
    }

    #[inline]
    fn push(&mut self, value: &[u8]) {
        self.push(value)
    }

    #[inline]
    fn extend_constant(&mut self, additional: usize, value: &[u8]) {
        assert_eq!(value.len(), 0);
        self.extend_constant(additional)
    }
}

#[derive(Debug)]
pub struct BinaryIter<'a> {
    values: &'a [u8],
}

impl<'a> BinaryIter<'a> {
    pub fn new(values: &'a [u8]) -> Self {
        Self { values }
    }
}

impl<'a> Iterator for BinaryIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.values.is_empty() {
            return None;
        }
        let (length, remaining) = self.values.split_at(4);
        let length = u32::from_le_bytes(length.try_into().unwrap()) as usize;
        let (result, remaining) = remaining.split_at(length);
        self.values = remaining;
        Some(result)
    }
}

#[derive(Debug)]
pub struct SizedBinaryIter<'a> {
    iter: BinaryIter<'a>,
    remaining: usize,
}

impl<'a> SizedBinaryIter<'a> {
    pub fn new(values: &'a [u8], size: usize) -> Self {
        let iter = BinaryIter::new(values);
        Self {
            iter,
            remaining: size,
        }
    }
}

impl<'a> Iterator for SizedBinaryIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        } else {
            self.remaining -= 1
        };
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
