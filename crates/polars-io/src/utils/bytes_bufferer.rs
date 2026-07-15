use std::num::NonZeroUsize;
use std::ops::Range;

use arrow::io::write_owned::WriteBytesOwned;
use polars_buffer::Buffer;

#[derive(Debug, Clone)]
pub struct BytesBuffererConfig {
    pub target_size: Range<NonZeroUsize>,
    /// `min..max` allocation size. `min` / `max` must be less than
    /// `target_size.min` / `target_size.max` respectively.
    pub copy_buffer_reserve_size: Range<NonZeroUsize>,
}

#[derive(Debug, Clone, Default)]
pub struct BytesBuffererStats {
    pub total_bytes: usize,
    pub chunks: usize,
    pub copied_bytes: usize,
    pub allocations: u64,
}

/// Utility for byte buffering logic. Accepts both owned [`Buffer<u8>`] and borrowed `&[u8]` incoming
/// bytes.
pub struct BytesBufferer {
    target_size: Range<NonZeroUsize>,
    copy_buffer_reserve_size: Range<NonZeroUsize>,
    buffered_bytes: Vec<Buffer<u8>>,
    copy_buffer: Vec<u8>,
    num_bytes_buffered: usize,

    // Metrics
    num_copied_bytes: usize,
    num_allocations: u64,
}

impl BytesBufferer {
    pub fn new(config: &BytesBuffererConfig) -> Self {
        let BytesBuffererConfig {
            target_size,
            copy_buffer_reserve_size,
        } = config;

        BytesBufferer {
            target_size: target_size.clone(),
            copy_buffer_reserve_size: copy_buffer_reserve_size.clone(),
            buffered_bytes: Vec::with_capacity(8),
            copy_buffer: vec![],
            num_bytes_buffered: 0,

            num_copied_bytes: 0,
            num_allocations: 0,
        }
    }

    pub fn stats(&self) -> BytesBuffererStats {
        BytesBuffererStats {
            total_bytes: self.num_bytes_buffered,
            chunks: self.buffered_bytes.len() + !self.copy_buffer.is_empty() as usize,
            copied_bytes: self.num_copied_bytes,
            allocations: self.num_allocations,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.num_bytes_buffered
    }

    pub fn push_owned(&mut self, bytes: &Buffer<u8>) {
        if bytes.len() <= self.copy_buffer.capacity() - self.copy_buffer.len() {
            self.push_slice(bytes);
            return;
        }

        let copy_buffer_was_empty = !self.commit_copy_buffer();
        let half_min_target_size = self.target_size.start.get() / 2;

        if copy_buffer_was_empty
            && bytes.len() < half_min_target_size
            && let Some(prev_bytes) = self
                .buffered_bytes
                .pop_if(|prev_bytes| prev_bytes.len() < half_min_target_size)
        {
            self.num_bytes_buffered -= prev_bytes.len();
            self.reserve_copy_buffer(prev_bytes.len() + bytes.len());
            self.push_slice(&prev_bytes);
            self.push_slice(bytes);
            return;
        }

        let bytes_len = bytes.len();

        if let Some((n_parts, part_size, rem)) = (bytes.len() / self.target_size.end.get()
            ..=bytes.len().div_ceil(self.target_size.end.get()))
            .filter(|n_parts| *n_parts != 0)
            .map(|n_parts| (n_parts, bytes.len() / n_parts, bytes.len() % n_parts))
            .max_by_key(|(_, part_size, _)| part_size.abs_diff(self.target_size.end.get()))
            && n_parts > 1
        {
            for i in 0..n_parts {
                let start = i * part_size + usize::min(rem, i);
                let end = start + part_size + (i < rem) as usize;

                self.buffered_bytes
                    .push(Buffer::clone(bytes).sliced(start..end));
            }
        } else {
            self.buffered_bytes.push(Buffer::clone(bytes));
        }

        self.num_bytes_buffered += bytes_len;
    }

    pub fn push_slice(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            if self.copy_buffer.is_empty() {
                self.reserve_copy_buffer(bytes.len());
            }

            if let n = self.copy_buffer.capacity() - self.copy_buffer.len()
                && n != 0
            {
                let n = usize::min(n, bytes.len());

                self.copy_buffer
                    .extend_from_slice(bytes.split_off(..n).unwrap());
                self.num_bytes_buffered += n;
                self.num_copied_bytes += n;
            }

            if self.copy_buffer.len() == self.copy_buffer.capacity() {
                self.commit_copy_buffer();
            } else {
                assert!(bytes.is_empty());
            }
        }
    }

    pub fn drain(&mut self) -> std::vec::Drain<'_, Buffer<u8>> {
        self.commit_copy_buffer();
        self.num_bytes_buffered = 0;
        self.buffered_bytes.drain(..)
    }

    /// Guarantees that `self.copy_buffer` has spare capacity. Does not guarantee how much spare
    /// capacity (should be re-called if more capacity is needed).
    ///
    /// # Panics
    /// Panics if `self.copy_buffer` is not empty.
    fn reserve_copy_buffer(&mut self, incoming_len: usize) {
        assert_eq!(self.copy_buffer.len(), 0);

        if self.copy_buffer.capacity() != 0 {
            return;
        }

        self.num_allocations += 1;

        let reserve_size = usize::max(self.num_bytes_buffered.saturating_mul(2), incoming_len)
            .clamp(
                self.copy_buffer_reserve_size.start.get(),
                self.copy_buffer_reserve_size.end.get(),
            );

        // Avoid over-allocating for small header pushes.
        if incoming_len.saturating_mul(128) < reserve_size
            && self.buffered_bytes.last().is_none_or(|bytes| {
                // Prevent this branch from being hit successively on multiple
                // small writes that accumulate to a large amount.
                bytes.len() > 16 * incoming_len
            })
        {
            self.copy_buffer.reserve_exact(8 * incoming_len);
            return;
        }

        self.copy_buffer.reserve_exact(reserve_size);
    }

    #[inline]
    fn commit_copy_buffer(&mut self) -> bool {
        if !self.copy_buffer.is_empty() {
            self.buffered_bytes
                .push(Buffer::from_vec(std::mem::take(&mut self.copy_buffer)));
            true
        } else {
            false
        }
    }
}

impl WriteBytesOwned for BytesBufferer {
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.push_slice(buf);
        Ok(())
    }

    fn write_all_owned(&mut self, bytes: &Buffer<u8>) -> std::io::Result<()> {
        self.push_owned(bytes);
        Ok(())
    }

    fn len(&self) -> usize {
        BytesBufferer::len(self)
    }
}

impl IntoIterator for BytesBufferer {
    type Item = <Vec<Buffer<u8>> as IntoIterator>::Item;
    type IntoIter = <Vec<Buffer<u8>> as IntoIterator>::IntoIter;

    fn into_iter(mut self) -> Self::IntoIter {
        self.commit_copy_buffer();
        self.buffered_bytes.into_iter()
    }
}
