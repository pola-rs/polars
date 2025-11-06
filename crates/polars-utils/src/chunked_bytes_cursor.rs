use std::num::NonZeroUsize;

/// Cursor over fixed size chunks of bytes.
pub struct FixedSizeChunkedBytesCursor<'a, T> {
    position: usize,
    total_size: usize,
    chunk_size: NonZeroUsize,
    /// Note, the last chunk is allowed to have a length shorter than the `chunk_size`.
    chunked_bytes: &'a [T],
}

#[derive(Debug)]
pub enum FixedSizeChunkedBytesCursorInitErr {
    ChunkLengthMismatch { index: usize },
    EmptyFirstChunk,
    NoChunks,
}

impl<'a, T> FixedSizeChunkedBytesCursor<'a, T>
where
    T: AsRef<[u8]>,
{
    /// Expects `chunked_bytes` to have a non-empty length `n`, where `chunked_bytes[..n - 1]` all have the same length.
    pub fn try_new(chunked_bytes: &'a [T]) -> Result<Self, FixedSizeChunkedBytesCursorInitErr> {
        use FixedSizeChunkedBytesCursorInitErr as E;

        if chunked_bytes.is_empty() {
            return Err(E::NoChunks);
        }

        let Ok(chunk_size) = NonZeroUsize::try_from(chunked_bytes[0].as_ref().len()) else {
            return Err(E::EmptyFirstChunk);
        };

        let mut total_size: usize = 0;

        for (i, bytes) in chunked_bytes.iter().enumerate() {
            let bytes = bytes.as_ref();

            if bytes.len() != chunk_size.get() && chunked_bytes.len() - i > 1 {
                return Err(E::ChunkLengthMismatch { index: i });
            }

            total_size = total_size.checked_add(bytes.len()).unwrap();
        }

        Ok(Self {
            position: 0,
            total_size,
            chunk_size,
            chunked_bytes,
        })
    }
}

impl<'a, T> std::io::Read for FixedSizeChunkedBytesCursor<'a, T>
where
    T: AsRef<[u8]>,
{
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let available_bytes = self.total_size.saturating_sub(self.position);
        let new_position = self.position + buf.len().min(available_bytes);

        let requested_byte_range = self.position..new_position;

        if requested_byte_range.is_empty() {
            return Ok(0);
        }

        // First chunk needs special handling as the offset within the chunk can be non-zero.
        let mut bytes_read = {
            let (first_chunk_idx, offset_in_chunk) = (
                requested_byte_range.start / self.chunk_size,
                requested_byte_range.start % self.chunk_size,
            );
            let chunk_bytes = self.chunked_bytes[first_chunk_idx].as_ref();
            let len = requested_byte_range
                .len()
                .min(chunk_bytes.len() - offset_in_chunk);

            buf[..len].copy_from_slice(&chunk_bytes[offset_in_chunk..offset_in_chunk + len]);

            len
        };

        assert!(
            (requested_byte_range.start + bytes_read).is_multiple_of(self.chunk_size.get())
                || bytes_read == requested_byte_range.len()
        );

        for chunk_idx in (requested_byte_range.start + bytes_read) / self.chunk_size
            ..requested_byte_range.end.div_ceil(self.chunk_size.get())
        {
            let chunk_bytes = self.chunked_bytes[chunk_idx].as_ref();
            let len = (requested_byte_range.len() - bytes_read).min(chunk_bytes.len());

            buf[bytes_read..bytes_read + len].copy_from_slice(&chunk_bytes[..len]);

            bytes_read += len;
        }

        assert_eq!(bytes_read, requested_byte_range.len());

        self.position = new_position;

        Ok(requested_byte_range.len())
    }
}

impl<'a, T> std::io::Seek for FixedSizeChunkedBytesCursor<'a, T> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        // Mostly copied from io::Cursor::seek().
        use std::io::SeekFrom;

        let (base_pos, offset) = match pos {
            SeekFrom::Start(n) => {
                self.position = usize::try_from(n).unwrap().min(self.total_size);
                return Ok(self.position as u64);
            },
            SeekFrom::End(n) => (self.total_size as u64, n),
            SeekFrom::Current(n) => (self.position as u64, n),
        };
        match base_pos.checked_add_signed(offset) {
            Some(n) => {
                self.position = usize::try_from(n).unwrap();
                Ok(self.position as u64)
            },
            None => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid seek to a negative or overflowing position",
            )),
        }
    }
}
