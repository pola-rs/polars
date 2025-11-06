/// Cursor over fixed size chunks of bytes.
pub struct FixedSizeChunkedBytesCursor<'a, T> {
    position: usize,
    total_size: usize,
    chunk_size: usize,
    chunked_bytes: &'a [T],
}

impl<'a, T> FixedSizeChunkedBytesCursor<'a, T>
where
    T: AsRef<[u8]>,
{
    /// Returns error if `chunked_bytes` is empty, or if a byte slice that is not the last one
    /// has a length != `chunk_size`;
    pub fn try_new(chunked_bytes: &'a [T], chunk_size: usize) -> Result<Self, ()> {
        if chunked_bytes.is_empty() {
            return Err(());
        }

        let mut total_size: usize = 0;

        for (i, bytes) in chunked_bytes.iter().enumerate() {
            let bytes = bytes.as_ref();

            if bytes.len() != chunk_size && chunked_bytes.len() - i > 1 {
                return Err(());
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
        let new_position =
            self.position + buf.len().min(self.total_size.saturating_sub(self.position));

        let byte_range = self.position..new_position;

        if byte_range.is_empty() {
            return Ok(0);
        }

        let mut bytes_read: usize = 0;

        {
            let chunk_bytes = self.chunked_bytes[byte_range.start / self.chunk_size].as_ref();
            let offset_in_chunk = (byte_range.start % self.chunk_size).min(chunk_bytes.len());
            let len = byte_range.len().min(chunk_bytes.len() - offset_in_chunk);

            buf[..len].copy_from_slice(&chunk_bytes[offset_in_chunk..][..len]);

            bytes_read += len;
        }

        {
            for chunk_idx in
                byte_range.start.div_ceil(self.chunk_size)..byte_range.end.div_ceil(self.chunk_size)
            {
                let chunk_bytes = self.chunked_bytes[chunk_idx].as_ref();
                let len = (byte_range.len() - bytes_read).min(chunk_bytes.len());

                buf[bytes_read..][..len].copy_from_slice(&chunk_bytes[..len]);

                bytes_read += len;
            }
        }

        assert_eq!(bytes_read, byte_range.len());

        self.position = new_position;

        Ok(byte_range.len())
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
