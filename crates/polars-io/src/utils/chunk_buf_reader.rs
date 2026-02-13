use std::io::{BufRead, Cursor};

use polars_buffer::Buffer;

// Encapsulated MPSC receiver with buffer.
pub struct ChunkBufReader {
    receiver: std::sync::mpsc::Receiver<Buffer<u8>>,
    current: Buffer<u8>,
    offset: usize,
    finished: bool,
}

impl ChunkBufReader {
    pub fn new(receiver: std::sync::mpsc::Receiver<Buffer<u8>>) -> Self {
        Self {
            receiver,
            current: Buffer::default(),
            offset: 0,
            finished: false,
        }
    }
}

impl std::io::Read for ChunkBufReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let remaining = self.fill_buf()?;
        if remaining.is_empty() {
            return Ok(0);
        }
        let n = buf.len().min(remaining.len());
        buf[..n].copy_from_slice(&remaining[..n]);
        self.consume(n);
        Ok(n)
    }
}

impl std::io::BufRead for ChunkBufReader {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        if self.offset >= self.current.len() {
            if self.finished {
                return Ok(&[]);
            }
            match self.receiver.recv() {
                Ok(chunk) => {
                    self.current = chunk;
                    self.offset = 0;
                },
                Err(_) => {
                    self.finished = true;
                    return Ok(&[]);
                },
            }
        }
        Ok(&self.current.as_ref()[self.offset..])
    }

    fn consume(&mut self, amt: usize) {
        self.offset += amt;
    }
}

// Supported reader sources for respectively from_memory and streaming.
pub enum ReaderSource {
    Memory(Cursor<Buffer<u8>>),
    Streaming(ChunkBufReader),
}

impl std::io::Read for ReaderSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Memory(r) => r.read(buf),
            Self::Streaming(r) => r.read(buf),
        }
    }
}

impl std::io::BufRead for ReaderSource {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        match self {
            Self::Memory(r) => r.fill_buf(),
            Self::Streaming(r) => r.fill_buf(),
        }
    }

    fn consume(&mut self, amt: usize) {
        match self {
            Self::Memory(r) => r.consume(amt),
            Self::Streaming(r) => r.consume(amt),
        }
    }
}
