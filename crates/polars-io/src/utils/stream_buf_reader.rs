use std::io::{BufRead, Cursor};

use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_utils::async_utils::tokio_handle_ext;
use tokio::sync::OwnedSemaphorePermit;

use crate::pl_async;

pub struct OpenReaderState {
    receiver: tokio::sync::mpsc::Receiver<(
        tokio_handle_ext::AbortOnDropHandle<PolarsResult<Buffer<u8>>>,
        OwnedSemaphorePermit,
    )>,
    producer_task_handle: tokio_handle_ext::AbortOnDropHandle<std::io::Result<()>>,
    current: Buffer<u8>,
}

/// `BufRead` interface for a channel that is receiving `Buffer<u8>` bytes.
pub enum StreamBufReader {
    Open(OpenReaderState),
    Finished,
}

impl StreamBufReader {
    pub fn new(
        receiver: tokio::sync::mpsc::Receiver<(
            tokio_handle_ext::AbortOnDropHandle<PolarsResult<Buffer<u8>>>,
            OwnedSemaphorePermit,
        )>,
        producer_task_handle: tokio_handle_ext::AbortOnDropHandle<std::io::Result<()>>,
    ) -> Self {
        Self::Open(OpenReaderState {
            receiver,
            producer_task_handle,
            current: Buffer::default(),
        })
    }

    fn get_open_state(&mut self) -> Option<&mut OpenReaderState> {
        match self {
            Self::Open(state) => Some(state),
            Self::Finished => None,
        }
    }

    fn finish(&mut self) -> std::io::Result<()> {
        let Self::Open(state) = std::mem::replace(self, Self::Finished) else {
            return Ok(());
        };

        drop(state.receiver);

        pl_async::get_runtime().block_in_place_on(state.producer_task_handle)?
    }
}

impl std::io::Read for StreamBufReader {
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

impl std::io::BufRead for StreamBufReader {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        let Some(state) = self.get_open_state() else {
            return Ok(&[]);
        };

        if state.current.is_empty() {
            match state.receiver.blocking_recv() {
                Some((handle, _permit)) => {
                    let fetched_bytes =
                        pl_async::get_runtime().block_in_place_on(handle).unwrap()?;
                    state.current = fetched_bytes;
                },
                None => {
                    self.finish()?;
                    return Ok(&[]);
                },
            }
        }

        let Some(state) = self.get_open_state() else {
            unreachable!();
        };

        Ok(state.current.as_ref())
    }

    fn consume(&mut self, amt: usize) {
        if let Some(state) = self.get_open_state() {
            state.current.slice_in_place(amt..);
        }
    }
}

// Supported reader sources for respectively from_memory and streaming.
pub enum ReaderSource {
    Memory(Cursor<Buffer<u8>>),
    Streaming(StreamBufReader),
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
