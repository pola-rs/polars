use std::num::NonZeroUsize;

use bytes::Bytes;
use object_store::PutPayload;

/// Utility for byte buffering logic. Accepts both owned [`Bytes`] and borrowed `&[u8]` incoming
/// bytes. Buffered bytes can be flushed to a [`PutPayload`].
pub(super) struct BytesBufferer {
    /// Buffer until this many bytes. If set to `0`, buffering is disabled.
    target_output_size: usize,
    buffered_bytes: Vec<Bytes>,
    /// Copy bytes from small or borrowed (`&[u8]`) incoming buffers.
    copy_buffer: Vec<u8>,
    copy_buffer_reserve_size: usize,
    /// Total bytes buffered, includes both `buffered_bytes` and `copy_buffer.len()`.
    num_bytes_buffered: usize,
    tail_coalesce_num_items: usize,
    tail_coalesce_byte_offset: usize,
}

impl BytesBufferer {
    pub(super) fn new(target_output_size: usize) -> Self {
        let copy_buffer_reserve_size = usize::min(target_output_size, get_copy_buffer_size().get());

        BytesBufferer {
            target_output_size,

            buffered_bytes: Vec::with_capacity(if target_output_size == 0 {
                1
            } else {
                usize::max(
                    target_output_size.div_ceil(copy_buffer_reserve_size),
                    get_coalesce_run_length(),
                )
            }),
            copy_buffer: vec![],
            copy_buffer_reserve_size,
            num_bytes_buffered: 0,
            tail_coalesce_num_items: 0,
            tail_coalesce_byte_offset: 0,
        }
    }

    /// Push owned [`Bytes`] into this bufferer. This will consume from a mutable reference
    /// via [`Bytes::split_to`] until either the bytes is fully consumed, or `self` is full.
    pub(super) fn push_owned(&mut self, bytes: &mut Bytes) {
        if bytes.is_empty() {
            return;
        }

        let available_capacity = self.available_capacity_current_chunk(bytes.len());

        if available_capacity == 0 {
            return;
        }

        loop {
            let copy_buffer_available_capacity = usize::min(
                available_capacity,
                self.copy_buffer.capacity() - self.copy_buffer.len(),
            );

            if bytes.len() <= copy_buffer_available_capacity {
                self.copy_buffer.extend_from_slice(bytes);
                self.num_bytes_buffered += bytes.len();
                *bytes = Bytes::new();

                return;
            }

            self.commit_active_copy_buffer();

            if self.tail_coalesce_num_items >= cloud_writer_coalesce_run_length() {
                self.coalesce_tail(self.tail_coalesce_num_items, self.tail_coalesce_byte_offset);
                self.reset_tail_coalesce_counters();
                continue;
            }

            break;
        }

        let bytes = bytes.split_to(usize::min(bytes.len(), available_capacity));

        let bytes_len = bytes.len();
        self.buffered_bytes.push(bytes);
        self.num_bytes_buffered += bytes_len;

        if self.num_bytes_buffered - self.tail_coalesce_byte_offset <= self.copy_buffer_reserve_size
        {
            self.tail_coalesce_num_items += 1;
        } else {
            self.reset_tail_coalesce_counters();
        }
    }

    /// Push borrowed `&[u8]` into this bufferer. This will consume from a mutable reference
    /// via `split_off` until either the slice is fully consumed, or `self` is full.
    pub(super) fn push_slice(&mut self, bytes: &mut &[u8]) {
        while !bytes.is_empty() {
            let available_capacity = self.available_capacity_current_chunk(bytes.len());

            if available_capacity == 0 {
                break;
            }

            let mut copy_buffer_available_capacity = usize::min(
                available_capacity,
                self.copy_buffer.capacity() - self.copy_buffer.len(),
            );

            if copy_buffer_available_capacity == 0 {
                self.commit_active_copy_buffer();
                copy_buffer_available_capacity =
                    self.reserve_active_copy_buffer(available_capacity);
            }

            let n = usize::min(bytes.len(), copy_buffer_available_capacity);

            self.copy_buffer
                .extend_from_slice(bytes.split_off(..n).unwrap());
            self.num_bytes_buffered += n;
        }
    }

    fn coalesce_tail(&mut self, num_items: usize, byte_offset: usize) {
        assert_eq!(self.copy_buffer.capacity(), 0);
        assert!(num_items >= 2);
        assert!(byte_offset < self.target_output_size);

        self.copy_buffer.reserve_exact(usize::min(
            self.copy_buffer_reserve_size,
            self.target_output_size - byte_offset,
        ));

        assert!(self.copy_buffer.capacity() >= (self.num_bytes_buffered - byte_offset));

        let drain_start = self.buffered_bytes.len() - num_items;

        self.buffered_bytes
            .drain(drain_start..)
            .for_each(|bytes| self.copy_buffer.extend_from_slice(&bytes));
    }

    fn reset_tail_coalesce_counters(&mut self) {
        self.tail_coalesce_byte_offset = self.num_bytes_buffered;
        self.tail_coalesce_num_items = 0;
    }

    pub(super) fn is_empty(&self) -> bool {
        if self.num_bytes_buffered == 0 {
            assert!(self.buffered_bytes.is_empty());
            assert_eq!(self.copy_buffer.capacity(), 0);
            true
        } else {
            false
        }
    }

    pub(super) fn has_complete_chunk(&self) -> bool {
        self.num_bytes_buffered >= usize::max(1, self.target_output_size)
    }

    pub(super) fn flush_complete_chunk(&mut self) -> Option<PutPayload> {
        self.has_complete_chunk().then(|| self.flush().unwrap())
    }

    pub(super) fn flush(&mut self) -> Option<PutPayload> {
        if self.is_empty() {
            return None;
        }

        self.commit_active_copy_buffer();

        self.num_bytes_buffered = 0;
        self.reset_tail_coalesce_counters();

        let payload = PutPayload::from_iter(self.buffered_bytes.drain(..));

        Some(payload)
    }

    fn available_capacity_current_chunk(&self, incoming_len: usize) -> usize {
        if self.target_output_size > 0 {
            self.target_output_size - self.num_bytes_buffered
        } else if self.is_empty() {
            incoming_len
        } else {
            0
        }
    }

    #[inline]
    fn commit_active_copy_buffer(&mut self) {
        if !self.copy_buffer.is_empty() {
            self.num_bytes_buffered -= self.copy_buffer.len();
            let mut bytes: Bytes = std::mem::take(&mut self.copy_buffer).into();
            self.push_owned(&mut bytes);
            assert!(bytes.is_empty());
        }
    }

    fn reserve_active_copy_buffer(&mut self, available_capacity_current_chunk: usize) -> usize {
        let n = if self.copy_buffer_reserve_size > 0 {
            usize::min(
                self.copy_buffer_reserve_size,
                available_capacity_current_chunk,
            )
        } else {
            available_capacity_current_chunk
        };

        self.copy_buffer.reserve_exact(n);

        usize::min(
            self.copy_buffer.capacity() - self.copy_buffer.len(),
            available_capacity_current_chunk,
        )
    }
}

/// Runs of this many values whose total bytes are <= `copy_buffer_reserve_size` will be copied into
/// a single contiguous chunk.
fn get_coalesce_run_length() -> usize {
    return *COALESCE_RUN_LENGTH;

    static COALESCE_RUN_LENGTH: LazyLock<usize> = LazyLock::new(|| {
        let mut v: usize = 64;

        if let Ok(x) = std::env::var("POLARS_UPLOAD_COALESCE_RUN_LENGTH") {
            v = x
                .parse::<usize>()
                .ok()
                .filter(|x| *x >= 2)
                .unwrap_or_else(|| {
                    panic!("invalid value for POLARS_UPLOAD_COALESCE_RUN_LENGTH: {x}")
                })
        }

        if polars_core::config::verbose() {
            eprintln!("upload coalesce_run_length: {v}")
        }

        v
    });

    use std::sync::LazyLock;
}

fn get_copy_buffer_size() -> NonZeroUsize {
    return *COPY_BUFFER_SIZE;

    static COPY_BUFFER_SIZE: LazyLock<NonZeroUsize> = LazyLock::new(|| {
        let mut v: NonZeroUsize = const { NonZeroUsize::new(16 * 1024 * 1024).unwrap() };

        if let Ok(x) = std::env::var("POLARS_UPLOAD_COPY_BUFFER_SIZE") {
            v = x
                .parse::<NonZeroUsize>()
                .unwrap_or_else(|_| panic!("invalid value for POLARS_UPLOAD_COPY_BUFFER_SIZE: {x}"))
        }

        if polars_core::config::verbose() {
            eprintln!("upload copy_buffer_size: {v}")
        }

        v
    });

    use std::sync::LazyLock;
}
