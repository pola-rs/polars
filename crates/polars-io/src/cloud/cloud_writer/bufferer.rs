use bytes::Bytes;
use object_store::PutPayload;

use crate::configs::{cloud_writer_coalesce_run_length, cloud_writer_copy_buffer_size};

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
        let copy_buffer_reserve_size =
            usize::min(target_output_size, cloud_writer_copy_buffer_size().get());

        BytesBufferer {
            target_output_size,

            buffered_bytes: Vec::with_capacity(if target_output_size == 0 {
                1
            } else {
                usize::max(
                    target_output_size.div_ceil(copy_buffer_reserve_size),
                    match cloud_writer_coalesce_run_length() {
                        n if n <= copy_buffer_reserve_size => n,
                        _ => 0,
                    },
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
                self.coalesce_tail();
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

    fn coalesce_tail(&mut self) {
        if self.tail_coalesce_num_items < 2 {
            return;
        }

        assert_eq!(self.copy_buffer.capacity(), 0);
        assert!(self.tail_coalesce_byte_offset < self.target_output_size);

        let copy_buffer_reserve = usize::min(
            self.copy_buffer_reserve_size,
            self.target_output_size - self.tail_coalesce_byte_offset,
        );

        assert!(copy_buffer_reserve >= (self.num_bytes_buffered - self.tail_coalesce_byte_offset));

        let drain_start: usize = self.buffered_bytes.len() - self.tail_coalesce_num_items;
        let drain_range = drain_start..;
        self.reset_tail_coalesce_counters();

        self.copy_buffer.reserve_exact(copy_buffer_reserve);
        self.buffered_bytes
            .drain(drain_range)
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

    pub(super) fn is_full(&self) -> bool {
        self.num_bytes_buffered >= usize::max(1, self.target_output_size)
    }

    pub(super) fn flush_full_chunk(&mut self) -> Option<PutPayload> {
        self.is_full().then(|| self.flush().unwrap())
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
