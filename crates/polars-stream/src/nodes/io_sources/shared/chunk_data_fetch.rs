use std::sync::Arc;

use polars_async::primitives::wait_group::WaitToken;
use polars_buffer::Buffer;
use polars_core::runtime::ASYNC;
use polars_error::PolarsResult;
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use tokio::sync::mpsc::Sender;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::utils::tokio_handle_ext;

/// Chunks of local files are read in concurrent parts of this size.
const LOCAL_READ_SIZE: usize = 4 * 1024 * 1024;

/// Number of chunks of a local file that are fetched ahead of its reader. Local reads have low
/// latency and every chunk is read concurrently, so fetching further ahead would only hold more of
/// the file in memory.
const LOCAL_PREFETCH_LIMIT: usize = 2;

pub(crate) struct ChunkDataFetcher {
    pub(crate) memory_prefetch_func: fn(&[u8]) -> (),
    pub(crate) byte_source: Arc<DynByteSource>,
    pub(crate) file_size: usize,
    pub(crate) chunk_size: usize,
    /// Already fetched source bytes from offset 0, at most `file_size` long. Sent whole as the
    /// first chunk, also when longer than `chunk_size`, and fetching continues after it.
    pub(crate) prefix: Option<Buffer<u8>>,
    pub(crate) prefetch_send: Sender<(
        tokio_handle_ext::AbortOnDropHandle<PolarsResult<Buffer<u8>>>,
        OwnedSemaphorePermit,
    )>,
    pub(crate) prefetch_semaphore: Arc<Semaphore>,
    pub(crate) prefetch_current_all_spawned: Option<WaitToken>,
}

impl ChunkDataFetcher {
    /// Number of chunks of `chunk_size` to fetch ahead of the reader of `byte_source`, which is
    /// also the capacity of the `prefetch_send` channel.
    pub(crate) fn prefetch_depth(
        byte_source: &DynByteSource,
        prefetch_limit: usize,
        file_size: usize,
        chunk_size: usize,
    ) -> usize {
        let limit = if let DynByteSource::File(_) = byte_source {
            prefetch_limit.min(LOCAL_PREFETCH_LIMIT)
        } else {
            prefetch_limit
        };

        limit.min(file_size.div_ceil(chunk_size)).max(1)
    }

    pub(crate) async fn run(&mut self) -> PolarsResult<()> {
        let mut byte_offset = 0;
        let file_size = self.file_size;

        let mut prefix = self.prefix.take().filter(|x| !x.is_empty());
        debug_assert!(prefix.as_ref().is_none_or(|x| x.len() <= file_size));

        while byte_offset < file_size {
            // Wait for a slot before fetching, so that at most the channel capacity of chunks is
            // fetched ahead of the reader.
            let Ok(slot) = self.prefetch_send.reserve().await else {
                break;
            };

            let fetch_permit = self
                .prefetch_semaphore
                .clone()
                .acquire_owned()
                .await
                .unwrap();

            let prefix = prefix.take();
            let chunk_size = prefix.as_ref().map_or(self.chunk_size, |x| x.len());
            let current_byte_source = self.byte_source.clone();
            let memory_prefetch_func = self.memory_prefetch_func;

            let range = byte_offset..std::cmp::min(file_size, byte_offset + chunk_size);
            let range_len = range.len();

            let handle = ASYNC.spawn(async move {
                if let Some(prefix) = prefix {
                    return Ok(prefix);
                }

                let fetched_bytes =
                    if let DynByteSource::Buffer(mem_slice) = current_byte_source.as_ref() {
                        let slice = mem_slice.0.as_ref();

                        if !std::ptr::eq(
                            memory_prefetch_func as *const (),
                            polars_utils::mem::prefetch::no_prefetch as *const (),
                        ) {
                            debug_assert!(range.end <= slice.len());
                            memory_prefetch_func(unsafe { slice.get_unchecked(range.clone()) })
                        }

                        mem_slice.0.clone().sliced(range)
                    } else if let DynByteSource::File(file) = current_byte_source.as_ref() {
                        file.get_range_concurrent(range, LOCAL_READ_SIZE).await?
                    } else {
                        // @NOTE. Performance can be optimized by grouping requests and downloading
                        // through `get_ranges()`.
                        current_byte_source.get_range(range.clone()).await?
                    };

                PolarsResult::Ok(fetched_bytes)
            });

            let handle = tokio_handle_ext::AbortOnDropHandle(handle);
            slot.send((handle, fetch_permit));

            byte_offset += range_len;
        }

        drop(self.prefetch_current_all_spawned.take());

        Ok(())
    }
}
