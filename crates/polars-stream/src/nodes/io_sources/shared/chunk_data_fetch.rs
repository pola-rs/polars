use std::sync::Arc;

use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use tokio::sync::mpsc::Sender;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::async_primitives::wait_group::WaitToken;
use crate::utils::tokio_handle_ext;

pub(crate) struct ChunkDataFetcher {
    pub(crate) memory_prefetch_func: fn(&[u8]) -> (),
    pub(crate) byte_source: Arc<DynByteSource>,
    pub(crate) file_size: usize,
    pub(crate) chunk_size: usize,
    pub(crate) prefetch_send: Sender<(
        tokio_handle_ext::AbortOnDropHandle<PolarsResult<Buffer<u8>>>,
        OwnedSemaphorePermit,
    )>,
    pub(crate) prefetch_semaphore: Arc<Semaphore>,
    pub(crate) prefetch_current_all_spawned: Option<WaitToken>,
}

impl ChunkDataFetcher {
    pub(crate) async fn run(&mut self) -> PolarsResult<()> {
        let mut byte_offset = 0;
        let file_size = self.file_size;

        while byte_offset < file_size {
            let fetch_permit = self
                .prefetch_semaphore
                .clone()
                .acquire_owned()
                .await
                .unwrap();

            let chunk_size = self.chunk_size;
            let current_byte_source = self.byte_source.clone();
            let memory_prefetch_func = self.memory_prefetch_func;
            let io_runtime = polars_io::pl_async::get_runtime();

            let range = byte_offset..std::cmp::min(file_size, byte_offset + chunk_size);
            let range_len = range.len();

            let handle = io_runtime.spawn(async move {
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
                    } else {
                        // @NOTE. Performance can be optimized by grouping requests and downloading
                        // through `get_ranges()`.
                        current_byte_source.get_range(range.clone()).await?
                    };

                PolarsResult::Ok(fetched_bytes)
            });

            let handle = tokio_handle_ext::AbortOnDropHandle(handle);
            if self
                .prefetch_send
                .send((handle, fetch_permit))
                .await
                .is_err()
            {
                break;
            }

            byte_offset += range_len;
        }

        drop(self.prefetch_current_all_spawned.take());

        Ok(())
    }
}
