use std::cmp::Reverse;
use std::fs::File;

use polars_error::PolarsResult;
use polars_utils::priority::Priority;

use super::MorselSeq;
use crate::async_primitives::linearizer::Linearizer;

#[cfg(feature = "csv")]
pub mod csv;
#[cfg(feature = "ipc")]
pub mod ipc;
#[cfg(feature = "parquet")]
pub mod parquet;

/// Write buffers coming from a linearizer to a [`File`].
async fn write_buffers_from_linearizer_to_file(
    mut file: File,
    mut linearizer: Linearizer<Priority<Reverse<MorselSeq>, Vec<u8>>>,
) -> PolarsResult<()> {
    // On Unix systems we can use the `pwrite64` syscall to parallelize multiple writes.
    //
    // This seems to speed-up writing by quite a bit.
    #[cfg(target_family = "unix")]
    {
        use std::io::Seek;
        use std::os::unix::fs::FileExt;

        // This is taken without too much thought. It might be good to couple to the number of
        // threads available.
        const NUM_WRITING_TASKS: usize = 4;

        // Get the initial position in the file. We will start writing from here.
        let mut seek_position = file.seek(std::io::SeekFrom::Current(0))?;
        let mut futures = Vec::with_capacity(NUM_WRITING_TASKS);

        let io_runtime = polars_io::pl_async::get_runtime();
        // A reference to file needs to be given to each writing task. Therefore, we put the file
        // in an Arc and clone it for each write.
        let file = std::sync::Arc::new(file);

        while let Some(Priority(_, buffer)) = linearizer.get().await {
            // If we have too many writing tasks out at the moment, wait for one to complete before
            // spawning another one.
            if futures.len() >= NUM_WRITING_TASKS {
                let result: Result<_, tokio::task::JoinError>;
                (result, _, futures) = futures::future::select_all(futures).await;
                result.unwrap_or_else(|err| {
                    if err.is_panic() {
                        // Resume the panic on the main task
                        std::panic::resume_unwind(err.into_panic());
                    }

                    Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Failed to join on write_all_at",
                    ))
                })?;
            }

            let buffer_len = buffer.len();
            let file = file.clone();
            futures.push(io_runtime.spawn_blocking(move || {
                // Move the buffer over. This allows freeing (i.e. munmap) it to be parallel as
                // well.
                let buffer = buffer;

                file.write_all_at(&buffer, seek_position)
            }));
            seek_position += buffer_len as u64;
        }
    }

    // @TODO:
    // It might be worth it to investigate optimizing this for WASI also with `write_all_at` and
    // for windows with `seek_write`.
    #[cfg(not(target_family = "unix"))]
    {
        use std::io::Write;
        while let Some(Priority(_, buffer)) = linearizer.get().await {
            file.write_all(&buffer)
        }
    }

    Ok(())
}
