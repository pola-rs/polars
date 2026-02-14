use std::num::NonZeroUsize;
use std::sync::LazyLock;

use polars_core::config::verbose;
use polars_utils::sys::total_memory;

pub fn upload_chunk_size() -> usize {
    return *UPLOAD_CHUNK_SIZE;

    static UPLOAD_CHUNK_SIZE: LazyLock<usize> = LazyLock::new(|| {
        let mut v: usize = 32 * 1024 * 1024;

        if let Ok(s) = std::env::var("POLARS_UPLOAD_CHUNK_SIZE") {
            v = s
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid value for POLARS_UPLOAD_CHUNK_SIZE: {s}"))
        }

        if verbose() {
            eprintln!("upload_chunk_size: {v}")
        }

        v
    });
}

pub fn partitioned_upload_chunk_size() -> usize {
    return *PARTITIONED_UPLOAD_CHUNK_SIZE;

    static PARTITIONED_UPLOAD_CHUNK_SIZE: LazyLock<usize> = LazyLock::new(|| {
        let mut v: usize = 6 * 1024 * 1024;

        if let Ok(s) = std::env::var("POLARS_PARTITIONED_UPLOAD_CHUNK_SIZE") {
            v = s.parse::<usize>().unwrap_or_else(|_| {
                panic!("invalid value for POLARS_PARTITIONED_UPLOAD_CHUNK_SIZE: {s}")
            })
        }

        if verbose() {
            eprintln!("partitioned_upload_chunk_size: {v}")
        }

        v
    });
}

/// Max concurrent tasks within a single cloud writer.
pub fn upload_concurrency() -> NonZeroUsize {
    return *UPLOAD_CONCURRENCY;

    static UPLOAD_CONCURRENCY: LazyLock<NonZeroUsize> = LazyLock::new(|| {
        let buffer_limit: usize = (total_memory() / 32) as _;

        let mut v: NonZeroUsize =
            NonZeroUsize::new(usize::clamp(buffer_limit / upload_chunk_size(), 8, 256)).unwrap();

        if let Ok(s) = std::env::var("POLARS_UPLOAD_CONCURRENCY") {
            v = s
                .parse::<NonZeroUsize>()
                .unwrap_or_else(|_| panic!("invalid value for POLARS_UPLOAD_CONCURRENCY: {s}"))
        }

        if verbose() {
            eprintln!("upload_concurrency: {v}")
        }

        v
    });
}

pub fn partitioned_upload_concurrency() -> NonZeroUsize {
    return *PARTITIONED_UPLOAD_CONCURRENCY;

    static PARTITIONED_UPLOAD_CONCURRENCY: LazyLock<NonZeroUsize> = LazyLock::new(|| {
        let mut v: NonZeroUsize = NonZeroUsize::new(64).unwrap();

        if let Ok(s) = std::env::var("POLARS_PARTITIONED_UPLOAD_CONCURRENCY") {
            v = s.parse::<NonZeroUsize>().unwrap_or_else(|_| {
                panic!("invalid value for POLARS_PARTITIONED_UPLOAD_CONCURRENCY: {s}")
            })
        }

        if verbose() {
            eprintln!("partitioned_upload_concurrency: {v}")
        }

        v
    });
}

/// Runs of this many values whose total bytes are <= `copy_buffer_reserve_size` will be copied into
/// a single contiguous chunk.
pub(crate) fn cloud_writer_coalesce_run_length() -> usize {
    return *COALESCE_RUN_LENGTH;

    static COALESCE_RUN_LENGTH: LazyLock<usize> = LazyLock::new(|| {
        let mut v: usize = 64;

        if let Ok(s) = std::env::var("POLARS_CLOUD_WRITER_COALESCE_RUN_LENGTH") {
            v = s
                .parse::<usize>()
                .ok()
                .filter(|x| *x >= 2)
                .unwrap_or_else(|| {
                    panic!("invalid value for POLARS_CLOUD_WRITER_COALESCE_RUN_LENGTH: {s}")
                })
        }

        if polars_core::config::verbose() {
            eprintln!("cloud_writer coalesce_run_length: {v}")
        }

        v
    });
}

pub(crate) fn cloud_writer_copy_buffer_size() -> NonZeroUsize {
    return *COPY_BUFFER_SIZE;

    static COPY_BUFFER_SIZE: LazyLock<NonZeroUsize> = LazyLock::new(|| {
        let mut v: NonZeroUsize = const { NonZeroUsize::new(16 * 1024 * 1024).unwrap() };

        if let Ok(s) = std::env::var("POLARS_CLOUD_WRITER_COPY_BUFFER_SIZE") {
            v = s.parse::<NonZeroUsize>().unwrap_or_else(|_| {
                panic!("invalid value for POLARS_CLOUD_WRITER_COPY_BUFFER_SIZE: {s}")
            })
        }

        if polars_core::config::verbose() {
            eprintln!("cloud_writer copy_buffer_size: {v}")
        }

        v
    });
}
