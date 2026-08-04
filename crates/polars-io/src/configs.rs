use std::num::NonZeroUsize;
use std::sync::LazyLock;

pub fn env_upload_chunk_size() -> Option<NonZeroUsize> {
    std::env::var("POLARS_UPLOAD_CHUNK_SIZE").ok().map(|s| {
        s.parse::<NonZeroUsize>()
            .unwrap_or_else(|_| panic!("invalid value for POLARS_UPLOAD_CHUNK_SIZE: {s}"))
    })
}

pub fn env_partitioned_upload_chunk_size() -> Option<NonZeroUsize> {
    std::env::var("POLARS_PARTITIONED_UPLOAD_CHUNK_SIZE")
        .ok()
        .map(|s| {
            s.parse::<NonZeroUsize>().unwrap_or_else(|_| {
                panic!("invalid value for POLARS_PARTITIONED_UPLOAD_CHUNK_SIZE: {s}")
            })
        })
}

/// Max concurrent tasks within a single cloud writer.
pub fn env_upload_concurrency() -> Option<NonZeroUsize> {
    std::env::var("POLARS_UPLOAD_CONCURRENCY").ok().map(|s| {
        s.parse::<NonZeroUsize>()
            .unwrap_or_else(|_| panic!("invalid value for POLARS_UPLOAD_CONCURRENCY: {s}"))
    })
}

pub fn env_partitioned_upload_concurrency() -> Option<NonZeroUsize> {
    std::env::var("POLARS_PARTITIONED_UPLOAD_CONCURRENCY")
        .ok()
        .map(|s| {
            s.parse::<NonZeroUsize>().unwrap_or_else(|_| {
                panic!("invalid value for POLARS_PARTITIONED_UPLOAD_CONCURRENCY: {s}")
            })
        })
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
