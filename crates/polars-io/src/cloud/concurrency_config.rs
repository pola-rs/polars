use crate::pl_async::{
    get_download_chunk_size, get_random_access_chunk_size, get_streaming_chunk_size,
};

/// Determines how in-flight concurrency for access to the back-end store is handled.
#[derive(Clone, Debug, Copy)]
pub enum ConcurrencyStrategy {
    /// (Almost) no in-flight concurrency control.
    /// Warning: this may result in an unbounded API call rate. Only use in the
    /// context of a rate-limited pipeline.
    Unbounded,
    /// In-flight concurrency control using a semi-static count-based budget.
    /// NOTE: This is a legacy strategy which does not scale up to the full potential.
    Legacy,
    /// In-flight concurrency control using a dynamically sensed bytes-budget, backed by
    /// a count-budget as fallback.
    BytesBased,
}

#[derive(Clone, Copy, Debug)]
pub struct FetchConfig {
    pub chunk_size: usize,
    pub strategy: ConcurrencyStrategy,
}

impl FetchConfig {
    /// Use for file formats that are randomly accessible, i.e. individual
    /// row groups (or record batches) and/or individual columns can be fetched
    /// directly using the metadata as an input. Example: Parquet, IPC with internal
    /// extensions.
    ///
    /// The chunk_size should be smaller to enable smooth operation of the
    /// bytes-based in-flight concurrency controller.
    pub fn random_access() -> Self {
        Self {
            chunk_size: get_random_access_chunk_size(),
            strategy: ConcurrencyStrategy::BytesBased,
        }
    }

    /// Use for file formats that have a sequential layout, i.e. the file bytes
    /// must be fetched and parsed sequentially. The pipeline is responsible for
    /// managing back-pressure and rate-limiting. Example: CSV.
    pub fn streaming() -> Self {
        Self {
            chunk_size: get_streaming_chunk_size(),
            // TODO: For now - keep as Legacy. Switch to Unbounded in a future PR.
            strategy: ConcurrencyStrategy::Legacy,
        }
    }

    /// Used for legacy fetch.
    /// @TODO: Deprecate over time.
    pub fn legacy() -> Self {
        Self {
            chunk_size: get_download_chunk_size(),
            strategy: ConcurrencyStrategy::Legacy,
        }
    }
}
