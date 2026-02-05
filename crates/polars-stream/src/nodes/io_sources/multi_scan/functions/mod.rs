use std::num::NonZeroUsize;

use polars_utils::slice_enum::Slice;

pub mod resolve_projections;
pub mod resolve_slice;

pub fn calc_n_readers_pre_init(
    num_pipelines: usize,
    num_sources: usize,
    pre_slice: Option<&Slice>,
) -> usize {
    if let Ok(v) = std::env::var("POLARS_NUM_READERS_PRE_INIT").map(|x| {
        x.parse::<NonZeroUsize>()
            .expect("invalid value for POLARS_NUM_READERS_PRE_INIT: {x}")
            .get()
    }) {
        return v;
    }

    let max_files_with_slice = match pre_slice {
        // Calculate the max number of files assuming 1 row per file.
        Some(v @ Slice::Positive { .. }) => v.end_position().max(1),
        Some(Slice::Negative { .. }) | None => usize::MAX,
    };

    // Set this generously high, there are users who scan 10,000's of small files from the cloud.
    num_pipelines
        .saturating_add(3)
        .min(max_files_with_slice)
        .min(num_sources)
        .clamp(1, 128)
}

pub fn calc_max_concurrent_scans(num_pipelines: usize, num_sources: usize) -> usize {
    if let Ok(v) = std::env::var("POLARS_MAX_CONCURRENT_SCANS").map(|x| {
        x.parse::<NonZeroUsize>()
            .expect("invalid value for POLARS_MAX_CONCURRENT_SCANS: {x}")
            .get()
    }) {
        return v;
    }

    num_pipelines.min(num_sources).clamp(1, 128)
}
