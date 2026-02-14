use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_io::metrics::IOMetrics;
use polars_io::utils::byte_source::{ByteSource, DynByteSourceBuilder};
use polars_io::utils::compression::SupportedCompression;
use polars_plan::dsl::ScanSource;
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
            .unwrap_or_else(|_| panic!("invalid value for POLARS_NUM_READERS_PRE_INIT: {x}"))
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
            .unwrap_or_else(|_| panic!("invalid value for POLARS_MAX_CONCURRENT_SCANS: {x}"))
            .get()
    }) {
        return v;
    }

    num_pipelines.min(num_sources).clamp(1, 128)
}

pub async fn is_compressed_source(
    scan_source: ScanSource,
    cloud_options: Option<Arc<CloudOptions>>,
    io_metrics: Option<Arc<IOMetrics>>,
) -> PolarsResult<bool> {
    let byte_source_builder = if scan_source.is_cloud_url() || polars_config::config().force_async()
    {
        DynByteSourceBuilder::ObjectStore
    } else {
        DynByteSourceBuilder::Mmap
    };

    let byte_source = scan_source
        .as_scan_source_ref()
        .to_dyn_byte_source(&byte_source_builder, cloud_options.as_deref(), io_metrics)
        .await?;

    let Ok(first_4_bytes) = byte_source.get_range(0..4).await else {
        return Ok(false);
    };

    Ok(SupportedCompression::check(&first_4_bytes).is_some())
}
