//! Parquet footer selection and reading, shared by the two resolve passes.
//!
//! Both passes read footers, but for different reasons, and
//! `POLARS_RESOLVE_METADATA_LEVEL` governs each differently.
//!
//! *Statistics resolution* runs for an ordinary schema-inferred `scan_parquet`
//! ([`parquet_file_info`]). It reads source 0 for the schema and row count, and
//! the mode decides what else:
//! - `none`, `row_counts`: retain only source 0's footer.
//! - `sampled`: read heavy sources first, then fill the remaining budget with a
//!   stratified sample.
//! - `full`: read every footer.
//!
//! *Splitting resolution* ([`resolve_for_splitting`]) runs for schema-provided
//! scans and expanded datasets, whose schema comes from elsewhere. It reads
//! source 0 plus the heavy sources so a distributed engine can split them by
//! row group, and only under `sampled` or `full`; under `none` and
//! `row_counts` it retains no footers at all, since source 0 is not needed for
//! a schema. This pass never adds statistics sampling.
//!
//! [`parquet_file_info`]: crate::plans::conversion::dsl_to_ir::scans::parquet_file_info

use std::io::Cursor;
use std::num::NonZeroU32;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_core::config::verbose;
use polars_core::error::{PolarsResult, feature_gated};
use polars_io::parquet::metadata::FileMetadataRef;

use crate::dsl::MetadataPerSource;
use crate::prelude::{ScanSourceRef, ScanSources};

/// Minimum sample so a scan still extrapolates from enough files; below this,
/// two-mode datasets can miss a mode entirely and misestimate badly.
const SAMPLE_FLOOR: usize = 16;

/// Maximum footers to read, including source 0.
///
/// The default limit is the I/O concurrency budget, at least [`SAMPLE_FLOOR`].
/// `POLARS_RESOLVE_SAMPLE_LIMIT` overrides it, even below the floor.
fn footer_budget(n_sources: usize) -> usize {
    let limit = polars_config::config().resolve_sample_limit().map_or_else(
        || (polars_io::pl_async::get_concurrency_limit() as usize).max(SAMPLE_FLOOR),
        |o| o as usize,
    );
    sample_size(n_sources, limit)
}

/// Select sources containing at least `1 / n_parts` of the total bytes.
///
/// Keep the largest sources that fit `budget`, returning indices in source order.
/// Source 0 is read separately: it counts toward the budget but is not returned.
fn heavy_source_indices(bytes: &[u64], n_parts: NonZeroU32, budget: usize) -> Vec<usize> {
    let total: u128 = bytes.iter().map(|&b| b as u128).sum();
    if total == 0 {
        return Vec::new();
    }

    let n_parts = u128::from(n_parts.get());
    let mut indices: Vec<usize> = (1..bytes.len())
        .filter(|&i| u128::from(bytes[i]) * n_parts >= total)
        .collect();

    indices.sort_unstable_by_key(|&i| std::cmp::Reverse(bytes[i]));
    let heavy = indices.len();
    indices.truncate(budget.saturating_sub(1));
    if verbose() {
        eprintln!(
            "parquet resolve: pinned {} / {heavy} heavy sources (footer budget {budget})",
            indices.len(),
        );
    }

    indices.sort_unstable();
    indices
}

/// Footer-wave size (incl. file 0) for `ResolveMode::Sampled`: `sqrt(n)`,
/// floored at `SAMPLE_FLOOR`, capped at `limit`, never above the file count.
fn sample_size(n_sources: usize, limit: usize) -> usize {
    // `limit` last so it stays a hard ceiling.
    ((n_sources as f64).sqrt().ceil() as usize)
        .max(SAMPLE_FLOOR)
        .min(limit.max(1))
        .min(n_sources)
}

/// Evenly-strided sample of `k - 1` indices in `1..n_sources` (file 0 is
/// read separately).
fn sampled_source_indices(n_sources: usize, k: usize) -> Vec<usize> {
    if k <= 1 {
        return Vec::new();
    }
    let extra = k - 1;
    let span = n_sources - 1;
    (0..extra).map(|j| 1 + (j * span) / extra).collect()
}

/// Footers to read besides source 0, as ascending indices in `1..n_sources`.
///
/// Heavy sources come first. With `fill_sample`, the remaining budget goes to a
/// stratified sample of the sources not already selected; without it, only the
/// heavy sources are returned.
pub(crate) fn select_footer_indices(
    n_sources: usize,
    bytes_per_source: Option<&[u64]>,
    resolve_heavy_sources: Option<NonZeroU32>,
    fill_sample: bool,
) -> Vec<usize> {
    // Footer budget including source 0, which is read separately.
    let budget = footer_budget(n_sources);

    if fill_sample && budget >= n_sources {
        return (1..n_sources).collect();
    }

    // Prioritize heavy files for distributed row-group splitting.
    let mut indices = resolve_heavy_sources
        .zip(bytes_per_source)
        .map(|(n_parts, bytes)| {
            debug_assert_eq!(bytes.len(), n_sources);
            heavy_source_indices(bytes, n_parts, budget)
        })
        .unwrap_or_default();

    if !fill_sample {
        return indices;
    }

    // Use the remaining budget for a stratified sample.
    let left = budget - indices.len();
    if indices.is_empty() {
        return sampled_source_indices(n_sources, left);
    }
    if left > 1 {
        // Sample only unselected sources to avoid wasting the budget.
        let rest: Vec<usize> = (1..n_sources)
            .filter(|i| indices.binary_search(i).is_err())
            .collect();
        // Add a dummy source 0, then map sampled positions to source indices.
        indices.extend(
            sampled_source_indices(rest.len() + 1, left)
                .into_iter()
                .map(|pos| rest[pos - 1]),
        );
        indices.sort_unstable();
    }
    indices
}

/// Read the footers of `indices` in one concurrency wave.
///
/// Failed reads are dropped: the source simply stays unresolved.
pub(crate) async fn read_footers(
    sources: &ScanSources,
    indices: &[usize],
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> Vec<(usize, FileMetadataRef)> {
    let mut futures = indices
        .iter()
        .map(|&i| async move {
            (
                i,
                read_parquet_metadata(sources.at(i), cloud_options)
                    .await
                    .ok(),
            )
        })
        .collect::<FuturesUnordered<_>>();

    let mut pairs = Vec::with_capacity(indices.len());
    while let Some((i, metadata)) = futures.next().await {
        if let Some(metadata) = metadata {
            pairs.push((i, metadata));
        }
    }
    pairs
}

/// Resolve footers for distributed row-group splitting.
///
/// See the module docs: reads source 0 plus the heavy sources, and only under
/// `sampled` or `full`.
pub(crate) async fn resolve_for_splitting(
    sources: &ScanSources,
    bytes: &[u64],
    n_parts: NonZeroU32,
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> MetadataPerSource {
    use polars_config::ResolveMode;

    let n_sources = sources.len();
    // Empty tables and fully pruned datasets have no source 0.
    if n_sources == 0 {
        return MetadataPerSource::Unresolved;
    }
    if matches!(
        polars_config::config().resolve_metadata_level(),
        ResolveMode::None | ResolveMode::RowCounts
    ) {
        return MetadataPerSource::Unresolved;
    }
    debug_assert_eq!(bytes.len(), n_sources);

    // Partial metadata requires source 0, even if it is not heavy.
    // This also provides row groups for single-file datasets.
    let indices: Vec<usize> = std::iter::once(0)
        .chain(select_footer_indices(
            n_sources,
            Some(bytes),
            Some(n_parts),
            false,
        ))
        .collect();

    MetadataPerSource::new(
        read_footers(sources, &indices, cloud_options).await,
        n_sources,
    )
}

/// Read one source's full Parquet footer.
pub(crate) async fn read_parquet_metadata(
    source: ScanSourceRef<'_>,
    #[allow(unused)] cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<FileMetadataRef> {
    if source.is_cloud_url() {
        #[allow(unused)]
        let path = source.as_path().unwrap();
        feature_gated!("cloud", {
            let mut reader =
                polars_io::prelude::ParquetObjectStore::from_uri(path.clone(), cloud_options, None)
                    .await?;
            reader.get_metadata().await.cloned()
        })
    } else {
        let memslice = source.to_memslice()?;
        let mut cursor = Cursor::new(memslice);
        let md = polars_parquet::parquet::read::read_metadata(&mut cursor)?;
        Ok(std::sync::Arc::new(md))
    }
}

/// Fetch one source's `num_rows` (thrift field 3 only); skips
/// schema, row_groups, and the rest. Used in `RowCounts` resolve mode.
pub(crate) async fn read_parquet_num_rows(
    source: ScanSourceRef<'_>,
    #[allow(unused)] cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<i64> {
    if source.is_cloud_url() {
        #[allow(unused)]
        let path = source.as_path().unwrap();
        feature_gated!("cloud", {
            let mut reader =
                polars_io::prelude::ParquetObjectStore::from_uri(path.clone(), cloud_options, None)
                    .await?;
            reader.num_rows_only().await
        })
    } else {
        let memslice = source.to_memslice()?;
        let mut cursor = Cursor::new(memslice);
        polars_parquet::parquet::read::read_num_rows(&mut cursor).map_err(Into::into)
    }
}
