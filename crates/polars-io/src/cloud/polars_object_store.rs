use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use polars_error::{to_compute_err, PolarsResult};

use crate::pl_async::{
    tune_with_concurrency_budget, with_concurrency_budget, MAX_BUDGET_PER_REQUEST,
};

/// Polars specific wrapper for `Arc<dyn ObjectStore>` that limits the number of
/// concurrent requests for the entire application.
#[derive(Debug, Clone)]
pub struct PolarsObjectStore(Arc<dyn ObjectStore>);

impl PolarsObjectStore {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self(store)
    }

    pub async fn get(&self, path: &Path) -> PolarsResult<Bytes> {
        tune_with_concurrency_budget(1, || async {
            self.0
                .get(path)
                .await
                .map_err(to_compute_err)?
                .bytes()
                .await
                .map_err(to_compute_err)
        })
        .await
    }

    pub async fn get_range(&self, path: &Path, range: Range<usize>) -> PolarsResult<Bytes> {
        tune_with_concurrency_budget(1, || self.0.get_range(path, range))
            .await
            .map_err(to_compute_err)
    }

    pub async fn get_ranges(
        &self,
        path: &Path,
        ranges: &[Range<usize>],
    ) -> PolarsResult<Vec<Bytes>> {
        tune_with_concurrency_budget(
            (ranges.len() as u32).clamp(0, MAX_BUDGET_PER_REQUEST as u32),
            || self.0.get_ranges(path, ranges),
        )
        .await
        .map_err(to_compute_err)
    }

    /// Fetch the metadata of the parquet file, do not memoize it.
    pub async fn head(&self, path: &Path) -> PolarsResult<ObjectMeta> {
        with_concurrency_budget(1, || self.0.head(path))
            .await
            .map_err(to_compute_err)
    }
}
