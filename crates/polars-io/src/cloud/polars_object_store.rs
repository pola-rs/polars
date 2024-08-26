use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use polars_error::{to_compute_err, PolarsResult};
use tokio::io::AsyncWriteExt;

use crate::pl_async::{
    self, tune_with_concurrency_budget, with_concurrency_budget, MAX_BUDGET_PER_REQUEST,
};

/// Polars specific wrapper for `Arc<dyn ObjectStore>` that limits the number of
/// concurrent requests for the entire application.
#[derive(Debug, Clone)]
pub struct PolarsObjectStore(Arc<dyn ObjectStore>);
pub type ObjectStorePath = object_store::path::Path;

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

    pub async fn download<F: tokio::io::AsyncWrite + std::marker::Unpin>(
        &self,
        path: &Path,
        file: &mut F,
    ) -> PolarsResult<()> {
        tune_with_concurrency_budget(1, || async {
            let mut stream = self
                .0
                .get(path)
                .await
                .map_err(to_compute_err)?
                .into_stream();

            let mut len = 0;
            while let Some(bytes) = stream.next().await {
                let bytes = bytes.map_err(to_compute_err)?;
                len += bytes.len();
                file.write(bytes.as_ref()).await.map_err(to_compute_err)?;
            }

            PolarsResult::Ok(pl_async::Size::from(len as u64))
        })
        .await?;
        Ok(())
    }

    /// Fetch the metadata of the parquet file, do not memoize it.
    pub async fn head(&self, path: &Path) -> PolarsResult<ObjectMeta> {
        with_concurrency_budget(1, || async {
            let head_result = self.0.head(path).await;

            if head_result.is_err() {
                // Pre-signed URLs forbid the HEAD method, but we can still retrieve the header
                // information with a range 0-0 request.
                let get_range_0_0_result = self
                    .0
                    .get_opts(
                        path,
                        object_store::GetOptions {
                            range: Some((0..1).into()),
                            ..Default::default()
                        },
                    )
                    .await;

                if let Ok(v) = get_range_0_0_result {
                    return Ok(v.meta);
                }
            }

            head_result
        })
        .await
        .map_err(to_compute_err)
    }
}
