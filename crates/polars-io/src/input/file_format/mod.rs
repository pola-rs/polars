use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt};
use opendal::Operator;
use polars_core::prelude::{DataFrame, PlHashMap, Schema};
use polars_error::PolarsResult;

use crate::input::file_listing::ObjectListingUrl;
use crate::input::try_blocking_io;
use crate::predicates::PhysicalIoExpr;

#[cfg(feature = "parquet")]
pub mod parquet;

pub type ObjectInfo = (String, Schema, (Option<usize>, usize));

pub trait FileFormatOptions {}

/// The number of objects to read in parallel when inferring schema
const SCHEMA_INFERENCE_CONCURRENCY: usize = 32;

pub type DynFileFormat = dyn FileFormat;

#[async_trait]
pub trait FileFormat: std::fmt::Display + Send + Sync + Debug + 'static {
    /// To instantiate
    fn create() -> Self;

    /// Uses a size hint obtained from the reader to produce,
    ///  - Known row count (may or may not be known)
    ///  - Estimated row count (can be calculated from reader hints)
    fn calculate_rows_count(
        &self,
        reader_size_hint: (usize, Option<usize>),
    ) -> (Option<usize>, usize);

    /// Globs object info (path, schema, size_hint) for a given set of options and
    /// a base [ObjectListingUrl].
    /// Operator to connect to the remote store is inferred internally.
    ///
    /// This is a sync API but runs the tasks for each child in an async manner internally  
    /// and blocks till all tasks are successfully completed.
    fn glob_object_info(
        &self,
        listing_url: ObjectListingUrl,
        cloud_opts: PlHashMap<String, String>,
        exclude_empty: bool,
        recursive: bool,
    ) -> PolarsResult<Vec<ObjectInfo>> {
        try_blocking_io(async {
            let url = listing_url.clone();
            let operator = url
                .infer_operator(cloud_opts)
                .expect("failed to create an operator for remote store");

            let objects = url
                .glob_object_list(&operator, "", exclude_empty, recursive)
                .await
                .expect("failed to glob objects from remote store");

            futures::stream::iter(objects)
                .map(|(path, _)| async { self.get_object_info(&operator, path).await })
                .buffer_unordered(SCHEMA_INFERENCE_CONCURRENCY)
                .try_collect::<Vec<_>>()
                .await
                .expect("failed to get info for one or more objects")
        })
    }

    /// Fetches metadata of an object from the provided `path` and returns the results as
    /// object info (path, schema, size_hint).
    ///
    /// The [Schema] is inferred from the format specific metadata.
    async fn get_object_info(&self, operator: &Operator, path: String) -> PolarsResult<ObjectInfo>;

    fn finish_read(
        &self,
        _n_rows: Option<usize>,
        _columns: Option<Vec<String>>,
        _predicate: Option<Arc<dyn PhysicalIoExpr>>,
        _projection: Option<Vec<usize>>,
    ) -> PolarsResult<DataFrame> {
        todo!()
    }

    fn get_batches(&self) -> PolarsResult<Vec<DataFrame>> {
        todo!()
    }
}
