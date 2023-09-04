use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt};
use opendal::Operator;
use polars_core::prelude::{ArrowSchema, Schema};
use polars_error::{polars_bail, polars_ensure, to_compute_err, PolarsResult};

#[cfg(feature = "avro")]
pub mod avro;
#[cfg(feature = "csv")]
pub mod csv;
#[cfg(any(feature = "ipc", feature = "ipc_streaming"))]
pub mod ipc;
#[cfg(feature = "json")]
pub mod ndjson;
#[cfg(feature = "parquet")]
pub mod parquet;

pub trait FileFormatOptions {}

/// Default max records to scan to infer the schema
const DEFAULT_SCHEMA_INFER_MAX_RECORD: usize = 1000;

/// The number of files to read in parallel when inferring schema
const SCHEMA_INFERENCE_CONCURRENCY: usize = 32;

#[async_trait]
pub trait FileFormat: Send + Sync + std::fmt::Debug {
    /// Infer the common schema of the provided objects.
    /// For more than one file, the schema of all the files must be merge-able if `strict_schema == true`
    /// or else this might fail.
    /// The implementations handle whether the schema inference from either file metadata or from its content.
    async fn infer_schema_async(
        &self,
        store_op: &Operator,
        objects: Vec<String>,
        strict_schema: bool,
    ) -> PolarsResult<Schema> {
        polars_ensure!(!objects.is_empty(), NoData: "at least one path must be provided to infer schema");

        let schemas: Vec<_> = futures::stream::iter(objects)
            .map(|object| self.fetch_schema_async(store_op, object))
            .boxed() // Workaround https://github.com/rust-lang/rust/issues/64552
            .buffered(SCHEMA_INFERENCE_CONCURRENCY)
            .try_collect()
            .await?;

        self.handle_schema(schemas, strict_schema)
    }

    /// Read and parse the schema of the Avro file at location `path`
    async fn fetch_schema_async(
        &self,
        store_op: &Operator,
        path: String,
    ) -> PolarsResult<ArrowSchema>;

    // fn read(
    //     &self,
    //     n_rows: Option<usize>,
    //     columns: Option<Vec<String>>,
    //     predicate: Option<Arc<dyn PhysicalIoExpr>>,
    //     projection: Option<Vec<usize>>,
    //     options: O,
    // ) -> PolarsResult<DataFrame>;
    //
    // fn get_batches(&self) -> PolarsResult<Vec<DataFrame>> {
    //     polars_bail!(ComputeError: "Functionality `get_batches` is currently not supported.")
    // }

    fn handle_schema(
        &self,
        schemas: Vec<ArrowSchema>,
        strict_schema: bool,
    ) -> PolarsResult<Schema> {
        let schema = if strict_schema {
            let s = schemas
                .windows(2)
                .all(|a| a[0] == a[1])
                .then(|| &schemas[0])
                .ok_or(to_compute_err("Schemas of all files must match."))?;

            Schema::from_iter(s.clone().fields.iter())
        } else {
            let mut default_schema = Schema::default();
            for s in schemas {
                default_schema.merge(Schema::from_iter(s.fields.iter()))
            }

            default_schema
        };

        Ok(schema)
    }
}
