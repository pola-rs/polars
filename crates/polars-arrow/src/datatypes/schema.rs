use std::sync::Arc;

use super::Field;

/// An ordered sequence of [`Field`]s
///
/// [`ArrowSchema`] is an abstraction used to read from, and write to, Arrow IPC format,
/// Apache Parquet, and Apache Avro. All these formats have a concept of a schema
/// with fields and metadata.
pub type ArrowSchema = polars_schema::Schema<Field>;
pub type ArrowSchemaRef = Arc<ArrowSchema>;
