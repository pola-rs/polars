use std::any::Any;
use std::fmt::{Debug, Formatter};

use polars_core::prelude::*;

use crate::dsl::Expr;

/// Arguments passed to `AnonymousScan::scan()` and `AnonymousScan::next_batch()`.
#[derive(Clone)]
pub struct AnonymousScanArgs {
    pub n_rows: Option<usize>,
    pub with_columns: Option<Arc<[PlSmallStr]>>,
    pub schema: SchemaRef,
    pub output_schema: Option<SchemaRef>,
    pub predicate: Option<Expr>,
}

impl Default for AnonymousScanArgs {
    fn default() -> Self {
        Self {
            n_rows: None,
            with_columns: None,
            schema: Arc::new(Schema::default()),
            output_schema: None,
            predicate: None,
        }
    }
}

pub trait AnonymousScan: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    /// Creates a DataFrame from the supplied function & scan options.
    fn scan(&self, scan_opts: AnonymousScanArgs) -> PolarsResult<DataFrame>;

    /// function to supply the schema.
    /// Allows for an optional infer schema argument for data sources with dynamic schemas
    fn schema(&self, _infer_schema_length: Option<usize>) -> PolarsResult<SchemaRef> {
        polars_bail!(ComputeError: "must supply either a schema or a schema function");
    }
    /// Specify if the scan provider should allow predicate pushdowns.
    ///
    /// Defaults to `false`
    fn allows_predicate_pushdown(&self) -> bool {
        false
    }
    /// Specify if the scan provider should allow projection pushdowns.
    ///
    /// Defaults to `false`
    fn allows_projection_pushdown(&self) -> bool {
        false
    }

    /// Whether this scan supports streaming execution via `next_batch`.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Returns the next batch of data for streaming execution.
    ///
    /// # Arguments
    /// * `args` - Arguments containing projection (`with_columns`) and row limit (`n_rows`).
    ///   Implementations should use `args.with_columns` to return only the requested columns
    ///   (projection pushdown) and respect `args.n_rows` for early termination.
    ///
    /// # Returns
    /// * `Ok(Some(df))` - The next batch of data
    /// * `Ok(None)` - No more data available (stream exhausted)
    fn next_batch(&self, _args: &AnonymousScanArgs) -> PolarsResult<Option<DataFrame>> {
        polars_bail!(ComputeError: "streaming not supported for this scan");
    }
}

impl Debug for dyn AnonymousScan {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "anonymous_scan")
    }
}
