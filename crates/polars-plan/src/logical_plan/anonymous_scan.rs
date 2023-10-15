use std::any::Any;
use std::fmt::{Debug, Formatter};

use polars_core::prelude::*;

pub use super::options::AnonymousScanOptions;
use crate::dsl::Expr;

pub struct AnonymousScanArgs {
    pub n_rows: Option<usize>,
    pub with_columns: Option<Arc<Vec<String>>>,
    pub schema: SchemaRef,
    pub output_schema: Option<SchemaRef>,
    pub predicate: Option<Expr>,
}

pub trait AnonymousScan: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    /// Creates a dataframe from the supplied function & scan options.
    fn scan(&self, scan_opts: AnonymousScanArgs) -> PolarsResult<DataFrame>;

    /// function to supply the schema.
    /// Allows for an optional infer schema argument for data sources with dynamic schemas
    fn schema(&self, _infer_schema_length: Option<usize>) -> PolarsResult<SchemaRef> {
        polars_bail!(ComputeError: "must supply either a schema or a schema function");
    }
    /// specify if the scan provider should allow predicate pushdowns
    ///
    /// Defaults to `false`
    fn allows_predicate_pushdown(&self) -> bool {
        false
    }
    /// specify if the scan provider should allow projection pushdowns
    ///
    /// Defaults to `false`
    fn allows_projection_pushdown(&self) -> bool {
        false
    }
    /// specify if the scan provider should allow slice pushdowns
    ///
    /// Defaults to `false`
    fn allows_slice_pushdown(&self) -> bool {
        false
    }
}

impl<F> AnonymousScan for F
where
    F: Fn(AnonymousScanArgs) -> PolarsResult<DataFrame> + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        unimplemented!()
    }

    fn scan(&self, scan_opts: AnonymousScanArgs) -> PolarsResult<DataFrame> {
        self(scan_opts)
    }
}

impl Debug for dyn AnonymousScan {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "anonymous_scan")
    }
}
