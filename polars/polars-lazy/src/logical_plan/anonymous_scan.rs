use crate::prelude::AnonymousScanOptions;
use polars_core::prelude::*;
use std::fmt::{Debug, Formatter};

pub trait AnonymousScan: Send + Sync {
    fn scan(&self, scan_opts: AnonymousScanOptions) -> Result<DataFrame>;
    fn schema(&self, _infer_schema_length: Option<usize>) -> Result<Schema> {
        Err(PolarsError::ComputeError(
            "Must supply either a schema or a schema function".into(),
        ))
    }
}

impl<F> AnonymousScan for F
where
    F: Fn(AnonymousScanOptions) -> Result<DataFrame> + Send + Sync,
{
    fn scan(&self, scan_opts: AnonymousScanOptions) -> Result<DataFrame> {
        self(scan_opts)
    }
}

impl Debug for dyn AnonymousScan {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "anonymous_scan")
    }
}
