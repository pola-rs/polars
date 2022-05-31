use crate::prelude::AnonymousScanOptions;
use polars_core::prelude::*;
use std::fmt::{Debug, Formatter};

pub trait AnonymousScan: Send + Sync {
    fn scan(&self, scan_opts: AnonymousScanOptions) -> Result<DataFrame>;
}

impl<F> AnonymousScan for F
where
    F: Fn(AnonymousScanOptions) -> Result<DataFrame> + Send + Sync,
{
    fn scan(&self, scan_opts: AnonymousScanOptions) -> Result<DataFrame> {
        self(scan_opts.clone())
    }
}

impl Debug for dyn AnonymousScan {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "anonymous_scan")
    }
}
