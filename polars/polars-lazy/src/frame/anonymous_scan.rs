use crate::prelude::*;
use polars_core::prelude::*;

impl LazyFrame {
    pub fn anonymous_scan<F>(function: F, options: AnonymousScanOptions) -> Self where F: 'static + Fn(AnonymousScanOptions) -> Result<DataFrame> + Send + Sync,
    {
        LogicalPlan::AnonymousScan {
            options,
            function: Arc::new(function)
        }
        .into()
    }
}
