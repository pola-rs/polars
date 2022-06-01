use crate::prelude::*;
use polars_core::prelude::*;

impl LazyFrame {
    pub fn anonymous_scan<F>(function: F, options: Option<AnonymousScanOptions>) -> Result<Self>
    where
        F: 'static + Fn(AnonymousScanOptions) -> Result<DataFrame> + Send + Sync,
    {

        let f = Arc::new(function);
        let options = options.unwrap_or_default();
        let lf: LazyFrame = LogicalPlanBuilder::anonymous_scan(f, options)?.build().into();
        Ok(lf)
    }
}
