use crate::prelude::*;
use polars_core::prelude::*;

impl LazyFrame {
    pub fn anonymous_scan<F>(function: F, schema: Schema) -> Result<Self>
    where
        F: 'static + Fn(AnonymousScanOptions) -> Result<DataFrame> + Send + Sync,
    {
        let f = Arc::new(function);
        let lf: LazyFrame = LogicalPlanBuilder::anonymous_scan(f, schema)?.build().into();
        Ok(lf)
    }
}
