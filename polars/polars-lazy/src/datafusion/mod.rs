mod conversion;
use crate::prelude::*;
use conversion::to_datafusion_lp;
use datafusion::physical_plan::collect;
use datafusion::prelude::{ExecutionConfig, ExecutionContext};
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use std::convert::TryFrom;
use tokio::runtime::Runtime;

impl LazyFrame {
    /// Collect Out of Core on the DataFusion query engine
    pub fn ooc(self) -> Result<DataFrame> {
        // Don't use Polars optimizer, but transpile and send to DataFusion
        let lp = self.logical_plan;
        let lp = to_datafusion_lp(lp)?;

        let ctx = ExecutionContext::with_config(ExecutionConfig::new().with_concurrency(8));
        let lp = ctx.optimize(&lp).unwrap();
        let pp = ctx.create_physical_plan(&lp).unwrap();

        let rt = Runtime::new().unwrap();
        let rbs = rt.block_on(collect(pp)).unwrap();

        let dfs = rbs.into_iter().map(|rb| DataFrame::try_from(rb).unwrap());
        accumulate_dataframes_vertical(dfs)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use polars_core::df;

    #[test]
    fn test_datafusion_query() -> Result<()> {
        // TODO! run same with string column, needs sort large-utf8 support from arrow.
        let df = df! {
            "a" => [1, 1, 1, 2, 2, 3],
            "b" => [1, 2, 3, 4, 5, 6]
        }?;

        let out = df
            .lazy()
            .groupby(vec![col("a")])
            .agg(vec![col("b").mean()])
            .select(vec![col("a"), col("b_mean")])
            .sort("a", false)
            .ooc()?;

        assert_eq!(
            Vec::from(out.column("b_mean")?.f64()?),
            &[Some(2.0), Some(4.5), Some(6.0)]
        );
        Ok(())
    }
}
