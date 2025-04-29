use polars_core::prelude::*;

use crate::prelude::*;

impl LazyFrame {
    /// Get a dot language representation of the LogicalPlan.
    pub fn to_dot(&self, optimized: bool) -> PolarsResult<String> {
        let lp = if optimized {
            self.clone().to_alp_optimized()
        } else {
            self.clone().to_alp()
        }?;

        Ok(lp.display_dot().to_string())
    }

    /// Get a dot language representation of the physical plan.
    pub fn to_dot_phys(&self, optimized: bool, engine: Engine) -> PolarsResult<String> {
        let mut lf = self.clone();
        lf = match engine {
            Engine::Streaming => lf.with_new_streaming(true),
            _ => lf,
        };

        let mut lp = if optimized {
            lf.to_alp_optimized()
        } else {
            lf.to_alp()
        }?;

        match engine {
            Engine::Streaming => polars_stream::visualize_physical_plan(
                lp.lp_top,
                &mut lp.lp_arena,
                &mut lp.expr_arena,
            ),
            Engine::Auto => polars_bail!(ComputeError: "no engine selected"),
            Engine::OldStreaming => {
                polars_bail!(ComputeError: "old streaming engine is not supported")
            },
            Engine::InMemory => polars_bail!(ComputeError: "in-memory engine is not supported"),
            Engine::Gpu => polars_bail!(ComputeError: "gpu engine is not supported"),
        }
    }
}
