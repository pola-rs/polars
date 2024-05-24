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
}
