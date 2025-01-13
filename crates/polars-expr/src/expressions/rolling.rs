use polars_time::{PolarsTemporalGroupby, RollingGroupOptions};

use super::*;

pub(crate) struct RollingExpr {
    /// the root column that the Function will be applied on.
    /// This will be used to create a smaller DataFrame to prevent taking unneeded columns by index
    /// TODO! support keys?
    /// The challenge is that the group_by will reorder the results and the
    /// keys, and time index would need to be updated, or the result should be joined back
    /// For now, don't support it.
    ///
    /// A function Expr. i.e. Mean, Median, Max, etc.
    pub(crate) function: Expr,
    pub(crate) phys_function: Arc<dyn PhysicalExpr>,
    pub(crate) out_name: Option<PlSmallStr>,
    pub(crate) options: RollingGroupOptions,
    pub(crate) expr: Expr,
}

impl PhysicalExpr for RollingExpr {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let groups_key = format!("{:?}", &self.options);

        let groups = {
            // Groups must be set by expression runner.
            state.window_cache.get_groups(&groups_key).clone()
        };

        // There can be multiple rolling expressions in a single expr.
        // E.g. `min().rolling() + max().rolling()`
        // So if we hit that we will compute them here.
        let groups = match groups {
            Some(groups) => groups,
            None => {
                let (_time_key, _keys, groups) = df.rolling(vec![], &self.options)?;
                state.window_cache.insert_groups(groups_key, groups.clone());
                groups
            },
        };

        let mut out = self
            .phys_function
            .evaluate_on_groups(df, &groups, state)?
            .finalize();
        polars_ensure!(out.len() == groups.len(), agg_len = out.len(), groups.len());
        if let Some(name) = &self.out_name {
            out.rename(name.clone());
        }
        Ok(out.into_column())
    }

    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        _groups: &'a GroupPositions,
        _state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        polars_bail!(InvalidOperation: "rolling expression not allowed in aggregation");
    }

    fn collect_live_columns(&self, lv: &mut PlIndexSet<PlSmallStr>) {
        self.phys_function.collect_live_columns(lv);
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.function.to_field(input_schema, Context::Default)
    }

    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn is_scalar(&self) -> bool {
        false
    }
}
