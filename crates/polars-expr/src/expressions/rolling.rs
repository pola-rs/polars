use polars_time::{ClosedWindow, Duration, PolarsTemporalGroupby, RollingGroupOptions};

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
    pub(crate) phys_function: Arc<dyn PhysicalExpr>,
    pub(crate) index_column: Arc<dyn PhysicalExpr>,
    pub(crate) period: Duration,
    pub(crate) offset: Duration,
    pub(crate) closed_window: ClosedWindow,
    pub(crate) expr: Expr,
    pub(crate) output_field: Field,
}

impl PhysicalExpr for RollingExpr {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let groups = if let Some(index_column_name) = self.index_column.as_column() {
            let options = RollingGroupOptions {
                index_column: index_column_name.clone(),
                period: self.period,
                offset: self.offset,
                closed_window: self.closed_window,
            };
            let groups_key = format!("{options:?}");
            let groups = {
                // Groups must be set by expression runner.
                state.window_cache.get_groups(&groups_key)
            };

            // There can be multiple rolling expressions in a single expr.
            // E.g. `min().rolling() + max().rolling()`
            // So if we hit that we will compute them here.
            match groups {
                Some(groups) => groups,
                None => {
                    let (_time_key, groups) = df.rolling(None, &options)?;
                    state.window_cache.insert_groups(groups_key, groups.clone());
                    groups
                },
            }
        } else {
            let index_column_name = PlSmallStr::from_static("__PL_INDEX_COL");
            let options = RollingGroupOptions {
                index_column: index_column_name.clone(),
                period: self.period,
                offset: self.offset,
                closed_window: self.closed_window,
            };

            let index_column = self.index_column.evaluate(df, state)?;

            let mut df = df.clone();
            df.with_column(index_column.with_name(index_column_name))?;
            let (_time_key, groups) = df.rolling(None, &options)?;
            groups
        };

        let out = self
            .phys_function
            .evaluate_on_groups(df, &groups, state)?
            .finalize();
        polars_ensure!(out.len() == groups.len(), agg_len = out.len(), groups.len());
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

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn is_scalar(&self) -> bool {
        false
    }
}
