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
    pub(crate) out_name: Option<Arc<str>>,
    pub(crate) options: RollingGroupOptions,
    pub(crate) expr: Expr,
}

impl PhysicalExpr for RollingExpr {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let groups_key = format!("{:?}", &self.options);

        let mut groups_map = state.group_tuples.read().unwrap();
        // Groups must be set by expression runner.
        let groups = groups_map.get(&groups_key);

        let mut groups_map_write;

        // There can be multiple rolling expressions in a single expr.
        // E.g. `min().rolling() + max().rolling()`
        // So if we hit that we will compute them here.
        let groups = match groups {
            Some(groups) => groups,
            None => {
                drop(groups_map);
                let (_time_key, _keys, groups) = df.group_by_rolling(vec![], &self.options)?;
                groups_map_write = state.group_tuples.write().unwrap();
                groups_map_write.entry_ref(&groups_key).or_insert(groups);

                drop(groups_map_write);

                // Get a reference to the read guard so that other threads
                // can continue
                groups_map = state.group_tuples.read().unwrap();
                groups_map.get(&groups_key).expect("impl error")
            },
        };

        let mut out = self
            .phys_function
            .evaluate_on_groups(df, groups, state)?
            .finalize();
        polars_ensure!(out.len() == groups.len(), agg_len = out.len(), groups.len());
        if let Some(name) = &self.out_name {
            out.rename(name.as_ref());
        }
        Ok(out)
    }

    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        _groups: &'a GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        polars_bail!(InvalidOperation: "rolling expression not allowed in aggregation");
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.function.to_field(input_schema, Context::Default)
    }

    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }
}
