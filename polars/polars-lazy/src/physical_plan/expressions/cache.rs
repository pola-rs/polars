use super::*;

pub struct CacheExpr {
    pub(crate) physical_expr: Arc<dyn PhysicalExpr>,
    expr: Expr,
    id: usize,
}

impl CacheExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, expr: Expr, id: usize) -> Self {
        Self {
            physical_expr,
            expr,
            id,
        }
    }
}

impl PhysicalExpr for CacheExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        if let Some(cached) = state.get_expr_cache(self.id) {
            let mut hit = true;
            let out = cached
                .get_or_try_init(|| {
                    hit = false;
                    self.physical_expr.evaluate(df, state)
                })
                .cloned();
            if state.verbose() {
                if hit {
                    eprintln!("cache hit: {:?}", self.expr)
                } else {
                    eprintln!("cache miss: {:?}", self.expr)
                }
            }
            out
        } else {
            self.physical_expr.evaluate(df, state)
        }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        if let Some(cached) = state.get_expr_cache(self.id) {
            let mut hit = true;
            let aggregated = cached
                .get_or_try_init(|| {
                    let mut agg = self.physical_expr.evaluate_on_groups(df, groups, state)?;
                    hit = false;
                    PolarsResult::Ok(agg.aggregated())
                })?
                .clone();
            if state.verbose() {
                if hit {
                    eprintln!("cache hit: {:?}", self.expr)
                } else {
                    eprintln!("cache miss: {:?}", self.expr)
                }
            }
            Ok(AggregationContext::new(
                aggregated,
                Cow::Borrowed(groups),
                true,
            ))
        } else {
            self.physical_expr.evaluate_on_groups(df, groups, state)
        }
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.physical_expr.to_field(input_schema)
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        None
    }
    fn is_valid_aggregation(&self) -> bool {
        self.physical_expr.is_valid_aggregation()
    }
}
