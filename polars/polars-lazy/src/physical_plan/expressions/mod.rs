pub mod default;
mod final_agg;

use crate::prelude::*;
pub use default::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use polars_io::PhysicalIoExpr;
use std::borrow::Cow;

/// Take a DataFrame and evaluate the expressions.
/// Implement this for Column, lt, eq, etc
pub trait PhysicalExpr: Send + Sync {
    fn as_expression(&self) -> &Expr {
        // for instance not needed for aggregations (for now)
        unimplemented!()
    }

    /// Take a DataFrame and evaluate the expression.
    fn evaluate(&self, df: &DataFrame) -> Result<Series>;

    /// Some expression that are not aggregations can be done per group
    /// Think of sort, slice, etc.
    ///
    /// defaults to ignoring the group
    // we allow this because we pass the vec to the Cow
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        self.evaluate(df).map(|s| (s, Cow::Borrowed(groups)))
    }

    /// Get the output field of this expr
    fn to_field(&self, input_schema: &Schema) -> Result<Field>;

    /// Convert to a aggregation expression.
    /// This can only be done for the final expressions that produce an aggregated result.
    ///
    /// The expression sum, min, max etc can be called as `evaluate` in the standard context,
    /// or during a groupby execution, this method is called to convert them to an AggPhysicalExpr
    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        let e = self.as_expression();
        Err(PolarsError::InvalidOperation(
            format!("{:?} is not an agg expression", e).into(),
        ))
    }
}

/// Wrapper struct that allow us to use a PhysicalExpr in polars-io.
///
/// This is used to filter rows during the scan of file.
pub struct PhysicalIoHelper {
    pub expr: Arc<dyn PhysicalExpr>,
}

impl PhysicalIoExpr for PhysicalIoHelper {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        self.expr.evaluate(df)
    }
}

pub trait PhysicalAggregation {
    #[allow(clippy::ptr_arg)]
    /// Should be called on the final aggregation node like sum, min, max, etc.
    /// When called on a tail, slice, sort, etc. it should return a list-array
    fn aggregate(&self, df: &DataFrame, groups: &GroupTuples) -> Result<Option<Series>>;

    #[allow(clippy::ptr_arg)]
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
    ) -> Result<Option<Vec<Series>>> {
        // we return a vec, such that an implementor can return more information, such as a sum and count.
        self.aggregate(df, groups).map(|opt| opt.map(|s| vec![s]))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_partitioned_final(
        &self,
        final_df: &DataFrame,
        groups: &GroupTuples,
    ) -> Result<Option<Series>> {
        self.aggregate(final_df, groups)
    }
}
