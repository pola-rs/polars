use crate::prelude::*;

pub trait PhysicalAggExpr {
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups(&self, df: &DataFrame, groups: &GroupPositions) -> PolarsResult<Series>;

    fn evaluate(&self, df: &DataFrame) -> PolarsResult<Column>;

    fn root_name(&self) -> PolarsResult<&PlSmallStr>;
}
