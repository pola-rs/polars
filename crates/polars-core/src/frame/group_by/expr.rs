use crate::prelude::*;

pub trait PhysicalAggExpr {
    #[allow(clippy::ptr_arg)]
    fn evaluate(&self, df: &DataFrame, groups: &GroupsProxy) -> PolarsResult<Series>;

    fn root_name(&self) -> PolarsResult<&str>;
}
