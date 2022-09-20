use crate::prelude::*;

pub trait PhysicalAggExpr {
    #[allow(clippy::ptr_arg)]
    fn evaluate<'a>(&self, df: &DataFrame, groups: &'a GroupsProxy) -> PolarsResult<Series>;

    fn root_name(&self) -> PolarsResult<&str>;
}
