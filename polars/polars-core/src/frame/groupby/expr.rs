use crate::prelude::*;

pub trait PhysicalAggExpr {
    #[allow(clippy::ptr_arg)]
    fn evaluate<'a>(&self, df: &DataFrame, groups: &'a GroupsProxy) -> Result<Series>;

    fn root_name(&self) -> Result<&str>;
}
