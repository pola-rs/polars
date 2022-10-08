use interface::*;

mod interface;
mod sum;

pub(crate) use interface::AggregateFn;
pub(crate) use sum::SumAgg;
