use interface::*;

mod interface;
mod sum;

pub(crate) use sum::SumAgg;
pub(crate) use interface::AggregateFn;