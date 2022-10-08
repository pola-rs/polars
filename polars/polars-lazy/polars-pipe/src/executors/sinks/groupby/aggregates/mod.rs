use interface::*;

mod interface;
mod sum;
mod convert;

pub(crate) use interface::AggregateFn;
pub(crate) use sum::SumAgg;
pub(crate) use convert::*;

