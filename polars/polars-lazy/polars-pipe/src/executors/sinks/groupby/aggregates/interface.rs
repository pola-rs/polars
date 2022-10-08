use std::any::Any;

use polars_core::prelude::AnyValue;

use crate::operators::IdxSize;

pub trait AggregateFn: Send {
    fn pre_agg(&mut self, chunk_idx: IdxSize, item: AnyValue);

    fn combine(&mut self, other: &dyn Any);
}
