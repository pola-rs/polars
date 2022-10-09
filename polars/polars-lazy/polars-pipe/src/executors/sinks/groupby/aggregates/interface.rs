use std::any::Any;
use std::fmt::Debug;

use polars_core::datatypes::DataType;
use polars_core::prelude::AnyValue;

use crate::operators::IdxSize;

pub trait AggregateFn: Send + Debug {
    fn pre_agg(&mut self, chunk_idx: IdxSize, item: AnyValue);

    fn dtype(&self) -> DataType;

    fn combine(&mut self, other: &dyn Any);

    fn split(&self) -> Box<dyn AggregateFn>;

    fn finalize(&mut self) -> AnyValue<'static>;

    fn as_any(&self) -> &dyn Any;
}
