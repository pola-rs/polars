use std::any::Any;
use polars_core::prelude::*;
use polars_core::export::arrow;
use polars_core::utils::_set_partition_size;
use polars_core::utils::arrow::types::NativeType;
use crate::operators::{DataChunk, PExecutionContext, Sink, SinkResult};
use super::aggregates::AggregateFn;


type PartitionedHashMap<K> = PlHashMap<K, *mut [Box<dyn AggregateFn>]>;

pub struct PrimitiveGroupbySink<K: NumericNative>{
    thread_no: usize,
    pre_agg: Vec<PartitionedHashMap<K>>
}

impl<K: NumericNative>  PrimitiveGroupbySink<K> {
    pub fn new(thread_no: usize) -> Self {
        Self {
            thread_no,
            pre_agg: Vec::with_capacity(_set_partition_size())
        }
    }

}

impl<K: NumericNative> Sink for PrimitiveGroupbySink<K> {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        todo!()
    }

    fn combine(&mut self, other: Box<dyn Sink>) {
        todo!()
    }

    fn split(&self, thread_no: usize) -> Box<dyn Sink> {

        todo!()
    }

    fn finalize(&mut self) -> PolarsResult<DataFrame> {
        todo!()
    }

    fn as_any(&self) -> &dyn Any {
        todo!()
    }
}

unsafe impl<K: NumericNative> Send for PrimitiveGroupbySink<K> {}
unsafe impl<K: NumericNative> Sync for PrimitiveGroupbySink<K> {}
