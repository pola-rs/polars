use std::any::Any;

use polars_core::datatypes::DataType;
use polars_core::prelude::AnyValue;
use polars_utils::debug_unwrap;

use crate::executors::sinks::groupby::aggregates::AggregateFn;
use crate::operators::IdxSize;

#[derive(Debug)]
pub struct FirstAgg {
    chunk_idx: IdxSize,
    first: Option<AnyValue<'static>>,
    dtype: DataType,
}

impl FirstAgg {
    pub(crate) fn new(dtype: DataType) -> Self {
        Self {
            chunk_idx: 0,
            first: None,
            dtype,
        }
    }
}

impl AggregateFn for FirstAgg {
    fn pre_agg(&mut self, chunk_idx: IdxSize, item: AnyValue) {
        if self.first.is_none() {
            self.chunk_idx = chunk_idx;
            self.first = Some(item.into_static().unwrap())
        }
    }

    fn dtype(&self) -> DataType {
        self.dtype.clone()
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { debug_unwrap(other.downcast_ref::<Self>()) };
        if other.first.is_some() && other.chunk_idx < self.chunk_idx {
            self.first = other.first.clone()
        };
    }

    fn split(&self) -> Box<dyn AggregateFn> {
        Box::new(Self::new(self.dtype.clone()))
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        std::mem::take(&mut self.first).unwrap_or(AnyValue::Null)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
