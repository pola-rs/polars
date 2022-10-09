use std::any::Any;

use polars_core::datatypes::DataType;
use polars_core::prelude::AnyValue;
use polars_utils::debug_unwrap;

use crate::executors::sinks::groupby::aggregates::AggregateFn;
use crate::operators::IdxSize;

pub struct LastAgg {
    chunk_idx: IdxSize,
    last: Option<AnyValue<'static>>,
    dtype: DataType,
}

impl LastAgg {
    pub(crate) fn new(dtype: DataType) -> Self {
        Self {
            chunk_idx: 0,
            last: None,
            dtype,
        }
    }
}

impl AggregateFn for LastAgg {
    fn pre_agg(&mut self, chunk_idx: IdxSize, item: AnyValue) {
        self.chunk_idx = chunk_idx;
        self.last = Some(item.into_static().unwrap())
    }

    fn dtype(&self) -> DataType {
        self.dtype.clone()
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { debug_unwrap(other.downcast_ref::<Self>()) };
        if other.last.is_some() && other.chunk_idx > self.chunk_idx {
            self.last = other.last.clone();
            self.chunk_idx = other.chunk_idx;
        };
    }

    fn split(&self) -> Box<dyn AggregateFn> {
        Box::new(Self::new(self.dtype.clone()))
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        std::mem::take(&mut self.last).unwrap_or(AnyValue::Null)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
