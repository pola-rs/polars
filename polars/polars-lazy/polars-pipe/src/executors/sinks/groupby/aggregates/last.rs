use std::any::Any;

use polars_core::datatypes::DataType;
use polars_core::prelude::AnyValue;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use crate::executors::sinks::groupby::aggregates::AggregateFn;
use crate::operators::IdxSize;

pub struct LastAgg {
    chunk_idx: IdxSize,
    last: Option<AnyValue<'static>>,
    pub(crate) dtype: DataType,
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
    fn pre_agg(&mut self, _chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe { item.next().unwrap_unchecked_release() };
        self.last = unsafe { Some(item.into_static().unwrap_unchecked()) };
    }

    fn dtype(&self) -> DataType {
        self.dtype.clone()
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };
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
