use std::any::Any;

use num::{FromPrimitive, NumCast};
use polars_core::datatypes::DataType;
use polars_core::prelude::AnyValue;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use crate::executors::sinks::groupby::aggregates::AggregateFn;
use crate::operators::IdxSize;

pub struct LastAgg {
    chunk_idx: IdxSize,
    last: AnyValue<'static>,
    pub(crate) dtype: DataType,
}

impl LastAgg {
    pub(crate) fn new(dtype: DataType) -> Self {
        Self {
            chunk_idx: 0,
            last: AnyValue::Null,
            dtype,
        }
    }

    fn pre_agg_primitive<K: Into<AnyValue<'static>>>(
        &mut self,
        chunk_idx: IdxSize,
        item: Option<K>,
    ) {
        self.chunk_idx = chunk_idx;
        self.last = item.into();
    }
}

impl AggregateFn for LastAgg {
    fn pre_agg(&mut self, chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe { item.next().unwrap_unchecked_release() };
        self.chunk_idx = chunk_idx;
        self.last = unsafe { item.into_static().unwrap_unchecked() };
    }

    fn dtype(&self) -> DataType {
        self.dtype.clone()
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };
        if other.chunk_idx > self.chunk_idx {
            self.last = other.last.clone();
            self.chunk_idx = other.chunk_idx;
        };
    }

    fn split(&self) -> Box<dyn AggregateFn> {
        Box::new(Self::new(self.dtype.clone()))
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        std::mem::take(&mut self.last)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
