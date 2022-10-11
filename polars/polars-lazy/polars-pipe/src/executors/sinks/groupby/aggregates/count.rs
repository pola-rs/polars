use std::any::Any;

use polars_core::datatypes::{AnyValue, DataType};
use polars_core::prelude::IDX_DTYPE;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;
use crate::operators::IdxSize;

pub struct CountAgg {
    count: IdxSize,
}

impl CountAgg {
    pub(crate) fn new() -> Self {
        CountAgg { count: 0 }
    }
}

impl AggregateFn for CountAgg {
    fn pre_agg(&mut self, _chunk_idx: IdxSize, _item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        self.count += 1;
    }

    fn dtype(&self) -> DataType {
        IDX_DTYPE
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };
        self.count += other.count;
    }

    fn split(&self) -> Box<dyn AggregateFn> {
        Box::new(Self::new())
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        AnyValue::from(self.count)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
