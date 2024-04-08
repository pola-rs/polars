use std::any::Any;

use polars_core::datatypes::{AnyValue, DataType};
use polars_core::prelude::{Series, IDX_DTYPE};
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;
use crate::operators::IdxSize;

pub(crate) struct CountAgg<const INCLUDE_NULL: bool> {
    count: IdxSize,
}

impl<const INCLUDE_NULL: bool> CountAgg<INCLUDE_NULL> {
    pub(crate) fn new() -> Self {
        CountAgg { count: 0 }
    }
}

impl<const INCLUDE_NULL: bool> AggregateFn for CountAgg<INCLUDE_NULL> {
    fn has_physical_agg(&self) -> bool {
        false
    }

    fn pre_agg(&mut self, _chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe { item.next().unwrap_unchecked_release() };
        if INCLUDE_NULL {
            self.count += 1;
        } else {
            self.count += !matches!(item, AnyValue::Null) as IdxSize;
        }
    }
    fn pre_agg_ordered(
        &mut self,
        _chunk_idx: IdxSize,
        _offset: IdxSize,
        length: IdxSize,
        _values: &Series,
    ) {
        self.count += length
    }

    fn dtype(&self) -> DataType {
        IDX_DTYPE
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };
        self.count += other.count;
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        AnyValue::from(self.count)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
