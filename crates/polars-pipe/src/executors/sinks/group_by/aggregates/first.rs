use std::any::Any;

use polars_core::datatypes::DataType;
use polars_core::prelude::{AnyValue, Series};
use polars_utils::unwrap::UnwrapUncheckedRelease;

use crate::executors::sinks::group_by::aggregates::AggregateFn;
use crate::operators::IdxSize;

pub(crate) struct FirstAgg {
    chunk_idx: IdxSize,
    first: Option<AnyValue<'static>>,
    pub(crate) dtype: DataType,
}

impl FirstAgg {
    pub(crate) fn new(dtype: DataType) -> Self {
        Self {
            chunk_idx: IdxSize::MAX,
            first: None,
            dtype,
        }
    }
}

impl AggregateFn for FirstAgg {
    fn pre_agg(&mut self, chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe { item.next().unwrap_unchecked_release() };
        if self.first.is_none() {
            self.chunk_idx = chunk_idx;
            self.first = Some(item.into_static().unwrap())
        }
    }
    fn pre_agg_ordered(
        &mut self,
        chunk_idx: IdxSize,
        offset: IdxSize,
        _length: IdxSize,
        values: &Series,
    ) {
        if self.first.is_none() {
            self.chunk_idx = chunk_idx;
            self.first = Some(
                unsafe { values.get_unchecked(offset as usize) }
                    .into_static()
                    .unwrap(),
            )
        }
    }

    fn dtype(&self) -> DataType {
        self.dtype.clone()
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };
        if other.first.is_some() && other.chunk_idx < self.chunk_idx {
            self.first.clone_from(&other.first);
            self.chunk_idx = other.chunk_idx;
        };
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        std::mem::take(&mut self.first).unwrap_or(AnyValue::Null)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
