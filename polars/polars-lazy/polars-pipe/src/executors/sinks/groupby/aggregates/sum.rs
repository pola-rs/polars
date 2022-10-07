use std::any::Any;
use std::ops::AddAssign;
use polars_core::datatypes::AnyValue;
use polars_core::export::num::NumCast;
use polars_core::utils::arrow::types::NativeType;
use polars_utils::debug_unwrap;
use crate::operators::IdxSize;
use super::*;


pub struct SumAgg<K: NativeType> {
    sum: K,
}

impl<K: NativeType + AddAssign + NumCast> AggregateFn for SumAgg<K> {
    fn pre_agg(&mut self, _chunk_idx: IdxSize, item: AnyValue) {
        let item = item.extract::<K>();
        let item = unsafe { debug_unwrap(item) };
        self.sum += item;
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = other.downcast_ref::<Self>();
        let other = unsafe { debug_unwrap(other) };
        self.sum += other.sum;
    }
}