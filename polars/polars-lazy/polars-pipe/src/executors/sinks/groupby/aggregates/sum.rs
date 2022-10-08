use std::any::Any;
use std::ops::{Add, AddAssign};

use polars_core::datatypes::AnyValue;
use polars_core::export::num::NumCast;
use polars_core::utils::arrow::types::NativeType;
use polars_utils::debug_unwrap;

use super::*;
use crate::operators::IdxSize;

pub struct SumAgg<K: NativeType> {
    sum: Option<K>,
}

impl<K: NativeType> SumAgg<K> {
    pub(crate) fn new() -> Self {
        SumAgg {
            sum: None
        }
    }
}

impl<K: NativeType + Add<Output=K> + NumCast> AggregateFn for SumAgg<K> {
    fn pre_agg(&mut self, _chunk_idx: IdxSize, item: AnyValue) {
        match (item.extract::<K>(), self.sum) {
            (Some(val), Some(sum)) => self.sum = Some(sum + val),
            (Some(val), None) => self.sum = Some(val),
            (None, _) => {}
        }
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = other.downcast_ref::<Self>();
        let other = unsafe { debug_unwrap(other) };
        let sum = match (self.sum, other.sum) {
            (Some(lhs), Some(rhs)) => Some(lhs + rhs),
            (Some(lhs), None) => Some(lhs),
            (None, Some(rhs)) => Some(rhs),
            _ => None
        };
        self.sum = sum;
    }

    fn split(&self) -> Box<dyn AggregateFn> {
        Box::new(Self::new())
    }
}