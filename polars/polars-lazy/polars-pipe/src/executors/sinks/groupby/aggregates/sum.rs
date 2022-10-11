use std::any::Any;
use std::ops::Add;

use polars_core::datatypes::{AnyValue, DataType};
use polars_core::export::arrow::datatypes::PrimitiveType;
use polars_core::export::num::NumCast;
use polars_core::prelude::NumericNative;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;
use crate::operators::{ArrowDataType, IdxSize};

pub struct SumAgg<K: NumericNative> {
    sum: Option<K>,
}

impl<K: NumericNative> SumAgg<K> {
    pub(crate) fn new() -> Self {
        SumAgg { sum: None }
    }
}

impl<K: NumericNative + Add<Output = K> + NumCast> AggregateFn for SumAgg<K> {
    fn pre_agg(&mut self, _chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe { item.next().unwrap_unchecked_release() };
        match (item.extract::<K>(), self.sum) {
            (Some(val), Some(sum)) => self.sum = Some(sum + val),
            (Some(val), None) => self.sum = Some(val),
            (None, _) => {}
        }
    }

    fn dtype(&self) -> DataType {
        (&ArrowDataType::from(K::PRIMITIVE)).into()
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };
        let sum = match (self.sum, other.sum) {
            (Some(lhs), Some(rhs)) => Some(lhs + rhs),
            (Some(lhs), None) => Some(lhs),
            (None, Some(rhs)) => Some(rhs),
            (None, None) => None,
        };
        self.sum = sum;
    }

    fn split(&self) -> Box<dyn AggregateFn> {
        Box::new(Self::new())
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        if let Some(val) = self.sum {
            unsafe {
                match K::PRIMITIVE {
                    PrimitiveType::Int8 => {
                        AnyValue::Int8(NumCast::from(val).unwrap_unchecked_release())
                    }
                    PrimitiveType::Int16 => {
                        AnyValue::Int16(NumCast::from(val).unwrap_unchecked_release())
                    }
                    PrimitiveType::Int32 => {
                        AnyValue::Int32(NumCast::from(val).unwrap_unchecked_release())
                    }
                    PrimitiveType::Int64 => {
                        AnyValue::Int64(NumCast::from(val).unwrap_unchecked_release())
                    }
                    PrimitiveType::UInt8 => {
                        AnyValue::UInt8(NumCast::from(val).unwrap_unchecked_release())
                    }
                    PrimitiveType::UInt16 => {
                        AnyValue::UInt16(NumCast::from(val).unwrap_unchecked_release())
                    }
                    PrimitiveType::UInt32 => {
                        AnyValue::UInt32(NumCast::from(val).unwrap_unchecked_release())
                    }
                    PrimitiveType::UInt64 => {
                        AnyValue::UInt64(NumCast::from(val).unwrap_unchecked_release())
                    }
                    PrimitiveType::Float32 => {
                        AnyValue::Float32(NumCast::from(val).unwrap_unchecked_release())
                    }
                    PrimitiveType::Float64 => {
                        AnyValue::Float64(NumCast::from(val).unwrap_unchecked_release())
                    }
                    _ => todo!(),
                }
            }
        } else {
            AnyValue::Null
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
