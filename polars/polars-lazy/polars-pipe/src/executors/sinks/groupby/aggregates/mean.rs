use std::any::Any;
use std::ops::Add;

use polars_core::datatypes::{AnyValue, DataType};
use polars_core::export::arrow::datatypes::PrimitiveType;
use polars_core::export::num::NumCast;
use polars_core::prelude::NumericNative;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;
use crate::operators::{ArrowDataType, IdxSize};

pub struct MeanAgg<K: NumericNative> {
    sum: Option<K>,
    count: IdxSize,
}

impl<K: NumericNative> MeanAgg<K> {
    pub(crate) fn new() -> Self {
        MeanAgg {
            sum: None,
            count: 0,
        }
    }
}

impl<K: NumericNative + Add<Output = K> + NumCast> AggregateFn for MeanAgg<K> {
    fn pre_agg(&mut self, _chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe { item.next().unwrap_unchecked_release() };
        match (item.extract::<K>(), self.sum) {
            (Some(val), Some(sum)) => {
                self.sum = Some(sum + val);
                self.count += 1;
            }
            (Some(val), None) => {
                self.sum = Some(val);
                self.count += 1;
            }
            _ => {}
        }
    }

    fn dtype(&self) -> DataType {
        (&ArrowDataType::from(K::PRIMITIVE)).into()
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };
        match (self.sum, other.sum) {
            (Some(lhs), Some(rhs)) => {
                self.sum = Some(lhs + rhs);
                self.count += other.count;
            }
            (None, Some(rhs)) => {
                self.sum = Some(rhs);
                self.count = other.count;
            }
            _ => {}
        };
    }

    fn split(&self) -> Box<dyn AggregateFn> {
        Box::new(Self::new())
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        if let Some(val) = self.sum {
            unsafe {
                match K::PRIMITIVE {
                    PrimitiveType::Float32 => AnyValue::Float32(
                        val.to_f32().unwrap_unchecked_release() / self.count as f32,
                    ),
                    PrimitiveType::Float64 => AnyValue::Float64(
                        val.to_f64().unwrap_unchecked_release() / self.count as f64,
                    ),
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
