use std::any::Any;
use std::ops::Add;

use arrow::array::{Array, PrimitiveArray};
use arrow::compute::aggregate::Sum;
use arrow::types::simd::Simd;
use polars_core::export::arrow::datatypes::PrimitiveType;
use polars_core::export::num::NumCast;
use polars_core::prelude::*;
use polars_core::utils::arrow::compute::aggregate::sum_primitive;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;

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

impl<K> AggregateFn for MeanAgg<K>
where
    K::PolarsType: PolarsNumericType,
    K: NumericNative + Add<Output = K>,
    <K as Simd>::Simd: Add<Output = <K as Simd>::Simd> + Sum<K>,
{
    fn has_physical_agg(&self) -> bool {
        true
    }

    fn pre_agg_primitive<T: NumCast>(&mut self, _chunk_idx: IdxSize, item: Option<T>) {
        match (item.map(|v| K::from(v).unwrap()), self.sum) {
            (Some(val), Some(sum)) => {
                self.sum = Some(sum + val);
                self.count += 1;
            },
            (Some(val), None) => {
                self.sum = Some(val);
                self.count += 1;
            },
            _ => {},
        }
    }

    fn pre_agg(&mut self, _chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe { item.next().unwrap_unchecked_release() };
        match (item.extract::<K>(), self.sum) {
            (Some(val), Some(sum)) => {
                self.sum = Some(sum + val);
                self.count += 1;
            },
            (Some(val), None) => {
                self.sum = Some(val);
                self.count += 1;
            },
            _ => {},
        }
    }

    fn pre_agg_ordered(
        &mut self,
        _chunk_idx: IdxSize,
        offset: IdxSize,
        length: IdxSize,
        values: &Series,
    ) {
        // we must cast because mean might be a different dtype
        let arr = unsafe {
            let arr = values.chunks().get_unchecked(0);
            arr.sliced_unchecked(offset as usize, length as usize)
        };
        let dtype = K::PolarsType::get_dtype().to_arrow(CompatLevel::newest());
        let arr = arrow::compute::cast::cast_unchecked(arr.as_ref(), &dtype).unwrap();
        let arr = unsafe {
            arr.as_any()
                .downcast_ref::<PrimitiveArray<K>>()
                .unwrap_unchecked_release()
        };
        match (sum_primitive(arr), self.sum) {
            (Some(val), Some(sum)) => {
                self.sum = Some(sum + val);
                self.count += (arr.len() - arr.null_count()) as IdxSize;
            },
            (Some(val), None) => {
                self.sum = Some(val);
                self.count += (arr.len() - arr.null_count()) as IdxSize;
            },
            _ => {},
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
            },
            (None, Some(rhs)) => {
                self.sum = Some(rhs);
                self.count = other.count;
            },
            _ => {},
        };
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
