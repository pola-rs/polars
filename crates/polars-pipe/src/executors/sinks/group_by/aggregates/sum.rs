use std::any::Any;
use std::ops::Add;

use arrow::array::PrimitiveArray;
use num_traits::NumCast;
use polars_compute::sum::{wrapping_sum_arr, WrappingSum};
use polars_core::prelude::*;

use super::*;

pub struct SumAgg<K: NumericNative> {
    sum: Option<K>,
}

impl<K: NumericNative> SumAgg<K> {
    pub(crate) fn new() -> Self {
        SumAgg { sum: None }
    }
}

impl<K> AggregateFn for SumAgg<K>
where
    K::PolarsType: PolarsNumericType,
    K: NumericNative + Add<Output = K> + WrappingSum,
{
    fn has_physical_agg(&self) -> bool {
        true
    }

    fn pre_agg(&mut self, _chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe { item.next().unwrap_unchecked() };
        self.pre_agg_primitive(0, item.extract::<K>())
    }
    fn pre_agg_primitive<T: NumCast>(&mut self, _chunk_idx: IdxSize, item: Option<T>) {
        match (item.map(|v| K::from(v).unwrap()), self.sum) {
            (Some(val), Some(sum)) => self.sum = Some(sum + val),
            (Some(val), None) => self.sum = Some(val),
            (None, _) => {},
        }
    }

    fn pre_agg_ordered(
        &mut self,
        _chunk_idx: IdxSize,
        offset: IdxSize,
        length: IdxSize,
        values: &Series,
    ) {
        // we must cast because sum output type might be different than input type.
        let arr = unsafe {
            let arr = values.chunks().get_unchecked(0);
            arr.sliced_unchecked(offset as usize, length as usize)
        };
        let dtype = K::PolarsType::get_dtype().to_arrow(CompatLevel::newest());
        let arr = polars_compute::cast::cast_unchecked(arr.as_ref(), &dtype).unwrap();
        let arr = unsafe {
            arr.as_any()
                .downcast_ref::<PrimitiveArray<K>>()
                .unwrap_unchecked()
        };
        match (wrapping_sum_arr(arr), self.sum) {
            (val, Some(sum)) => {
                self.sum = Some(sum + val);
            },
            (val, None) => {
                self.sum = Some(val);
            },
        }
    }

    fn dtype(&self) -> DataType {
        DataType::from_arrow_dtype(&ArrowDataType::from(K::PRIMITIVE))
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked() };
        let sum = match (self.sum, other.sum) {
            (Some(lhs), Some(rhs)) => Some(lhs + rhs),
            (Some(lhs), None) => Some(lhs),
            (None, Some(rhs)) => Some(rhs),
            (None, None) => None,
        };
        self.sum = sum;
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        if let Some(val) = self.sum {
            val.into()
        } else {
            K::zero().into()
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
