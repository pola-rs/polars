use std::any::Any;
use std::cmp::Ordering;

use polars_arrow::export::arrow::array::PrimitiveArray;
use polars_arrow::export::arrow::compute::aggregate::SimdOrd;
use polars_arrow::kernels::rolling::{compare_fn_nan_max, compare_fn_nan_min};
use polars_core::datatypes::{AnyValue, DataType};
use polars_core::export::arrow::types::simd::Simd;
use polars_core::export::num::NumCast;
use polars_core::prelude::*;
use polars_core::utils::arrow::compute::aggregate::{max_primitive, min_primitive};
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;
use crate::operators::{ArrowDataType, IdxSize};

#[inline]
fn compare_fn_min<T: NumericNative>(a: &T, b: &T) -> Ordering {
    // reverse the ordering
    compare_fn_nan_max(b, a)
}

pub(super) fn new_min<K: NumericNative>() -> MinMaxAgg<K, fn(&K, &K) -> Ordering> {
    MinMaxAgg::new(compare_fn_min, true)
}

pub(super) fn new_max<K: NumericNative>() -> MinMaxAgg<K, fn(&K, &K) -> Ordering> {
    MinMaxAgg::new(compare_fn_nan_min, false)
}

pub struct MinMaxAgg<K: NumericNative, F: Fn(&K, &K) -> Ordering> {
    agg: Option<K>,
    cmp_fn: F,
    is_min: bool,
}

impl<K: NumericNative, F: Fn(&K, &K) -> Ordering + Copy> MinMaxAgg<K, F> {
    pub(crate) fn new(f: F, is_min: bool) -> Self {
        MinMaxAgg {
            agg: None,
            cmp_fn: f,
            is_min,
        }
    }

    pub(crate) fn split(&self) -> Self {
        MinMaxAgg {
            agg: None,
            cmp_fn: self.cmp_fn,
            is_min: self.is_min,
        }
    }
}

impl<K: NumericNative, F: Fn(&K, &K) -> Ordering> MinMaxAgg<K, F> {
    fn pre_agg_primitive<T: NumCast>(&mut self, item: Option<T>) {
        match (item.map(|v| K::from(v).unwrap()), self.agg) {
            (Some(val), Some(current_agg)) => {
                // The ordering is static, we swap the arguments in the compare fn to minic
                // min/max behavior.
                if (self.cmp_fn)(&current_agg, &val) == Ordering::Less {
                    self.agg = Some(val);
                }
            }
            (Some(val), None) => self.agg = Some(val),
            (None, _) => {}
        }
    }
}

impl<K, F: Fn(&K, &K) -> Ordering + Send + Sync + 'static> AggregateFn for MinMaxAgg<K, F>
where
    K: Simd + NumericNative,
    <K as Simd>::Simd: SimdOrd<K>,
{
    fn has_physical_agg(&self) -> bool {
        true
    }

    fn pre_agg(&mut self, _chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe { item.next().unwrap_unchecked_release() };
        self.pre_agg_primitive(item.extract::<K>())
    }

    fn pre_agg_ordered(
        &mut self,
        _chunk_idx: IdxSize,
        offset: IdxSize,
        length: IdxSize,
        values: &Series,
    ) {
        let ca: &ChunkedArray<K::POLARSTYPE> = values.as_ref().as_ref();
        let arr = ca.downcast_iter().next().unwrap();
        let arr = unsafe { arr.slice_typed_unchecked(offset as usize, length as usize) };
        // convince the compiler that K::POLARSTYPE::Native == K
        let arr = unsafe { std::mem::transmute::<PrimitiveArray<_>, PrimitiveArray<K>>(arr) };
        let agg = if self.is_min {
            min_primitive(&arr)
        } else {
            max_primitive(&arr)
        };
        self.pre_agg_primitive(agg)
    }
    fn pre_agg_i8(&mut self, _chunk_idx: IdxSize, item: Option<i8>) {
        self.pre_agg_primitive(item)
    }
    fn pre_agg_u8(&mut self, _chunk_idx: IdxSize, item: Option<u8>) {
        self.pre_agg_primitive(item)
    }

    fn pre_agg_i16(&mut self, _chunk_idx: IdxSize, item: Option<i16>) {
        self.pre_agg_primitive(item)
    }
    fn pre_agg_u16(&mut self, _chunk_idx: IdxSize, item: Option<u16>) {
        self.pre_agg_primitive(item)
    }
    fn pre_agg_i32(&mut self, _chunk_idx: IdxSize, item: Option<i32>) {
        self.pre_agg_primitive(item)
    }
    fn pre_agg_i64(&mut self, _chunk_idx: IdxSize, item: Option<i64>) {
        self.pre_agg_primitive(item)
    }
    fn pre_agg_u32(&mut self, _chunk_idx: IdxSize, item: Option<u32>) {
        self.pre_agg_primitive(item)
    }
    fn pre_agg_u64(&mut self, _chunk_idx: IdxSize, item: Option<u64>) {
        self.pre_agg_primitive(item)
    }
    fn pre_agg_f32(&mut self, _chunk_idx: IdxSize, item: Option<f32>) {
        self.pre_agg_primitive(item)
    }
    fn pre_agg_f64(&mut self, _chunk_idx: IdxSize, item: Option<f64>) {
        self.pre_agg_primitive(item)
    }

    fn dtype(&self) -> DataType {
        (&ArrowDataType::from(K::PRIMITIVE)).into()
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };
        self.pre_agg_primitive(other.agg)
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        if let Some(val) = self.agg {
            val.into()
        } else {
            AnyValue::Null
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
