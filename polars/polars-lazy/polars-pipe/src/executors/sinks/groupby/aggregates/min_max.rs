use std::any::Any;
use std::cmp::Ordering;

use polars_arrow::kernels::rolling::{compare_fn_nan_max, compare_fn_nan_min};
use polars_core::datatypes::{AnyValue, DataType};
use polars_core::export::num::NumCast;
use polars_core::prelude::NumericNative;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;
use crate::operators::{ArrowDataType, IdxSize};

pub(super) fn new_min<K: NumericNative>() -> MinMaxAgg<K, fn(&K, &K) -> Ordering> {
    MinMaxAgg::new(compare_fn_nan_min)
}

pub(super) fn new_max<K: NumericNative>() -> MinMaxAgg<K, fn(&K, &K) -> Ordering> {
    MinMaxAgg::new(compare_fn_nan_max)
}

pub struct MinMaxAgg<K: NumericNative, F: Fn(&K, &K) -> Ordering> {
    agg: Option<K>,
    cmp_fn: F,
}

impl<K: NumericNative, F: Fn(&K, &K) -> Ordering + Copy> MinMaxAgg<K, F> {
    pub(crate) fn new(f: F) -> Self {
        MinMaxAgg {
            agg: None,
            cmp_fn: f,
        }
    }

    pub(crate) fn split(&self) -> Self {
        MinMaxAgg {
            agg: None,
            cmp_fn: self.cmp_fn,
        }
    }
}

impl<K: NumericNative, F: Fn(&K, &K) -> Ordering> MinMaxAgg<K, F> {
    fn pre_agg_primitive<T: NumCast>(&mut self, item: Option<T>) {
        match (item.map(|v| K::from(v).unwrap()), self.agg) {
            (Some(val), Some(current_agg)) => {
                if (self.cmp_fn)(&current_agg, &val) == Ordering::Less {
                    self.agg = Some(current_agg);
                }
            }
            (Some(val), None) => self.agg = Some(val),
            (None, _) => {}
        }
    }
}

impl<K: NumericNative, F: Fn(&K, &K) -> Ordering + Send + Sync + 'static> AggregateFn
    for MinMaxAgg<K, F>
{
    fn has_physical_agg(&self) -> bool {
        true
    }

    fn pre_agg(&mut self, _chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe { item.next().unwrap_unchecked_release() };
        self.pre_agg_primitive(item.extract::<K>())
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
