use std::any::Any;

use arrow::array::PrimitiveArray;
use polars_compute::min_max::MinMaxKernel;
use polars_core::export::num::NumCast;
use polars_core::prelude::*;
use polars_utils::min_max::MinMax;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;

pub(super) fn new_min<K: NumericNative>() -> MinMaxAgg<K, fn(K, K) -> K> {
    MinMaxAgg::new(MinMax::min_ignore_nan, true)
}

pub(super) fn new_max<K: NumericNative>() -> MinMaxAgg<K, fn(K, K) -> K> {
    MinMaxAgg::new(MinMax::max_ignore_nan, false)
}

pub struct MinMaxAgg<K: NumericNative, F> {
    agg: Option<K>,
    agg_fn: F,
    is_min: bool,
}

impl<K: NumericNative, F: Fn(K, K) -> K + Copy> MinMaxAgg<K, F> {
    pub(crate) fn new(f: F, is_min: bool) -> Self {
        MinMaxAgg {
            agg: None,
            agg_fn: f,
            is_min,
        }
    }

    pub(crate) fn split(&self) -> Self {
        MinMaxAgg {
            agg: None,
            agg_fn: self.agg_fn,
            is_min: self.is_min,
        }
    }
}

impl<K, F: Fn(K, K) -> K + Send + Sync + 'static> AggregateFn for MinMaxAgg<K, F>
where
    K: NumericNative,
    PrimitiveArray<K>: for<'a> MinMaxKernel<Scalar<'a> = K>,
{
    fn has_physical_agg(&self) -> bool {
        true
    }

    fn pre_agg(&mut self, chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe { item.next().unwrap_unchecked_release() };
        self.pre_agg_primitive(chunk_idx, item.extract::<K>())
    }

    fn pre_agg_primitive<T: NumCast>(&mut self, _chunk_idx: IdxSize, item: Option<T>) {
        match (item.map(|v| K::from(v).unwrap()), self.agg) {
            (Some(val), Some(current_agg)) => {
                self.agg = Some((self.agg_fn)(current_agg, val));
            },
            (Some(val), None) => self.agg = Some(val),
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
        let ca: &ChunkedArray<K::PolarsType> = values.as_ref().as_ref();
        let arr = ca.downcast_iter().next().unwrap();
        let arr = unsafe { arr.slice_typed_unchecked(offset as usize, length as usize) };
        // convince the compiler that K::POLARSTYPE::Native == K
        let arr = unsafe { std::mem::transmute::<PrimitiveArray<_>, PrimitiveArray<K>>(arr) };
        let agg = if self.is_min {
            arr.min_ignore_nan_kernel()
        } else {
            arr.max_ignore_nan_kernel()
        };
        self.pre_agg_primitive(0, agg)
    }

    fn dtype(&self) -> DataType {
        (&ArrowDataType::from(K::PRIMITIVE)).into()
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };
        self.pre_agg_primitive(0, other.agg)
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
