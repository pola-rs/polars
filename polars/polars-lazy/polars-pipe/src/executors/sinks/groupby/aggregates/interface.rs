use std::any::Any;
use std::cmp::Ordering;

use enum_dispatch::enum_dispatch;
use polars_core::datatypes::DataType;
use polars_core::prelude::{AnyValue, Series};

use crate::executors::sinks::groupby::aggregates::count::CountAgg;
use crate::executors::sinks::groupby::aggregates::first::FirstAgg;
use crate::executors::sinks::groupby::aggregates::last::LastAgg;
use crate::executors::sinks::groupby::aggregates::mean::MeanAgg;
use crate::executors::sinks::groupby::aggregates::min_max::MinMaxAgg;
use crate::executors::sinks::groupby::aggregates::null::NullAgg;
use crate::executors::sinks::groupby::aggregates::SumAgg;
use crate::operators::IdxSize;

#[enum_dispatch(AggregateFunction)]
pub(crate) trait AggregateFn: Send + Sync {
    fn has_physical_agg(&self) -> bool {
        false
    }
    fn pre_agg(&mut self, _chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>);
    fn pre_agg_ordered(
        &mut self,
        _chunk_idx: IdxSize,
        offset: IdxSize,
        length: IdxSize,
        values: &Series,
    );
    fn pre_agg_u8(&mut self, _chunk_idx: IdxSize, _item: Option<u8>) {
        unimplemented!()
    }
    fn pre_agg_u16(&mut self, _chunk_idx: IdxSize, _item: Option<u16>) {
        unimplemented!()
    }
    fn pre_agg_u32(&mut self, _chunk_idx: IdxSize, _item: Option<u32>) {
        unimplemented!()
    }
    fn pre_agg_u64(&mut self, _chunk_idx: IdxSize, _item: Option<u64>) {
        unimplemented!()
    }
    fn pre_agg_i8(&mut self, _chunk_idx: IdxSize, _item: Option<i8>) {
        unimplemented!()
    }
    fn pre_agg_i16(&mut self, _chunk_idx: IdxSize, _item: Option<i16>) {
        unimplemented!()
    }
    fn pre_agg_i32(&mut self, _chunk_idx: IdxSize, _item: Option<i32>) {
        unimplemented!()
    }
    fn pre_agg_i64(&mut self, _chunk_idx: IdxSize, _item: Option<i64>) {
        unimplemented!()
    }
    fn pre_agg_f32(&mut self, _chunk_idx: IdxSize, _item: Option<f32>) {
        unimplemented!()
    }
    fn pre_agg_f64(&mut self, _chunk_idx: IdxSize, _item: Option<f64>) {
        unimplemented!()
    }

    fn dtype(&self) -> DataType;

    fn combine(&mut self, other: &dyn Any);

    fn finalize(&mut self) -> AnyValue<'static>;

    fn as_any(&self) -> &dyn Any;
}

// We dispatch via an enum
// as that saves an indirection
#[enum_dispatch]
pub(crate) enum AggregateFunction {
    First(FirstAgg),
    Last(LastAgg),
    Count(CountAgg),
    SumF32(SumAgg<f32>),
    SumF64(SumAgg<f64>),
    SumU32(SumAgg<u32>),
    SumU64(SumAgg<u64>),
    SumI32(SumAgg<i32>),
    SumI64(SumAgg<i64>),
    MeanF32(MeanAgg<f32>),
    MeanF64(MeanAgg<f64>),
    Null(NullAgg),
    MinMaxF32(MinMaxAgg<f32, fn(&f32, &f32) -> Ordering>),
    MinMaxF64(MinMaxAgg<f64, fn(&f64, &f64) -> Ordering>),
    MinMaxU8(MinMaxAgg<u8, fn(&u8, &u8) -> Ordering>),
    MinMaxU16(MinMaxAgg<u16, fn(&u16, &u16) -> Ordering>),
    MinMaxU32(MinMaxAgg<u32, fn(&u32, &u32) -> Ordering>),
    MinMaxU64(MinMaxAgg<u64, fn(&u64, &u64) -> Ordering>),
    MinMaxI8(MinMaxAgg<i8, fn(&i8, &i8) -> Ordering>),
    MinMaxI16(MinMaxAgg<i16, fn(&i16, &i16) -> Ordering>),
    MinMaxI32(MinMaxAgg<i32, fn(&i32, &i32) -> Ordering>),
    MinMaxI64(MinMaxAgg<i64, fn(&i64, &i64) -> Ordering>),
}

impl AggregateFunction {
    pub(crate) fn split2(&self) -> Self {
        use AggregateFunction::*;
        match self {
            First(agg) => First(FirstAgg::new(agg.dtype.clone())),
            Last(agg) => Last(LastAgg::new(agg.dtype.clone())),
            SumF32(_) => SumF32(SumAgg::new()),
            SumF64(_) => SumF64(SumAgg::new()),
            SumU32(_) => SumU32(SumAgg::new()),
            SumU64(_) => SumU64(SumAgg::new()),
            SumI32(_) => SumI32(SumAgg::new()),
            SumI64(_) => SumI64(SumAgg::new()),
            MeanF32(_) => MeanF32(MeanAgg::new()),
            MeanF64(_) => MeanF64(MeanAgg::new()),
            Count(_) => Count(CountAgg::new()),
            Null(a) => Null(a.clone()),
            MinMaxF32(inner) => MinMaxF32(inner.split()),
            MinMaxF64(inner) => MinMaxF64(inner.split()),
            MinMaxU8(inner) => MinMaxU8(inner.split()),
            MinMaxU16(inner) => MinMaxU16(inner.split()),
            MinMaxU32(inner) => MinMaxU32(inner.split()),
            MinMaxU64(inner) => MinMaxU64(inner.split()),
            MinMaxI8(inner) => MinMaxI8(inner.split()),
            MinMaxI16(inner) => MinMaxI16(inner.split()),
            MinMaxI32(inner) => MinMaxI32(inner.split()),
            MinMaxI64(inner) => MinMaxI64(inner.split()),
        }
    }
}
