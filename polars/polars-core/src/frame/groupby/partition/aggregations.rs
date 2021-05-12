use crate::prelude::*;
use super::*;
use crate::frame::groupby::GroupedMap;
use num::{Bounded, NumCast, Num, ToPrimitive, Zero};
use rayon::prelude::*;
use crate::POOL;
use crate::chunked_array::kernels::take_agg::{take_agg_no_null_primitive_iter_unchecked, take_agg_primitive_iter_unchecked_count_nulls};
use hashbrown::HashMap;
use ahash::RandomState;
use std::any::Any;

pub type AggMap = HashMap<Option<u64>, AnyValue<'static>, RandomState>;

pub trait AggState {
    fn merge(&mut self, other: Vec<&mut dyn AggState>);
    fn as_any(&mut self) -> &mut dyn Any;
    fn finish(self) -> Series;
}


pub struct SumAggState {
    agg: AggMap
}

impl AggState for SumAggState {
    fn merge(&mut self, other: Vec<&mut dyn AggState>) {

        let mut stack = Vec::with_capacity(other.len());
        stack.push(&mut self.agg);
        let mut other: Vec<_> = other.into_iter().map(|tbl| {
            tbl.as_any().downcast_mut::<SumAggState>().unwrap()
        }).collect();

        // Go from left to right
        // take keys from left and traverse other maps to collect and remove the values.
        // so if "foo" is in the first hashtable it will be removed from the others and added
        // to the left table
        // In a next hashtable we might find "bar", but we know it is not in the tables we already
        // visited, otherwise it would have been removed
        //
        // in the end all hash tables should have unique keys.
        while let Some(agg) = stack.pop() {
            for (k, l) in agg.iter_mut() {
                for tbl in other.iter_mut() {
                    tbl.agg.remove(k).map(|r| {
                        *l = l.add(&r);
                    });
                }

                if let Some(o) = other.pop() {
                    stack.push(&mut o.agg)
                }
            };
        }
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn finish(self) -> Series {
        todo!()
    }
}

pub(crate) trait PartitionAgg {
    fn agg_sum(&self, _groups: &GroupedMap<Option<u64>>) -> Option<Box<dyn AggState>> {
        None
    }
}


impl<T> PartitionAgg for ChunkedArray<T>
    where
        T: PolarsNumericType + Sync,
        T::Native: std::ops::Add<Output = T::Native> + Num + NumCast + Bounded + Into<AnyValue<'static>>,
        ChunkedArray<T>: IntoSeries,
{

    fn agg_sum(&self, groups: &GroupedMap<Option<u64>>) -> Option<Box<dyn AggState>> {
        let agg: AggMap = POOL.install(|| {
            groups.into_par_iter()
                .map(|(k, (first, idx))| {
                    let agg = if idx.len() == 1 {
                        self.get(*first as usize).map(|sum| sum.to_f64().unwrap())
                    } else {
                        match (self.null_count(), self.chunks.len()) {
                            (0, 1) => unsafe {
                                take_agg_no_null_primitive_iter_unchecked(
                                    self.downcast_iter().next().unwrap(),
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| a + b,
                                    T::Native::zero(),
                                )
                            }
                                .to_f64()
                                .map(|sum| sum / idx.len() as f64),
                            (_, 1) => unsafe {
                                take_agg_primitive_iter_unchecked_count_nulls(
                                    self.downcast_iter().next().unwrap(),
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| a + b,
                                    T::Native::zero(),
                                )
                            }
                                .map(|(sum, null_count)| {
                                    sum.to_f64()
                                        .map(|sum| sum / (idx.len() as f64 - null_count as f64))
                                        .unwrap()
                                }),
                            _ => {
                                let take =
                                    unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                                let opt_sum: Option<T::Native> = take.sum();
                                opt_sum.map(|sum| sum.to_f64().unwrap() / idx.len() as f64)
                            }
                        }
                    };
                    (*k, agg.into())
                }).collect()
        });

        Some(Box::new(SumAggState {
            agg
        }))
    }
}
