use crate::chunked_array::kernels::take_agg::{
    take_agg_no_null_primitive_iter_unchecked, take_agg_primitive_iter_unchecked,
};
use crate::frame::groupby::GroupedMap;
use crate::prelude::*;
use crate::utils::{CustomIterTools, NoNull};
use crate::POOL;
use ahash::RandomState;
use hashbrown::HashMap;
use num::{Bounded, Num, NumCast, Zero};
use rayon::prelude::*;
use std::any::Any;

pub trait AggState {
    fn merge(&mut self, other: Vec<Box<dyn AggState>>);
    fn as_any(&mut self) -> &mut dyn Any;
    fn keys(&self) -> UInt32Chunked;
    fn finish(&mut self, dtype: DataType) -> Series;
}

// u32: an index of the group (can be used to get the keys)
// AnyValue: the aggregation
pub type AggMap = HashMap<Option<u64>, (u32, AnyValue<'static>), RandomState>;

pub struct SumAggState {
    agg: Vec<AggMap>,
}

impl AggState for SumAggState {
    fn merge(&mut self, mut other: Vec<Box<dyn AggState>>) {
        let mut finished = Vec::with_capacity(other.len() + 1);
        let mut stack = Vec::with_capacity(other.len());
        stack.push(&mut self.agg[0]);
        let mut other: Vec<_> = other
            .iter_mut()
            .map(|tbl| tbl.as_any().downcast_mut::<SumAggState>().unwrap())
            .collect();

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
                    tbl.agg[0].remove(k).map(|r| {
                        *l = (l.0, l.1.add(&r.1));
                    });
                }
            }

            // pop from the probe tables and add to the stack so it will be used
            // next as the left table
            if let Some(o) = other.pop() {
                stack.push(&mut o.agg[0])
            }
            finished.push(std::mem::take(agg));
        }
        self.agg = finished;
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn keys(&self) -> UInt32Chunked {
        let len = self.agg.iter().map(|tbl| tbl.len()).sum();
        let ca: NoNull<UInt32Chunked> = self
            .agg
            .iter()
            .map(|tbl| tbl.iter().map(|(_, v)| v.0))
            .flatten()
            .trust_my_length(len)
            .collect();
        ca.into_inner()
    }

    fn finish(&mut self, dtype: DataType) -> Series {
        match dtype {
            DataType::Int64 => self
                .agg
                .iter()
                .map(|tbl| {
                    tbl.iter().map(|(_, v)| {
                        let i: Option<i64> = v.1.clone().into();
                        i
                    })
                })
                .flatten()
                .collect(),
            _ => todo!(),
        }
    }
}

pub(crate) trait PartitionAgg {
    fn part_agg_sum(&self, _groups: &GroupedMap<Option<u64>>) -> Option<Box<dyn AggState>> {
        None
    }
}

impl<T> PartitionAgg for ChunkedArray<T>
where
    T: PolarsNumericType + Sync,
    T::Native:
        std::ops::Add<Output = T::Native> + Num + NumCast + Bounded + Into<AnyValue<'static>>,
    ChunkedArray<T>: IntoSeries,
{
    fn part_agg_sum(&self, groups: &GroupedMap<Option<u64>>) -> Option<Box<dyn AggState>> {
        let agg: AggMap = POOL.install(|| {
            groups
                .into_par_iter()
                .map(|(k, (first, idx))| {
                    let agg = if idx.len() == 1 {
                        self.get(*first as usize)
                    } else {
                        match (self.null_count(), self.chunks.len()) {
                            (0, 1) => unsafe {
                                Some(take_agg_no_null_primitive_iter_unchecked(
                                    self.downcast_iter().next().unwrap(),
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| a + b,
                                    T::Native::zero(),
                                ))
                            },
                            (_, 1) => unsafe {
                                take_agg_primitive_iter_unchecked(
                                    self.downcast_iter().next().unwrap(),
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| a + b,
                                    T::Native::zero(),
                                )
                            },
                            _ => {
                                let take = unsafe {
                                    self.take_unchecked(idx.iter().map(|i| *i as usize).into())
                                };
                                take.sum()
                            }
                        }
                    };
                    (*k, (*first, agg.into()))
                })
                .collect()
        });

        Some(Box::new(SumAggState { agg: vec![agg] }))
    }
}
