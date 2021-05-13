use crate::chunked_array::kernels::take_agg::{
    take_agg_no_null_primitive_iter_unchecked, take_agg_primitive_iter_unchecked,
};
use crate::frame::groupby::{fmt_groupby_column, GroupByMethod, GroupedMap};
use crate::prelude::*;
use crate::utils::{CustomIterTools, NoNull};
use crate::POOL;
use ahash::RandomState;
use arrow::ipc::Utf8;
use hashbrown::HashMap;
use num::{Bounded, Num, NumCast, Zero};
use rayon::prelude::*;
use std::any::Any;

pub trait AggState {
    fn merge(&mut self, other: Vec<Box<dyn AggState>>);
    fn as_any(&mut self) -> &mut dyn Any;
    fn finish(&mut self) -> Series;
}

// u32: an index of the group (can be used to get the keys)
// AnyValue: the aggregation
pub type AggMap = HashMap<Option<u64>, (u32, AnyValue<'static>), RandomState>;

pub struct SumAggState {
    agg: Vec<AggMap>,
    dtype: DataType,
    name: String,
}

impl AggState for SumAggState {
    fn merge(&mut self, mut other: Vec<Box<dyn AggState>>) {
        let mut finished = Vec::with_capacity(other.len() + 1);
        let mut stack = Vec::with_capacity(other.len());
        stack.push(&mut self.agg[0]);

        // README! we revert the order so that they are added to the stack
        // in the original order. The ordering is important to make first,
        // and last work
        // TODO! Have this logic somewhere more global.
        let mut other: Vec<_> = other
            .iter_mut()
            .rev()
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

    fn finish(&mut self) -> Series {
        let mut s: Series = match self.dtype {
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
        };

        let out_name = fmt_groupby_column(&*self.name, GroupByMethod::Sum);
        s.rename(&out_name);
        s
    }
}

pub trait PartitionAgg {
    fn part_agg_sum(&self, _groups: &GroupedMap<Option<u64>>) -> Option<Box<dyn AggState>> {
        None
    }
}

impl PartitionAgg for BooleanChunked {}
impl PartitionAgg for Utf8Chunked {}
impl PartitionAgg for ListChunked {}
impl PartitionAgg for CategoricalChunked {}
#[cfg(feature = "object")]
impl<T> PartitionAgg for ObjectChunked<T> {}

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

        Some(Box::new(SumAggState {
            agg: vec![agg],
            name: self.name().to_string(),
            dtype: self.dtype().clone(),
        }))
    }
}
