#![allow(unsafe_op_in_unsafe_fn)]
use polars_core::error::constants::LENGTH_LIMIT_MSG;

use super::*;

pub struct CountReduce {
    counts: Vec<u64>,
    evicted_counts: Vec<u64>,
    include_nulls: bool,
}

impl CountReduce {
    pub fn new(include_nulls: bool) -> Self {
        Self {
            counts: Vec::new(),
            evicted_counts: Vec::new(),
            include_nulls,
        }
    }
}

impl GroupedReduction for CountReduce {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(self.include_nulls))
    }

    fn reserve(&mut self, additional: usize) {
        self.counts.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.counts.resize(num_groups as usize, 0);
    }

    fn update_group(
        &mut self,
        values: &Column,
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        let mut count = values.len();
        if !self.include_nulls {
            count -= values.null_count();
        }
        self.counts[group_idx as usize] += count as u64;
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &Column,
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(subset.len() == group_idxs.len());
        let values = values.as_materialized_series(); // @scalar-opt
        let chunks = values.chunks();
        assert!(chunks.len() == 1);
        let arr = &*chunks[0];
        if arr.has_nulls() && !self.include_nulls {
            let valid = arr.validity().unwrap();
            for (i, g) in subset.iter().zip(group_idxs) {
                let grp = self.counts.get_unchecked_mut(g.idx());
                if g.should_evict() {
                    self.evicted_counts.push(*grp);
                    *grp = 0;
                }
                *grp += valid.get_bit_unchecked(*i as usize) as u64;
            }
        } else {
            for (_, g) in subset.iter().zip(group_idxs) {
                let grp = self.counts.get_unchecked_mut(g.idx());
                if g.should_evict() {
                    self.evicted_counts.push(*grp);
                    *grp = 0;
                }
                *grp += 1;
            }
        }
        Ok(())
    }

    unsafe fn combine_subset(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        assert!(subset.len() == group_idxs.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                *self.counts.get_unchecked_mut(*g as usize) +=
                    *other.counts.get_unchecked(*i as usize);
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            counts: core::mem::take(&mut self.evicted_counts),
            evicted_counts: Vec::new(),
            include_nulls: self.include_nulls,
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let ca: IdxCa = self
            .counts
            .drain(..)
            .map(|l| IdxSize::try_from(l).expect(LENGTH_LIMIT_MSG))
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
