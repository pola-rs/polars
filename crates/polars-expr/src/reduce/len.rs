#![allow(unsafe_op_in_unsafe_fn)]
use polars_core::error::constants::LENGTH_LIMIT_MSG;

use super::*;

#[derive(Default)]
pub struct LenReduce {
    groups: Vec<u64>,
    evictions: Vec<u64>,
}

impl GroupedReduction for LenReduce {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::default())
    }

    fn reserve(&mut self, additional: usize) {
        self.groups.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.groups.resize(num_groups as usize, 0);
    }

    fn update_group(
        &mut self,
        values: &Column,
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        self.groups[group_idx as usize] += values.len() as u64;
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        _values: &Column,
        _subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for g in group_idxs.iter() {
                let grp = self.groups.get_unchecked_mut(g.idx());
                if g.should_evict() {
                    self.evictions.push(*grp);
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
                *self.groups.get_unchecked_mut(*g as usize) +=
                    *other.groups.get_unchecked(*i as usize);
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            groups: core::mem::take(&mut self.evictions),
            evictions: Vec::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let ca: IdxCa = self
            .groups
            .drain(..)
            .map(|l| IdxSize::try_from(l).expect(LENGTH_LIMIT_MSG))
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
