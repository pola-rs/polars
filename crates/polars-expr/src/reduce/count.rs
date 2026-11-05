#![allow(unsafe_op_in_unsafe_fn)]
use polars_core::error::constants::LENGTH_LIMIT_MSG;

use super::*;

pub struct CountReduce {
    counts: Vec<u64>,
    layout: LaneLayout,
    evicted_counts: Vec<u64>,
    include_nulls: bool,
}

impl CountReduce {
    pub fn new(include_nulls: bool) -> Self {
        Self {
            counts: Vec::new(),
            layout: LaneLayout::single(0),
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
        self.layout
            .resize(&mut self.counts, num_groups as usize, 0, |a, b| *a += b);
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        let mut count = values.len();
        if !self.include_nulls {
            count -= values.null_count();
        }
        self.counts[group_idx as usize] += count as u64;
        Ok(())
    }

    unsafe fn update_groups_subset(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(subset.len() == group_idxs.len());
        let values = values.as_materialized_series(); // @scalar-opt
        let chunks = values.chunks();
        assert!(chunks.len() == 1);
        let arr = &*chunks[0];
        let valid = arr
            .validity()
            .filter(|_| arr.has_nulls() && !self.include_nulls);

        self.layout.expand(&mut self.counts, 0);
        let stride = self.layout.stride;
        let counts = self.counts.as_mut_slice();
        let mut offset = 0usize;
        if let Some(valid) = valid {
            for (i, g) in subset.iter().zip(group_idxs) {
                let slot = *g as usize + (offset & LANE_MASK);
                *counts.get_unchecked_mut(slot) += valid.get_bit_unchecked(*i as usize) as u64;
                offset = offset.wrapping_add(stride);
            }
        } else {
            for g in group_idxs {
                let slot = *g as usize + (offset & LANE_MASK);
                *counts.get_unchecked_mut(slot) += 1;
                offset = offset.wrapping_add(stride);
            }
        }
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(subset.len() == group_idxs.len());
        let values = values.as_materialized_series(); // @scalar-opt
        let chunks = values.chunks();
        assert!(chunks.len() == 1);
        let arr = &*chunks[0];
        self.layout.collapse(&mut self.counts, |a, b| *a += b);
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
        let other_lanes = other.layout.num_lanes();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let grp = self.counts.get_unchecked_mut(*g as usize);
                for l in 0..other_lanes {
                    *grp += *other
                        .counts
                        .get_unchecked(*i as usize + l * other.layout.stride);
                }
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        let counts = core::mem::take(&mut self.evicted_counts);
        Box::new(Self {
            layout: LaneLayout::single(counts.len()),
            counts,
            evicted_counts: Vec::new(),
            include_nulls: self.include_nulls,
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        self.layout.fold(&mut self.counts, |a, b| *a += b);
        self.layout = LaneLayout::single(0);
        let v: Vec<u64> = core::mem::take(&mut self.counts);
        let len = v.len();

        let v: Vec<IdxSize> = v
            .into_iter()
            .filter_map(|x| IdxSize::try_from(x).ok())
            .collect();

        polars_ensure!(
            v.len() == len,
            ComputeError:
            LENGTH_LIMIT_MSG
        );

        Ok(IdxCa::from_vec(PlSmallStr::EMPTY, v).into_series())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct NullCountReduce {
    counts: Vec<u64>,
    evicted_counts: Vec<u64>,
}

impl NullCountReduce {
    pub fn new() -> Self {
        Self {
            counts: Vec::new(),
            evicted_counts: Vec::new(),
        }
    }
}

impl GroupedReduction for NullCountReduce {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.counts.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.counts.resize(num_groups as usize, 0);
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        self.counts[group_idx as usize] += values.null_count() as u64;
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(subset.len() == group_idxs.len());
        let values = values.as_materialized_series(); // @scalar-opt
        let chunks = values.chunks();
        assert!(chunks.len() == 1);
        let arr = &*chunks[0];
        if arr.has_nulls() {
            let valid = arr.validity().unwrap();
            for (i, g) in subset.iter().zip(group_idxs) {
                let grp = self.counts.get_unchecked_mut(g.idx());
                if g.should_evict() {
                    self.evicted_counts.push(*grp);
                    *grp = 0;
                }
                *grp += (!valid.get_bit_unchecked(*i as usize)) as u64;
            }
        } else {
            for (_, g) in subset.iter().zip(group_idxs) {
                let grp = self.counts.get_unchecked_mut(g.idx());
                if g.should_evict() {
                    self.evicted_counts.push(*grp);
                    *grp = 0;
                }
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
