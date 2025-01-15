use polars_core::error::constants::LENGTH_LIMIT_MSG;

use super::*;
use crate::reduce::partition::partition_vec;

#[derive(Default)]
pub struct LenReduce {
    groups: Vec<u64>,
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
        values: &Series,
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        self.groups[group_idx as usize] += values.len() as u64;
        Ok(())
    }

    unsafe fn update_groups(
        &mut self,
        values: &Series,
        group_idxs: &[IdxSize],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.len() == group_idxs.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for g in group_idxs.iter() {
                *self.groups.get_unchecked_mut(*g as usize) += 1;
            }
        }
        Ok(())
    }

    unsafe fn combine(
        &mut self,
        other: &dyn GroupedReduction,
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        assert!(other.groups.len() == group_idxs.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, v) in group_idxs.iter().zip(other.groups.iter()) {
                *self.groups.get_unchecked_mut(*g as usize) += v;
            }
        }
        Ok(())
    }

    unsafe fn gather_combine(
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

    fn finalize(&mut self) -> PolarsResult<Series> {
        let ca: IdxCa = self
            .groups
            .drain(..)
            .map(|l| IdxSize::try_from(l).expect(LENGTH_LIMIT_MSG))
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }

    unsafe fn partition(
        self: Box<Self>,
        partition_sizes: &[IdxSize],
        partition_idxs: &[IdxSize],
    ) -> Vec<Box<dyn GroupedReduction>> {
        partition_vec(self.groups, partition_sizes, partition_idxs)
            .into_iter()
            .map(|groups| Box::new(Self { groups }) as _)
            .collect()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
