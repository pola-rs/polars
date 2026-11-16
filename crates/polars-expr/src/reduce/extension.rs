#![allow(unsafe_op_in_unsafe_fn)]
use super::*;

struct StorageReduction {
    inner: Box<dyn GroupedReduction>,
    dtype: DataType,
}

/// Runs `inner` on the storage of the inputs, giving values of `dtype`.
pub fn new_storage_reduction(
    inner: Box<dyn GroupedReduction>,
    dtype: DataType,
) -> Box<dyn GroupedReduction> {
    Box::new(StorageReduction { inner, dtype })
}

fn storage_columns(values: &[&Column]) -> Vec<Column> {
    values
        .iter()
        .map(|c| {
            if c.dtype().is_extension() {
                c.as_materialized_series()
                    .to_storage()
                    .clone()
                    .into_column()
            } else {
                (*c).clone()
            }
        })
        .collect()
}

impl GroupedReduction for StorageReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        new_storage_reduction(self.inner.new_empty(), self.dtype.clone())
    }

    fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.inner.resize(num_groups);
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        seq_id: u64,
    ) -> PolarsResult<()> {
        let values = storage_columns(values);
        let values = values.iter().collect::<Vec<_>>();
        self.inner.update_group(&values, group_idx, seq_id)
    }

    unsafe fn update_groups_subset(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
        seq_id: u64,
    ) -> PolarsResult<()> {
        let values = storage_columns(values);
        let values = values.iter().collect::<Vec<_>>();
        self.inner
            .update_groups_subset(&values, subset, group_idxs, seq_id)
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        seq_id: u64,
    ) -> PolarsResult<()> {
        let values = storage_columns(values);
        let values = values.iter().collect::<Vec<_>>();
        self.inner
            .update_groups_while_evicting(&values, subset, group_idxs, seq_id)
    }

    unsafe fn combine_subset(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.inner
            .combine_subset(other.inner.as_ref(), subset, group_idxs)
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        new_storage_reduction(self.inner.take_evictions(), self.dtype.clone())
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let values = self.inner.finalize()?;
        Ok(match &self.dtype {
            DataType::Extension(typ, _) => values.into_extension(typ.clone()),
            _ => values,
        })
    }

    fn state_size(&self) -> Option<StateSize> {
        self.inner.state_size()
    }

    fn num_fixed_states(&self) -> usize {
        self.inner.num_fixed_states()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
