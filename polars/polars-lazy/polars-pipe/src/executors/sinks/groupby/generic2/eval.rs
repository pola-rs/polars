use std::cell::UnsafeCell;

use polars_core::export::ahash::RandomState;

use super::*;
use crate::executors::sinks::utils::hash_series;
use crate::expressions::PhysicalPipedExpr;

pub(super) struct Eval {
    // the keys that will be aggregated on
    key_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    // the columns that will be aggregated
    aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    hb: RandomState,
    // amortize allocations
    aggregation_series: UnsafeCell<Vec<Series>>,
    keys_series: UnsafeCell<Vec<Series>>,
    hashes: Vec<u64>,
}

impl Eval {
    pub(super) fn new(
        key_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    ) -> Self {
        let hb = RandomState::default();
        Self {
            key_columns,
            aggregation_columns,
            hb,
            aggregation_series: Default::default(),
            keys_series: Default::default(),
            hashes: Default::default(),
        }
    }
    pub(super) fn split(&self) -> Self {
        Self {
            key_columns: self.key_columns.clone(),
            aggregation_columns: self.aggregation_columns.clone(),
            hb: self.hb.clone(),
            aggregation_series: Default::default(),
            keys_series: Default::default(),
            hashes: Default::default(),
        }
    }

    pub(super) unsafe fn clear(&mut self) {
        let keys_series = &mut *self.keys_series.get();
        let aggregation_series = &mut *self.aggregation_series.get();
        keys_series.clear();
        aggregation_series.clear();
        self.hashes.clear();
    }

    pub(super) unsafe fn evaluate_keys_aggs_and_hashes(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<()> {
        let keys_series = &mut *self.keys_series.get();
        let aggregation_series = &mut *self.aggregation_series.get();

        for phys_e in self.aggregation_columns.iter() {
            let s = phys_e.evaluate(chunk, context.execution_state.as_any())?;
            let s = s.to_physical_repr();
            aggregation_series.push(s.rechunk());
        }
        for phys_e in self.key_columns.iter() {
            let s = phys_e.evaluate(chunk, context.execution_state.as_any())?;
            let s = s.to_physical_repr();
            keys_series.push(s.rechunk());
        }

        // write the hashes to self.hashes buffer
        hash_series(&keys_series, &mut self.hashes, &self.hb);
        Ok(())
    }

    pub(super) unsafe fn get_keys_iters(&self) -> Vec<SeriesPhysIter> {
        let keys_series = &*self.keys_series.get();
        keys_series.iter().map(|s| s.phys_iter()).collect()
    }
    pub(super) unsafe fn get_aggs_iters(&self) -> Vec<SeriesPhysIter> {
        let aggregation_series = &*self.aggregation_series.get();
        aggregation_series.iter().map(|s| s.phys_iter()).collect()
    }
}
