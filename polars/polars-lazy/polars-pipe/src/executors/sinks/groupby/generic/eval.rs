use std::cell::UnsafeCell;

use polars_arrow::export::arrow::array::BinaryArray;
use polars_core::export::ahash::RandomState;
use polars_row::SortField;

use super::*;
use crate::executors::sinks::utils::hash_rows;
use crate::expressions::PhysicalPipedExpr;

pub(super) struct Eval {
    // the keys that will be aggregated on
    key_columns_expr: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    // the columns that will be aggregated
    aggregation_columns_expr: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    hb: RandomState,
    // amortize allocations
    aggregation_series: UnsafeCell<Vec<Series>>,
    keys_columns: UnsafeCell<Vec<ArrayRef>>,
    hashes: Vec<u64>,
    key_fields: Vec<SortField>,
    keys_array: Option<BinaryArray<i64>>,
}

impl Eval {
    pub(super) fn new(
        key_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    ) -> Self {
        let hb = RandomState::default();
        Self {
            key_columns_expr: key_columns,
            aggregation_columns_expr: aggregation_columns,
            hb,
            aggregation_series: Default::default(),
            keys_columns: Default::default(),
            hashes: Default::default(),
            key_fields: Default::default(),
            keys_array: None,
        }
    }
    pub(super) fn split(&self) -> Self {
        Self {
            key_columns_expr: self.key_columns_expr.clone(),
            aggregation_columns_expr: self.aggregation_columns_expr.clone(),
            hb: self.hb.clone(),
            aggregation_series: Default::default(),
            keys_columns: Default::default(),
            hashes: Default::default(),
            key_fields: vec![Default::default(); self.key_columns_expr.len()],
            keys_array: None,
        }
    }

    pub(super) unsafe fn clear(&mut self) {
        let keys_series = &mut *self.keys_columns.get();
        let aggregation_series = &mut *self.aggregation_series.get();
        keys_series.clear();
        aggregation_series.clear();
        self.hashes.clear();
        self.keys_array = None;
    }

    pub(super) unsafe fn evaluate_keys_aggs_and_hashes(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<()> {
        let keys_columns = &mut *self.keys_columns.get();
        let aggregation_series = &mut *self.aggregation_series.get();

        for phys_e in self.aggregation_columns_expr.iter() {
            let s = phys_e.evaluate(chunk, context.execution_state.as_any())?;
            let s = s.to_physical_repr();
            aggregation_series.push(s.rechunk());
        }
        for phys_e in self.key_columns_expr.iter() {
            let s = phys_e.evaluate(chunk, context.execution_state.as_any())?;
            let s = match s.dtype() {
                // todo! add binary to phyical repr?
                DataType::Utf8 => unsafe { s.cast_unchecked(&DataType::Binary).unwrap() },
                _ => s.to_physical_repr().rechunk(),
            };
            keys_columns.push(s.to_arrow(0));
        }

        let key_rows = polars_row::convert_columns(keys_columns, &self.key_fields);
        let keys_array = key_rows.into_array();
        // drop the series, all data is in the rows encoding now
        keys_columns.clear();

        // write the hashes to self.hashes buffer
        hash_rows(&keys_array, &mut self.hashes, &self.hb);
        self.keys_array = Some(keys_array);
        Ok(())
    }

    pub(super) fn get_keys_iter(&self) -> &BinaryArray<i64> {
        self.keys_array.as_ref().unwrap()
    }
    pub(super) unsafe fn get_aggs_iters(&self) -> Vec<SeriesPhysIter> {
        let aggregation_series = &*self.aggregation_series.get();
        aggregation_series.iter().map(|s| s.phys_iter()).collect()
    }

    pub(super) fn hashes(&self) -> &[u64] {
        &self.hashes
    }
}
