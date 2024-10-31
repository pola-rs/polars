use std::cell::UnsafeCell;

use polars_row::{EncodingField, RowsEncoded};

use super::*;
use crate::executors::sinks::group_by::utils::prepare_key;
use crate::executors::sinks::utils::hash_rows;
use crate::expressions::PhysicalPipedExpr;

pub(super) struct Eval {
    // the keys that will be aggregated on
    key_columns_expr: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    // the columns that will be aggregated
    aggregation_columns_expr: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    hb: PlRandomState,
    // amortize allocations
    aggregation_series: UnsafeCell<Vec<Series>>,
    keys_columns: UnsafeCell<Vec<ArrayRef>>,
    hashes: Vec<u64>,
    key_fields: Vec<EncodingField>,
    // amortizes the encoding buffers
    rows_encoded: RowsEncoded,
}

impl Eval {
    pub(super) fn new(
        key_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    ) -> Self {
        let hb = PlRandomState::default();
        Self {
            key_columns_expr: key_columns,
            aggregation_columns_expr: aggregation_columns,
            hb,
            aggregation_series: Default::default(),
            keys_columns: Default::default(),
            hashes: Default::default(),
            key_fields: Default::default(),
            rows_encoded: Default::default(),
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
            rows_encoded: Default::default(),
        }
    }

    pub(super) unsafe fn clear(&mut self) {
        let keys_series = &mut *self.keys_columns.get();
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
        let keys_columns = &mut *self.keys_columns.get();
        let aggregation_series = &mut *self.aggregation_series.get();

        for phys_e in self.aggregation_columns_expr.iter() {
            let s = phys_e.evaluate(chunk, &context.execution_state)?;
            let s = s.to_physical_repr();
            aggregation_series.push(s.into_owned());
        }
        for phys_e in self.key_columns_expr.iter() {
            let s = phys_e.evaluate(chunk, &context.execution_state)?;
            let s = match s.dtype() {
                // todo! add binary to physical repr?
                DataType::String => unsafe { s.cast_unchecked(&DataType::Binary).unwrap() },
                _ => s.to_physical_repr().into_owned(),
            };
            let s = prepare_key(&s, chunk);
            keys_columns.push(s.to_arrow(0, CompatLevel::newest()));
        }

        polars_row::convert_columns_amortized(
            keys_columns,
            &self.key_fields,
            &mut self.rows_encoded,
        );
        // drop the series, all data is in the rows encoding now
        keys_columns.clear();

        // write the hashes to self.hashes buffer
        let keys_array = self.rows_encoded.borrow_array();
        hash_rows(&keys_array, &mut self.hashes, &self.hb);
        Ok(())
    }

    /// # Safety
    /// Caller must ensure `self.rows_encoded` stays alive as the lifetime
    /// is bound to the returned array.
    pub(super) unsafe fn get_keys_iter(&self) -> BinaryArray<i64> {
        self.rows_encoded.borrow_array()
    }
    pub(super) unsafe fn get_aggs_iters(&self) -> Vec<SeriesPhysIter> {
        let aggregation_series = &*self.aggregation_series.get();
        aggregation_series.iter().map(|s| s.phys_iter()).collect()
    }

    pub(super) fn hashes(&self) -> &[u64] {
        &self.hashes
    }
}
