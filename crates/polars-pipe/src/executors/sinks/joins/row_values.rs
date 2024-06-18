use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, StaticArray};
use arrow::compute::utils::combine_validities_and_many;
use polars_core::error::PolarsResult;
use polars_row::RowsEncoded;

use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, PExecutionContext};

#[derive(Clone)]
pub(super) struct RowValues {
    current_rows: RowsEncoded,
    join_column_eval: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    join_columns_material: Vec<ArrayRef>,
    // Location of join columns.
    // These column locations need to be dropped from the rhs
    pub join_column_idx: Option<Vec<usize>>,
    det_join_idx: bool,
}

impl RowValues {
    pub(super) fn new(
        join_column_eval: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        det_join_idx: bool,
    ) -> Self {
        Self {
            current_rows: Default::default(),
            join_column_eval,
            join_column_idx: None,
            join_columns_material: vec![],
            det_join_idx,
        }
    }

    pub(super) fn clear(&mut self) {
        self.join_columns_material.clear();
    }

    pub(super) fn get_values(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
        join_nulls: bool,
    ) -> PolarsResult<BinaryArray<i64>> {
        // Memory should already be cleared on previous iteration.
        debug_assert!(self.join_columns_material.is_empty());
        let determine_idx = self.det_join_idx && self.join_column_idx.is_none();
        let mut names = vec![];

        for phys_e in self.join_column_eval.iter() {
            let s = phys_e.evaluate(chunk, &context.execution_state)?;
            let mut s = s.to_physical_repr().rechunk();
            if chunk.data.is_empty() {
                s = s.clear()
            };
            if determine_idx {
                names.push(s.name().to_string());
            }
            self.join_columns_material.push(s.array_ref(0).clone());
        }

        // We determine the indices of the columns that have to be removed
        // if swapped the join column is already removed from the `build_df` as that will
        // be the rhs one.
        if determine_idx {
            let mut idx = names
                .iter()
                .filter_map(|name| chunk.data.get_column_index(name))
                .collect::<Vec<_>>();
            // Ensure that it is sorted so that we can later remove columns in
            // a predictable order
            idx.sort_unstable();
            self.join_column_idx = Some(idx);
        }
        polars_row::convert_columns_amortized_no_order(
            &self.join_columns_material,
            &mut self.current_rows,
        );

        // SAFETY: we keep rows-encode alive
        let array = unsafe { self.current_rows.borrow_array() };
        Ok(if join_nulls {
            array
        } else {
            let validities = self
                .join_columns_material
                .iter()
                .map(|arr| arr.validity())
                .collect::<Vec<_>>();
            let validity = combine_validities_and_many(&validities);
            array.with_validity_typed(validity)
        })
    }
}
