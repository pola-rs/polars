use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::IntoColumn;
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_expr::groups::Grouper;
use polars_expr::reduce::GroupedReduction;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::InputPayload;
use super::pre_agg::PreAgg;
use crate::execute::StreamingExecutionState;
use crate::hash_shuffle::frame_with_keys::FrameWithKeys;

/// What the sink and every partition table read.
pub struct GroupByParams {
    pub key_schema: Arc<Schema>,
    pub output_schema: Arc<Schema>,
    /// Input stream i feeds grouped_reductions[k] for each k in reductions_per_input[i].
    pub reductions_per_input: Vec<Vec<usize>>,
    /// grouped_reductions[k] is passed input cols grouped_reduction_cols[k].
    pub grouped_reduction_cols: Vec<Vec<PlSmallStr>>,
    pub payload_per_input: Vec<InputPayload>,
    /// The template for the grouper of each partition table.
    pub grouper: Box<dyn Grouper>,
    /// The templates for the reductions of each hot table and partition table.
    pub grouped_reductions: Vec<Box<dyn GroupedReduction>>,
    pub has_order_sensitive_agg: bool,
}

impl GroupByParams {
    /// Adds the fused columns of the input to a frame of its cold rows, evaluated once on all
    /// its rows.
    pub async fn prepare_frame(
        &self,
        input: usize,
        mut frame: FrameWithKeys,
        state: &StreamingExecutionState,
    ) -> PolarsResult<FrameWithKeys> {
        self.payload_per_input[input]
            .push_fused(&mut frame.frame, &state.in_memory_exec_state)
            .await?;
        Ok(frame)
    }
}

/// `rows`, or else the first `len` row indices, from `all_rows`.
fn rows_or_all<'a>(
    rows: Option<&'a [IdxSize]>,
    all_rows: &'a mut Vec<IdxSize>,
    len: usize,
) -> &'a [IdxSize] {
    match rows {
        Some(rows) => rows,
        None => {
            if all_rows.len() < len {
                all_rows.extend(all_rows.len() as IdxSize..len as IdxSize);
            }
            &all_rows[..len]
        },
    }
}

/// The groups and reduction states of one partition.
pub struct PartitionTable {
    params: Arc<GroupByParams>,
    grouper: Box<dyn Grouper>,
    reductions: Vec<Box<dyn GroupedReduction>>,
    group_idxs: Vec<IdxSize>,
    all_rows: Vec<IdxSize>,
}

impl PartitionTable {
    pub fn new(params: Arc<GroupByParams>, reserve_groups: usize) -> Self {
        let mut grouper = params.grouper.new_empty();
        grouper.reserve(reserve_groups);
        let reductions = params
            .grouped_reductions
            .iter()
            .map(|r| {
                let mut r = r.new_empty();
                r.reserve(reserve_groups);
                r
            })
            .collect();
        Self {
            params,
            grouper,
            reductions,
            group_idxs: Vec::new(),
            all_rows: Vec::new(),
        }
    }

    /// Combines the groups `groups` (all if `None`) of a pre-aggregate into this table.
    pub fn add_pre_agg(
        &mut self,
        pre_agg: &PreAgg,
        groups: Option<&[IdxSize]>,
    ) -> PolarsResult<()> {
        let groups = rows_or_all(groups, &mut self.all_rows, pre_agg.keys.len());
        self.group_idxs.clear();
        unsafe {
            self.grouper
                .insert_keys_subset(&pre_agg.keys, groups, Some(&mut self.group_idxs));
        }
        for (pre_reduction, r_idx) in pre_agg.reductions.iter().zip(pre_agg.reduction_idxs.iter()) {
            let r = &mut self.reductions[*r_idx];
            r.resize(self.grouper.num_groups());
            unsafe { r.combine_subset(&**pre_reduction, groups, &self.group_idxs)? };
        }
        Ok(())
    }

    /// Inserts the rows `rows` (all if `None`) of a prepared frame of `input` into this table.
    pub fn add_rows(
        &mut self,
        input: usize,
        frame: &FrameWithKeys,
        rows: Option<&[IdxSize]>,
    ) -> PolarsResult<()> {
        let rows = rows_or_all(rows, &mut self.all_rows, frame.keys.len());
        self.group_idxs.clear();
        unsafe {
            self.grouper
                .insert_keys_subset(&frame.keys, rows, Some(&mut self.group_idxs));
        }
        let num_groups = self.grouper.num_groups();
        let params = &*self.params;
        let mut in_cols = Vec::new();
        for r_idx in &params.reductions_per_input[input] {
            let r = &mut self.reductions[*r_idx];
            r.resize(num_groups);
            in_cols.clear();
            in_cols.extend(
                params.grouped_reduction_cols[*r_idx]
                    .iter()
                    .map(|col| frame.frame.column(col).unwrap()),
            );
            // Cold rows only exist without order-sensitive reductions.
            unsafe { r.update_groups_subset(&in_cols, rows, &self.group_idxs, 0)? };
        }
        Ok(())
    }

    /// The keys in group order followed by the finalized reductions. Leaves the table empty.
    pub fn finish(&mut self) -> PolarsResult<DataFrame> {
        let grouper = std::mem::replace(&mut self.grouper, self.params.grouper.new_empty());
        let num_groups = grouper.num_groups();
        let mut out = grouper.get_keys_in_group_order(&self.params.key_schema);
        drop(grouper);
        let out_names = self.params.output_schema.iter_names().skip(out.width());
        for (r, name) in self.reductions.iter_mut().zip(out_names) {
            // Each input only resizes its own reductions.
            r.resize(num_groups);
            let c = r.finalize()?.with_name(name.clone()).into_column();
            unsafe { out.push_column_unchecked(c) };
        }
        Ok(out)
    }
}
