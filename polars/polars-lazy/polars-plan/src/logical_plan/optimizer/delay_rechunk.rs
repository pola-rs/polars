use std::collections::BTreeSet;

use polars_utils::arena::{Arena, Node};

use super::*;

#[derive(Default)]
pub(super) struct DelayRechunk {
    processed: BTreeSet<usize>,
}

impl DelayRechunk {
    pub(super) fn new() -> Self {
        Default::default()
    }
}

impl OptimizationRule for DelayRechunk {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        _expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        match lp_arena.get(node) {
            // An aggregation can be partitioned, its wasteful to rechunk before that partition.
            ALogicalPlan::Aggregate { input, .. } => {
                if !self.processed.insert(node.0) {
                    return None;
                };

                use ALogicalPlan::*;
                let mut input_node = None;
                let mut union_parent = None;
                let mut previous_node = *input;
                for (node, lp) in (&*lp_arena).iter(*input) {
                    match lp {
                        // we get the input node
                        #[cfg(feature = "parquet")]
                        ParquetScan { .. } => {
                            input_node = Some(node);
                            break;
                        }
                        #[cfg(feature = "csv-file")]
                        CsvScan { .. } => {
                            input_node = Some(node);
                            break;
                        }
                        #[cfg(feature = "ipc")]
                        IpcScan { .. } => {
                            input_node = Some(node);
                            break;
                        }
                        Union { .. } => union_parent = Some(previous_node),
                        // don't delay rechunk if there is a join first
                        Join { .. } => break,
                        _ => {}
                    }
                    previous_node = node;
                }

                if let Some(node) = input_node {
                    match lp_arena.get_mut(node) {
                        #[cfg(feature = "csv-file")]
                        CsvScan { options, .. } => {
                            options.rechunk = false;
                        }
                        #[cfg(feature = "parquet")]
                        ParquetScan { options, .. } => options.rechunk = false,
                        #[cfg(feature = "ipc")]
                        IpcScan { options, .. } => {
                            options.rechunk = false;
                        }
                        _ => unreachable!(),
                    }
                };
                if let Some(parent_node) = union_parent {
                    // remove the rechunk function
                    if let MapFunction {
                        input,
                        function: FunctionNode::Rechunk,
                        ..
                    } = lp_arena.get(parent_node)
                    {
                        lp_arena.swap(*input, parent_node)
                    }
                }

                None
            }
            _ => None,
        }
    }
}
