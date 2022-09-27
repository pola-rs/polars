use polars_utils::arena::{Arena, Node};

use super::*;

pub(super) struct DelayRechunk {}

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
                use ALogicalPlan::*;
                let mut input_node = None;
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

                        // don't delay rechunk if there is a join first
                        Join { .. } => break,
                        _ => {}
                    }
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

                None
            }
            _ => None,
        }
    }
}
