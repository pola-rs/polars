use std::collections::BTreeSet;

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
        lp_arena: &mut Arena<IR>,
        _expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<IR> {
        match lp_arena.get(node) {
            // An aggregation can be partitioned, its wasteful to rechunk before that partition.
            #[allow(unused_mut)]
            IR::GroupBy { input, keys, .. } => {
                // Multiple keys on multiple chunks is much slower, so rechunk.
                if !self.processed.insert(node.0) || keys.len() > 1 {
                    return None;
                };

                use IR::*;
                let mut input_node = None;
                for (node, lp) in (&*lp_arena).iter(*input) {
                    match lp {
                        Scan { .. } => {
                            input_node = Some(node);
                            break;
                        },
                        Union { .. } => {
                            input_node = Some(node);
                            break;
                        },
                        // don't delay rechunk if there is a join first
                        Join { .. } => break,
                        _ => {},
                    }
                }

                if let Some(node) = input_node {
                    match lp_arena.get_mut(node) {
                        Scan {
                            file_options: options,
                            ..
                        } => {
                            options.rechunk = false;
                        },
                        Union { options, .. } => {
                            options.rechunk = false;
                        },
                        _ => unreachable!(),
                    }
                };

                None
            },
            _ => None,
        }
    }
}
