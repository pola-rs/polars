//! Sample pushdown optimization.
//!
//! This optimization pushes `FunctionIR::Sample` operations into `IR::Scan` nodes,
//! allowing sampling to happen at the scan level for better memory efficiency.
//! 
//! When sample is pushed to the Scan node, the parquet reader can use pre-filtered
//! decode to only decode columns for sampled rows, dramatically reducing memory usage.

use polars_utils::arena::{Arena, Node};

use crate::dsl::ScanSampleArgs;
use crate::plans::ir::IR;
use crate::plans::{AExpr, FunctionIR};

/// Push sample operations into scan nodes.
pub fn sample_pushdown(root: Node, ir_arena: &mut Arena<IR>, _expr_arena: &Arena<AExpr>) {
    let mut stack = vec![root];

    while let Some(node) = stack.pop() {
        let ir = ir_arena.get(node);

        // Skip Invalid nodes (left over from previous swaps in this pass)
        if matches!(ir, IR::Invalid) {
            continue;
        }

        ir.copy_inputs(&mut stack);

        // Extract sample info if this is a MapFunction(Sample)
        let sample_info: Option<(Node, f64, bool, u64)> = {
            let ir = ir_arena.get(node);
            if let IR::MapFunction { input, function } = ir {
                if let FunctionIR::Sample {
                    fraction,
                    with_replacement,
                    seed,
                } = function
                {
                    Some((*input, *fraction, *with_replacement, seed.unwrap_or(0)))
                } else {
                    None
                }
            } else {
                None
            }
        };

        // If we found a Sample, check if input is a Scan and push down
        if let Some((input_node, fraction, with_replacement, seed)) = sample_info {
            // Check if input is a Scan that doesn't already have a sample
            let can_push = {
                let input_ir = ir_arena.get(input_node);
                matches!(input_ir, IR::Scan { unified_scan_args, .. } if unified_scan_args.sample.is_none())
            };

            if can_push {
                // Set sample in the Scan's unified_scan_args
                if let IR::Scan {
                    unified_scan_args, ..
                } = ir_arena.get_mut(input_node)
                {
                    unified_scan_args.sample = Some(ScanSampleArgs {
                        fraction,
                        with_replacement,
                        seed,
                    });
                }

                // Replace the MapFunction node with the Scan (skip the Sample)
                let scan = ir_arena.take(input_node);
                ir_arena.replace(node, scan);
            }
        }
    }
}
