//! Prune `FileScanIR::Parquet::metadata` to projected + predicate columns.
//!
//! Runs at the end of the optimizer pipeline, after projection and predicate
//! pushdown have resolved each scan's projection / predicate. Walks the IR
//! arena and replaces each parquet scan's metadata via
//! `FileMetadata::pruned`.
//!
//! Gated on `POLARS_PRUNE_PARQUET_METADATA=1` (default off; single-node
//! execution gets no benefit since metadata never crosses a wire).

use polars_utils::arena::{Arena, Node};
use polars_utils::unitvec;

#[cfg(feature = "parquet")]
use crate::dsl::FileScanIR;
#[cfg(feature = "parquet")]
use crate::plans::AExpr;
use crate::plans::IR;
#[cfg(feature = "parquet")]
use crate::utils::aexpr_to_leaf_names_iter;

#[cfg(feature = "parquet")]
pub(super) fn prune_parquet_metadata(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) {
    // Gate on env var; default off (single-node has no benefit).
    if !polars_config::config().prune_parquet_metadata() {
        return;
    }

    let mut stack = unitvec![root];

    while let Some(node) = stack.pop() {
        ir_arena.get(node).copy_inputs(&mut stack);

        let IR::Scan {
            scan_type,
            unified_scan_args,
            predicate,
            ..
        } = ir_arena.get_mut(node)
        else {
            continue;
        };

        let Some(projection) = unified_scan_args.projection.clone() else {
            continue;
        };

        // Extract predicate column names. Filter to those also in the
        // projection (post-pushdown predicate cols are typically ⊆
        // projection, but be defensive).
        let predicate_cols: Vec<polars_utils::pl_str::PlSmallStr> = predicate
            .as_ref()
            .map(|pred_ir| {
                aexpr_to_leaf_names_iter(pred_ir.node(), expr_arena)
                    .filter(|name| projection.iter().any(|p| p == name.as_str()))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        let FileScanIR::Parquet {
            metadata: Some(meta_arc),
            ..
        } = scan_type.as_mut()
        else {
            continue;
        };

        // Pruning failure (chunks-vs-leaves desync inside `from_compact`)
        // is recoverable; fall back to unpruned metadata so the query still
        // runs. The wire form just stays larger for this scan.
        let Ok(pruned) = meta_arc.pruned(&projection, &predicate_cols) else {
            continue;
        };

        // Only swap if the pruned form is actually smaller (number of leaves).
        // Avoids allocating a new Arc when nothing changed.
        if pruned.schema_descr.columns().len() < meta_arc.schema_descr.columns().len() {
            *meta_arc = std::sync::Arc::new(pruned);
        }
    }
}

#[cfg(not(feature = "parquet"))]
pub(super) fn prune_parquet_metadata(
    _root: Node,
    _ir_arena: &mut Arena<IR>,
    _expr_arena: &Arena<crate::plans::AExpr>,
) {
    // No-op when parquet feature is disabled.
}
