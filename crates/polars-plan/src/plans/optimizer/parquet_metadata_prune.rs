//! Prune `FileScanIR::Parquet`'s pre-decoded metadata (`first_metadata`
//! and `metadata_per_source`) to projected + predicate columns.
//!
//! Runs at the end of the optimizer pipeline, after projection and predicate
//! pushdown have resolved each scan's projection / predicate. Walks the IR
//! arena and replaces each entry via `FileMetadata::pruned`.
//!
//! Gated on `POLARS_PRUNE_PARQUET_METADATA=1` (default off; single-node
//! execution gets no benefit since metadata never crosses a wire).

#[cfg(feature = "parquet")]
use std::sync::Arc;

#[cfg(feature = "parquet")]
use polars_io::parquet::metadata::FileMetadataRef;
use polars_utils::arena::{Arena, Node};
#[cfg(feature = "parquet")]
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "parquet")]
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

        // Variant check first: skip the projection/predicate work entirely
        // for non-Parquet scans (CSV, IPC, NDJson, Python, Anonymous).
        let FileScanIR::Parquet {
            first_metadata,
            metadata_per_source,
            ..
        } = scan_type.as_mut()
        else {
            continue;
        };

        let Some(projection) = unified_scan_args.projection.clone() else {
            continue;
        };

        // Predicate cols are a subset of the projection by this point:
        // projection pushdown adds predicate-referenced cols to the scan
        // (else the pushed-down predicate would `ColumnNotFound` at execute).
        let predicate_cols: Vec<PlSmallStr> = predicate
            .as_ref()
            .map(|pred_ir| {
                aexpr_to_leaf_names_iter(pred_ir.node(), expr_arena)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        // Pruning failure (chunks-vs-leaves desync inside `from_compact`) is
        // recoverable; fall back to unpruned metadata so the query still runs.
        // The wire form just stays larger for this entry. Swap-only-if-smaller
        // (leaf count) avoids allocating a new inner Arc when nothing changed.
        let prune_one = |m: &FileMetadataRef| -> FileMetadataRef {
            m.pruned(&projection, &predicate_cols)
                .ok()
                .filter(|p| p.schema_descr.columns().len() < m.schema_descr.columns().len())
                .map(Arc::new)
                .unwrap_or_else(|| m.clone())
        };

        *first_metadata = first_metadata.as_ref().map(prune_one);
        *metadata_per_source = metadata_per_source
            .as_ref()
            .map(|s| s.iter().map(prune_one).collect());
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
