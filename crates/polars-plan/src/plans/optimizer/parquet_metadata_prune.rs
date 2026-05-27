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
#[cfg(feature = "parquet")]
use polars_utils::aliases::PlHashSet;
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

        // O(1) membership lookup; avoids O(|predicate_leaves| × |projection|)
        // when projection is wide (10k+ cols).
        let projection_set: PlHashSet<&str> =
            PlHashSet::from_iter(projection.iter().map(|s| s.as_str()));

        // Filter predicate cols to those in the projection. `pruned()` would
        // otherwise keep a predicate-only column the projection dropped (its
        // `keep` map unions both name lists).
        let predicate_cols: Vec<PlSmallStr> = predicate
            .as_ref()
            .map(|pred_ir| {
                aexpr_to_leaf_names_iter(pred_ir.node(), expr_arena)
                    .filter(|name| projection_set.contains(name.as_str()))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        // Pruning failure (chunks-vs-leaves desync inside `from_compact`) is
        // recoverable; fall back to unpruned metadata so the query still runs.
        // The wire form just stays larger for this entry. Swap-only-if-smaller
        // (leaf count) avoids allocating a new inner Arc when nothing changed.
        let prune_one = |meta_arc: &mut FileMetadataRef| {
            if let Ok(pruned) = meta_arc.pruned(&projection, &predicate_cols)
                && pruned.schema_descr.columns().len() < meta_arc.schema_descr.columns().len()
            {
                *meta_arc = Arc::new(pruned);
            }
        };

        if let Some(meta) = first_metadata {
            prune_one(meta);
        }

        // `Arc<[T]>` has no `make_mut` (slices aren't `Clone`); rebuild instead.
        if let Some(slice) = metadata_per_source {
            *slice = slice
                .iter()
                .cloned()
                .map(|mut m| {
                    prune_one(&mut m);
                    m
                })
                .collect();
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
