//! Building the reordered join chain.

use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_utils::arena::{Arena, Node};

use super::cluster::Cluster;
use crate::plans::schema::det_join_schema;
use crate::plans::{
    AExpr, ExprIR, IR, JoinOptionsIR, JoinTypeOptionsIR, ProjectionOptions, SchemaRef,
};
use crate::utils::check_input_node;

/// Emit a left-deep join chain over `order`, projected back to the cluster's
/// original schema.
///
/// A join's output columns are its left input's followed by its right input's, so
/// permuting the leaves permutes the output columns. The projection restores the
/// original order.
pub(super) fn rebuild(
    cluster: &Cluster,
    order: &[usize],
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<Node> {
    let mut acc_node = cluster.leaves[order[0]].node;
    let mut acc_schema = cluster.leaves[order[0]].schema.clone();
    let mut is_placed = vec![false; cluster.leaves.len()];
    is_placed[order[0]] = true;

    let mut pending = cluster.residuals.clone();
    acc_node = apply_ready_residuals(&mut pending, &acc_schema, acc_node, ir_arena, expr_arena);

    for &next in &order[1..] {
        let leaf = &cluster.leaves[next];
        let on = keys_joining(cluster, &is_placed, next);

        // Field-by-field rather than struct-update syntax, which would clone the
        // old key vector only to overwrite it.
        let options = Arc::new(JoinOptionsIR {
            options: JoinTypeOptionsIR::Equi {
                on,
                fused_predicate: None,
            },
            args: cluster.options.args.clone(),
            allow_parallel: cluster.options.allow_parallel,
            force_parallel: cluster.options.force_parallel,
            runtime_filters: Vec::new(),
            pass_through_above: None,
        });

        let schema = det_join_schema(&acc_schema, &leaf.schema, &options)?;

        acc_node = ir_arena.add(IR::Join {
            input_left: acc_node,
            input_right: leaf.node,
            schema: schema.clone(),
            options,
        });
        acc_schema = schema;
        is_placed[next] = true;

        acc_node = apply_ready_residuals(&mut pending, &acc_schema, acc_node, ir_arena, expr_arena);
    }

    // A coalescing join folds its key columns away, so a residual reading one is
    // never ready. It still has to be applied.
    for predicate in pending {
        acc_node = ir_arena.add(IR::Filter {
            input: acc_node,
            predicate,
        });
    }

    if !cluster.restore.is_empty() {
        // The leaves were renamed apart, so the original names are restored by alias
        // rather than selected by name.
        acc_node = ir_arena.add(IR::Select {
            input: acc_node,
            expr: cluster.restore.clone(),
            schema: cluster.output_schema.clone(),
            options: ProjectionOptions {
                run_parallel: false,
                duplicate_check: false,
                should_broadcast: false,
                maintain_dataframe_height: false,
            },
        });
    } else if acc_schema != cluster.output_schema {
        acc_node = ir_arena.add(IR::SimpleProjection {
            input: acc_node,
            columns: cluster.output_schema.clone(),
        });
    }

    Ok(acc_node)
}

/// Emit a spanning tree for each equality class reached by the new leaf.
fn keys_joining(cluster: &Cluster, is_placed: &[bool], candidate: usize) -> Vec<(ExprIR, ExprIR)> {
    let mut on = Vec::new();
    for &class in &cluster.classes_by_leaf[candidate] {
        let keys = &cluster.key_classes[class];
        let mut placed = keys.iter().filter(|key| is_placed[key.leaf]);
        let Some(first) = placed.next() else {
            continue;
        };
        let mut candidates = keys.iter().filter(|key| key.leaf == candidate);
        let first_candidate = candidates.next().unwrap();
        on.push((first.key.clone(), first_candidate.key.clone()));
        for key in candidates {
            on.push((first.key.clone(), key.key.clone()));
        }
        // Once the class spans two joined leaves, all its placed columns have
        // already been equated. Before then, one leaf may hold several columns
        // whose equality still needs to be enforced by this join.
        if placed.clone().all(|key| key.leaf == first.leaf) {
            for key in placed {
                on.push((key.key.clone(), first_candidate.key.clone()));
            }
        }
    }
    for bridge in cluster.direct_bridging(is_placed, candidate) {
        let pair = (bridge.placed_key.clone(), bridge.candidate_key.clone());
        if !on.contains(&pair) {
            on.push(pair);
        }
    }
    on
}

/// Apply every pending residual whose columns the chain now carries, innermost first.
fn apply_ready_residuals(
    pending: &mut Vec<ExprIR>,
    schema: &SchemaRef,
    mut acc_node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Node {
    let mut i = 0;
    while i < pending.len() {
        if check_input_node(pending[i].node(), schema, expr_arena) {
            let predicate = pending.remove(i);
            acc_node = ir_arena.add(IR::Filter {
                input: acc_node,
                predicate,
            });
        } else {
            i += 1;
        }
    }
    acc_node
}
