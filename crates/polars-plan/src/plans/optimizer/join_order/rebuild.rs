//! Building the reordered join chain.

use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::schema::Schema;
use polars_utils::arena::{Arena, Node};

use super::cluster::Cluster;
use crate::plans::schema::det_join_schema;
use crate::plans::{AExpr, ExprIR, IR, JoinOptionsIR, JoinTypeOptionsIR};

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

    for &next in &order[1..] {
        let leaf = &cluster.leaves[next];
        let on = keys_joining(cluster, &is_placed, next);

        // Field-by-field rather than struct-update syntax, which would clone the
        // old key vector only to overwrite it.
        let options = Arc::new(JoinOptionsIR {
            options: JoinTypeOptionsIR::Equi { on },
            args: cluster.options.args.clone(),
            allow_parallel: cluster.options.allow_parallel,
            force_parallel: cluster.options.force_parallel,
        });

        let schema = det_join_schema(&acc_schema, &leaf.schema, &options, expr_arena)?;

        acc_node = ir_arena.add(IR::Join {
            input_left: acc_node,
            input_right: leaf.node,
            schema: schema.clone(),
            options,
        });
        acc_schema = schema;
        is_placed[next] = true;
    }

    if !same_column_order(&acc_schema, &cluster.output_schema) {
        acc_node = ir_arena.add(IR::SimpleProjection {
            input: acc_node,
            columns: cluster.output_schema.clone(),
        });
    }

    Ok(acc_node)
}

/// The key pairs bridging the already-joined leaves and `candidate`, oriented so the
/// accumulated side comes first.
fn keys_joining(cluster: &Cluster, is_placed: &[bool], candidate: usize) -> Vec<(ExprIR, ExprIR)> {
    cluster
        .bridging(is_placed, candidate)
        .map(|bridge| (bridge.placed_key.clone(), bridge.candidate_key.clone()))
        .collect()
}

fn same_column_order(a: &Schema, b: &Schema) -> bool {
    a.len() == b.len() && a.iter_names().zip(b.iter_names()).all(|(x, y)| x == y)
}
