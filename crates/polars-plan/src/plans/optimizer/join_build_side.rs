//! Choose the build side of a join from plan-time statistics.

use std::sync::Arc;

use polars_core::prelude::Schema;
use polars_ops::prelude::JoinBuildSide;
use polars_utils::arena::{Arena, Node};

use crate::plans::{AExpr, IR, NodeStats, node_stats};
use crate::prelude::MaintainOrderJoin;

/// A side has to be this many times smaller to be picked as the build side.
const LOPSIDED_FACTOR: f64 = 8.0;

/// Bytes assumed for a value whose width neither the statistics nor the dtype give.
const DEFAULT_VALUE_WIDTH: f64 = 16.0;

/// Set the preferred build side of every join whose inputs are confidently
/// lopsided.
pub(super) fn set_join_build_sides(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let ir = ir_arena.get(node);
        ir.copy_inputs(&mut stack);
        let IR::Join {
            input_left,
            input_right,
            options,
            ..
        } = ir
        else {
            continue;
        };
        if options.args.build_side.is_some()
            || options.args.maintain_order != MaintainOrderJoin::None
            || !(options.args.how.is_equi() || options.args.how.is_cross())
        {
            continue;
        }
        let (left, right) = (*input_left, *input_right);
        let Some(side) = build_side(left, right, ir_arena, expr_arena) else {
            continue;
        };
        let IR::Join { options, .. } = ir_arena.get_mut(node) else {
            unreachable!()
        };
        Arc::make_mut(options).args.build_side = Some(side);
    }
}

/// The side to prefer building the hash table from, or `None` if the statistics
/// do not settle it.
///
/// A preference only settles the case where the engine saturates its sample on
/// both sides; anything it measures in full outranks it.
fn build_side(
    left: Node,
    right: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Option<JoinBuildSide> {
    let left = side_bytes(left, ir_arena, expr_arena)?;
    let right = side_bytes(right, ir_arena, expr_arena)?;
    if left * LOPSIDED_FACTOR <= right {
        Some(JoinBuildSide::PreferLeft)
    } else if right * LOPSIDED_FACTOR <= left {
        Some(JoinBuildSide::PreferRight)
    } else {
        None
    }
}

/// An upper bound on the bytes one input of the join holds.
fn side_bytes(node: Node, ir_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Option<f64> {
    let stats = node_stats(node, ir_arena, expr_arena)?;
    let rows = stats.max_rows()?;
    let schema = ir_arena.get(node).schema(ir_arena);
    Some(rows * row_width(&schema, &stats))
}

/// Bytes one row of `schema` takes, from the statistics where they describe a
/// column and from its dtype otherwise.
fn row_width(schema: &Schema, stats: &NodeStats) -> f64 {
    schema
        .iter()
        .map(|(name, dtype)| {
            stats
                .column(name)
                .and_then(|c| c.avg_byte_width)
                .map_or_else(
                    || dtype.byte_width().map_or(DEFAULT_VALUE_WIDTH, |w| w as f64),
                    f64::from,
                )
        })
        .sum()
}
