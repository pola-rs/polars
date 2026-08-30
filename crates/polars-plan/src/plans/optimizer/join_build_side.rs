//! Choose the build side of a join from plan-time statistics.

use std::sync::Arc;

use polars_core::prelude::{DataType, Schema};
use polars_ops::prelude::JoinBuildSide;
use polars_utils::arena::{Arena, Node};

use crate::plans::{AExpr, IR, NodeStats, node_stats};
use crate::prelude::MaintainOrderJoin;

/// A side has to be this many times smaller to be picked as the build side.
const LOPSIDED_FACTOR: f64 = 8.0;

/// Bytes assumed for a value whose width neither the statistics nor the dtype give.
const DEFAULT_VALUE_WIDTH: f64 = 16.0;

/// Set the build side of every join whose inputs are confidently lopsided.
pub(super) fn set_join_build_sides(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) {
    let sample_limit = polars_config::config().join_sample_limit() as f64;
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
        let Some(side) = build_side(left, right, sample_limit, ir_arena, expr_arena) else {
            continue;
        };
        let IR::Join { options, .. } = ir_arena.get_mut(node) else {
            unreachable!()
        };
        Arc::make_mut(options).args.build_side = Some(side);
    }
}

/// The side to build the hash table from, or `None` if the statistics do not
/// settle it.
fn build_side(
    left: Node,
    right: Node,
    sample_limit: f64,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Option<JoinBuildSide> {
    let left = side_size(left, ir_arena, expr_arena)?;
    let right = side_size(right, ir_arena, expr_arena)?;
    if left.rows <= sample_limit && left.bytes * LOPSIDED_FACTOR <= right.bytes {
        Some(JoinBuildSide::ForceLeft)
    } else if right.rows <= sample_limit && right.bytes * LOPSIDED_FACTOR <= left.bytes {
        Some(JoinBuildSide::ForceRight)
    } else {
        None
    }
}

/// An upper bound on what one input of the join holds.
struct SideSize {
    rows: f64,
    bytes: f64,
}

fn side_size(node: Node, ir_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Option<SideSize> {
    let stats = node_stats(node, ir_arena, expr_arena)?;
    let rows = stats.max_rows()?;
    let schema = ir_arena.get(node).schema(ir_arena);
    Some(SideSize {
        rows,
        bytes: rows * row_width(&schema, &stats),
    })
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
                .map_or_else(|| dtype_width(dtype), f64::from)
        })
        .sum()
}

/// Bytes one value of `dtype` takes, for the dtypes whose width does not depend on
/// the value.
fn dtype_width(dtype: &DataType) -> f64 {
    use DataType::*;
    match dtype {
        Boolean | Int8 | UInt8 => 1.0,
        Int16 | UInt16 | Float16 => 2.0,
        Int32 | UInt32 | Float32 | Date => 4.0,
        Int64 | UInt64 | Float64 | Datetime(..) | Duration(_) | Time => 8.0,
        Int128 | UInt128 => 16.0,
        #[cfg(feature = "dtype-categorical")]
        Enum(..) | Categorical(..) => 4.0,
        _ => DEFAULT_VALUE_WIDTH,
    }
}
