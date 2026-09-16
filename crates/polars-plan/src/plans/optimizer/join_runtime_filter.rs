//! Let a hash join tell the parquet scan under its probe side which keys the build
//! side holds.
//!
//! A join whose one side is known to be small gets that side forced as build side,
//! and every probe key that reads a parquet scan column unchanged gets a batch-only
//! dynamic predicate on that scan. The join publishes the range of its build keys
//! once the build is done; the scan then skips row groups outside it and never
//! filters rows by it. The probe side is only read after the build, because a
//! forced join blocks its probe input until then and a predicate is only carried
//! across joins that are forced the same way.
//!
//! Only a scan that skips batches by their statistics can use the range, so a plan
//! without one is left alone, and a join is only forced once a key reached one.

use std::sync::Arc;

use polars_core::prelude::PlIndexMap;
use polars_ops::prelude::JoinBuildSide;
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::pl_str::PlSmallStr;

use super::join_build_side::{LOPSIDED_FACTOR, side_stats};
use super::predicate_pushdown::utils::{
    PushdownEligibility, map_column_references, pushdown_eligibility, temporary_unique_key,
};
use crate::dsl::{FileScanIR, ScanFlags};
use crate::plans::aexpr::predicates::supports_runtime_range;
use crate::plans::optimizer::predicate_pushdown::new_batch_only_dynamic_pred;
use crate::plans::options::RuntimeFilter;
use crate::plans::schema::join_right_output_names;
use crate::plans::stats::StatsCache;
use crate::plans::{AExpr, ExprIR, IR, JoinOptionsIR, JoinTypeOptionsIR, Operator, into_column};
use crate::prelude::{JoinType, MaintainOrderJoin};
use crate::utils::has_aexpr;

/// Estimated bytes a forced build side may take.
const FORCED_BUILD_BYTES: f64 = 256.0 * 1024.0 * 1024.0;

pub(super) fn attach_join_runtime_filters(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) {
    // Inputs before their join, so a join lower in a probe chain is forced before
    // an outer one tries to carry a predicate through it.
    let mut joins = Vec::new();
    let mut has_pruning_scan = false;
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let ir = ir_arena.get(node);
        match ir {
            IR::Join { .. } => joins.push(node),
            IR::Scan { scan_type, .. } => has_pruning_scan |= skips_batches(scan_type),
            _ => {},
        }
        ir.copy_inputs(&mut stack);
    }
    if !has_pruning_scan {
        return;
    }
    let mut scratch = UnitVec::new();
    let mut stats = StatsCache::default();
    for node in joins.into_iter().rev() {
        process_join(node, ir_arena, expr_arena, &mut scratch, &mut stats);
    }
}

fn process_join(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut UnitVec<Node>,
    stats: &mut StatsCache,
) {
    let IR::Join {
        input_left,
        input_right,
        options,
        ..
    } = ir_arena.get(node)
    else {
        return;
    };
    let (input_left, input_right) = (*input_left, *input_right);
    if !is_eligible_join(options)
        || matches!(
            options.args.build_side,
            Some(JoinBuildSide::ForceLeft | JoinBuildSide::ForceRight)
        )
    {
        return;
    }
    let Some(build_left) = choose_build_side(input_left, input_right, ir_arena, expr_arena, stats)
    else {
        return;
    };
    let JoinTypeOptionsIR::Equi { on, .. } = &options.options else {
        return;
    };
    let (probe_input, probe_keys): (Node, Vec<Option<PlSmallStr>>) = if build_left {
        (
            input_right,
            on.iter()
                .map(|(_, key)| into_column(key.node(), expr_arena).cloned())
                .collect(),
        )
    } else {
        (
            input_left,
            on.iter()
                .map(|(key, _)| into_column(key.node(), expr_arena).cloned())
                .collect(),
        )
    };
    let probe_schema = ir_arena.get(probe_input).schema(ir_arena).into_owned();

    let mut filters = Vec::new();
    for (key_idx, name) in probe_keys.into_iter().enumerate() {
        let Some(name) = name else { continue };
        if !probe_schema.get(&name).is_some_and(supports_runtime_range) {
            continue;
        }
        let column = expr_arena.add(AExpr::Column(name));
        let (dyn_node, pred) = new_batch_only_dynamic_pred(column, expr_arena);
        let mut predicate = ExprIR::from_node(dyn_node, expr_arena);
        if let Some(scan) = scan_origin(probe_input, &mut predicate, ir_arena, expr_arena, scratch)
        {
            attach_to_scan(scan, predicate, ir_arena, expr_arena);
            filters.push(RuntimeFilter { key_idx, pred });
        }
    }
    if filters.is_empty() {
        return;
    }

    let IR::Join { options, .. } = ir_arena.get_mut(node) else {
        unreachable!()
    };
    let options = Arc::make_mut(options);
    options.args.build_side = Some(if build_left {
        JoinBuildSide::ForceLeft
    } else {
        JoinBuildSide::ForceRight
    });
    options.runtime_filters = filters;
}

/// Whether the join may publish a range or be crossed by one: an inner equi join
/// the streaming engine can run as a hash join that blocks its probe side until
/// the build is done. Sorted inputs may still make it a merge join, which drops
/// the range.
fn is_eligible_join(options: &JoinOptionsIR) -> bool {
    let args = &options.args;
    matches!(&options.options, JoinTypeOptionsIR::Equi { on, .. } if !on.is_empty())
        && args.how == JoinType::Inner
        && args.maintain_order == MaintainOrderJoin::None
        && !args.nulls_equal
        && args.slice.is_none()
        && !args.validation.needs_checks()
}

/// The side to force as build side: filtered, bounded, within the byte budget, and
/// much smaller than the other side's estimate. `true` for the left side.
fn choose_build_side(
    left: Node,
    right: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    stats: &mut StatsCache,
) -> Option<bool> {
    let (left_stats, left_width) = side_stats(left, ir_arena, expr_arena, stats)?;
    let (right_stats, right_width) = side_stats(right, ir_arena, expr_arena, stats)?;
    let bound = |stats: &crate::plans::NodeStats, width: f64| {
        stats
            .max_rows()
            .filter(|rows| stats.filtered < stats.unfiltered && rows * width <= FORCED_BUILD_BYTES)
    };
    let left_bound = bound(&left_stats, left_width);
    let right_bound = bound(&right_stats, right_width);
    let left_ok = left_bound.is_some_and(|b| right_stats.filtered >= LOPSIDED_FACTOR * b);
    let right_ok = right_bound.is_some_and(|b| left_stats.filtered >= LOPSIDED_FACTOR * b);
    match (left_ok, right_ok) {
        (true, true) => Some(left_bound <= right_bound),
        (true, false) => Some(true),
        (false, true) => Some(false),
        (false, false) => None,
    }
}

/// The scan whose column `predicate` reads, following the column down from `node`
/// through nodes a predicate may be pushed past. `predicate` is rewritten to the
/// scan's column name on the way.
///
/// Besides the checks static predicate pushdown makes, a window function anywhere
/// on the path is a barrier: it is lowered to a group-by that reads the scan on
/// its own, before the forced join above has built.
fn scan_origin(
    mut node: Node,
    predicate: &mut ExprIR,
    ir_arena: &Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut UnitVec<Node>,
) -> Option<Node> {
    let has_window = |exprs: &[ExprIR], expr_arena: &Arena<AExpr>| {
        exprs
            .iter()
            .any(|e| has_aexpr(e.node(), expr_arena, |ae| matches!(ae, AExpr::Over { .. })))
    };
    // Only tells the predicates of one eligibility check apart.
    let key = PlSmallStr::from_static("__POLARS_RUNTIME_FILTER");
    loop {
        match ir_arena.get(node) {
            IR::Scan { scan_type, .. } => return skips_batches(scan_type).then_some(node),
            IR::SimpleProjection { input, columns } => {
                let name = column_name(predicate, expr_arena);
                if !columns.contains(name) {
                    return None;
                }
                node = *input;
            },
            IR::Filter {
                input,
                predicate: filter,
            } => {
                if has_window(std::slice::from_ref(filter), expr_arena) {
                    return None;
                }
                let mut acc = PlIndexMap::default();
                acc.insert(key.clone(), predicate.clone());
                let tmp_key = temporary_unique_key(&acc);
                acc.insert(tmp_key.clone(), filter.clone());
                let (eligibility, _) = pushdown_eligibility(
                    &[],
                    &[(&tmp_key, filter.clone())],
                    &acc,
                    expr_arena,
                    scratch,
                    true,
                    ir_arena.get(*input),
                )
                .ok()?;
                if !matches!(eligibility, PushdownEligibility::Full) {
                    return None;
                }
                node = *input;
            },
            IR::Select { input, expr, .. }
            | IR::HStack {
                input, exprs: expr, ..
            } => {
                if has_window(expr, expr_arena) {
                    return None;
                }
                let mut acc = PlIndexMap::default();
                acc.insert(key.clone(), predicate.clone());
                let (eligibility, renames) = pushdown_eligibility(
                    expr,
                    &[],
                    &acc,
                    expr_arena,
                    scratch,
                    true,
                    ir_arena.get(*input),
                )
                .ok()?;
                if !matches!(eligibility, PushdownEligibility::Full) {
                    return None;
                }
                map_column_references(predicate, expr_arena, &renames);
                node = *input;
            },
            IR::Join {
                input_left,
                input_right,
                options,
                ..
            } => {
                let probe_left = match options.args.build_side {
                    Some(JoinBuildSide::ForceRight) => true,
                    Some(JoinBuildSide::ForceLeft) => false,
                    _ => return None,
                };
                if !is_eligible_join(options) {
                    return None;
                }
                let name = column_name(predicate, expr_arena).clone();
                let schema_left = ir_arena.get(*input_left).schema(ir_arena);
                let schema_right = ir_arena.get(*input_right).schema(ir_arena);
                let right_names =
                    join_right_output_names(&schema_left, &schema_right, options, expr_arena)
                        .ok()?;
                let from_right = right_names
                    .iter()
                    .position(|output| output.as_ref() == Some(&name));
                match from_right {
                    Some(idx) if !probe_left => {
                        let input_name = schema_right.get_at_index(idx)?.0.clone();
                        if input_name != name {
                            let mut renames = PlIndexMap::default();
                            renames.insert(name, input_name);
                            map_column_references(predicate, expr_arena, &renames);
                        }
                        node = *input_right;
                    },
                    None if probe_left && schema_left.contains(&name) => node = *input_left,
                    _ => return None,
                }
            },
            _ => return None,
        }
    }
}

fn skips_batches(scan_type: &FileScanIR) -> bool {
    scan_type
        .flags()
        .contains(ScanFlags::SKIPS_BATCHES_BY_STATISTICS)
}

fn column_name<'a>(predicate: &ExprIR, expr_arena: &'a Arena<AExpr>) -> &'a PlSmallStr {
    let AExpr::Function { input, .. } = expr_arena.get(predicate.node()) else {
        unreachable!()
    };
    into_column(input[0].node(), expr_arena).unwrap()
}

fn attach_to_scan(
    scan: Node,
    predicate: ExprIR,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) {
    let IR::Scan {
        predicate: existing,
        ..
    } = ir_arena.get_mut(scan)
    else {
        unreachable!()
    };
    *existing = Some(match existing.take() {
        None => predicate,
        Some(existing) => {
            let node = expr_arena.add(AExpr::BinaryExpr {
                left: existing.node(),
                op: Operator::And,
                right: predicate.node(),
            });
            ExprIR::from_node(node, expr_arena)
        },
    });
}
