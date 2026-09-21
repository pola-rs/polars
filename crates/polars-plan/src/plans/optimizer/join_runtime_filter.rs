//! Let a hash join tell the parquet scan under its probe side which keys the build
//! side holds.
//!
//! A join whose one side is known to be small gets that side forced as build side;
//! one whose side is only estimated small gets it preferred. Every probe key that
//! reads a parquet scan column unchanged gets a dynamic predicate on that scan. The
//! join publishes the range of its build keys once the build is done, and the scan
//! skips row groups outside it. When the build keys are estimated to be a small
//! share of the scan's distinct keys the join also publishes a bloom filter over
//! them, and the predicate is evaluated per row; otherwise it only skips batches.
//!
//! A forced join blocks its probe input until the build is done, so its range is
//! always published before the scan opens. A preferred join with a filter reads its
//! preferred side first; when that side ends under the sample limit its range is
//! published before the other side is read, and the sample then decides which side
//! to build. When it reaches the limit both sides are sampled and the filter is
//! only set if the preferred side is built. Either way the result is exact, only
//! the pruning is lost. A predicate is only carried across joins that read their
//! sides in this order.
//!
//! Only a scan that skips batches by their statistics can use the range, so a plan
//! without one is left alone, and a join is only given a build side once a key
//! reached one.
//!
//! A semi join publishes from either side and an anti join from its left side
//! only, as the rows of its right side that match no left key change nothing. The
//! semi/anti join node reads its sides in order only when one is forced, so those
//! joins get forced build sides only.

use std::sync::Arc;

use polars_core::prelude::PlIndexMap;
use polars_defs::join::JoinBuildSide;
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::pl_str::PlSmallStr;

use super::join_build_side::{LOPSIDED_FACTOR, side_stats};
use super::predicate_pushdown::utils::{
    PushdownEligibility, map_column_references, pushdown_eligibility, temporary_unique_key,
};
use crate::dsl::{FileScanIR, ScanFlags};
use crate::plans::aexpr::predicates::supports_runtime_range;
use crate::plans::optimizer::predicate_pushdown::{DynamicPred, new_batch_only_dynamic_pred};
use crate::plans::options::{MAX_BUILD_PROBE_DISTINCT_RATIO, RuntimeFilter};
use crate::plans::schema::join_right_output_names;
use crate::plans::stats::StatsCache;
use crate::plans::{
    AExpr, ExprIR, IR, IRFunctionExpr, JoinOptionsIR, JoinTypeOptionsIR, NodeStats, Operator,
    into_column, is_inherently_nondeterministic,
};
use crate::prelude::{JoinType, MaintainOrderJoin};
use crate::utils::has_aexpr;

/// Estimated bytes a build side chosen here may take.
const BUILD_BYTES: f64 = 256.0 * 1024.0 * 1024.0;

pub(super) fn attach_join_runtime_filters(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) {
    if !polars_config::config().join_runtime_filters() {
        return;
    }
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
    if !is_eligible_join(options, expr_arena)
        || matches!(
            options.args.build_side,
            Some(JoinBuildSide::ForceLeft | JoinBuildSide::ForceRight)
        )
    {
        return;
    }
    let JoinTypeOptionsIR::Equi { on, .. } = &options.options else {
        return;
    };
    let on = on.clone();
    let Some((left_stats, left_width)) = side_stats(input_left, ir_arena, expr_arena, stats) else {
        return;
    };
    let Some((right_stats, right_width)) = side_stats(input_right, ir_arena, expr_arena, stats)
    else {
        return;
    };

    // A bounded side before an estimated one, the smaller of two alike; the first
    // whose range prunes a scan much larger than itself is taken.
    // Try the right side first when candidates rank equally.
    let mut sides = build_candidates(false, &right_stats, right_width);
    sides.extend(build_candidates(true, &left_stats, left_width));
    let how = &options.args.how;
    if how.is_semi_anti() {
        sides.retain(|s| s.forced && (how.is_semi() || s.left));
    }
    sides.sort_by(|a, b| b.forced.cmp(&a.forced).then(a.rows.total_cmp(&b.rows)));

    // A side is traced once, as a forced and a preferred candidate share the trace.
    let mut traced: [Option<(Vec<TracedKey>, f64)>; 2] = [None, None];
    let chosen = sides.into_iter().find_map(|side| {
        let (filters, probe_rows) = traced[side.left as usize].get_or_insert_with(|| {
            let probe_input = if side.left { input_right } else { input_left };
            let probe_keys: Vec<Option<PlSmallStr>> = on
                .iter()
                .map(|(left_key, right_key)| {
                    let key = if side.left { right_key } else { left_key };
                    into_column(key.node(), expr_arena).cloned()
                })
                .collect();
            let filters = trace_probe_keys(probe_input, probe_keys, ir_arena, expr_arena, scratch);
            let other = if side.left { &right_stats } else { &left_stats };
            let probe_rows = filters
                .iter()
                .filter_map(|f| side_stats(f.scan, ir_arena, expr_arena, stats))
                .fold(other.filtered, |acc, (scan, _)| acc.max(scan.filtered));
            (filters, probe_rows)
        });
        (!filters.is_empty() && *probe_rows >= LOPSIDED_FACTOR * side.rows).then_some(side)
    });
    let Some(BuildCandidate { left, forced, rows }) = chosen else {
        return;
    };
    let (filters, _) = traced[left as usize].take().unwrap();
    let build_stats = if left { &left_stats } else { &right_stats };

    let mut runtime_filters = Vec::with_capacity(filters.len());
    for filter in filters {
        // The key's distinct count is that of the unfiltered side.
        let build_key = &on[filter.key_idx];
        let build_key = if left { &build_key.0 } else { &build_key.1 };
        let kept = (build_stats.filtered / build_stats.unfiltered).min(1.0);
        let build_distinct = into_column(build_key.node(), expr_arena)
            .and_then(|name| build_stats.key_distinct_estimate(name))
            .map_or(rows, |ndv| (ndv * kept).min(rows));
        let scan_key = column_name(&filter.predicate, expr_arena).clone();
        let probe_distinct = side_stats(filter.scan, ir_arena, expr_arena, stats)
            .and_then(|(scan_stats, _)| scan_stats.key_distinct_estimate(&scan_key));
        let bloom = !probe_distinct
            .is_some_and(|probe| build_distinct > probe * MAX_BUILD_PROBE_DISTINCT_RATIO);
        if polars_config::config().verbose() {
            eprintln!(
                "runtime filter on {scan_key}: {build_distinct:.0} distinct build keys of {rows:.0} rows, {probe_distinct:?} distinct probe keys, bloom: {bloom}"
            );
        }
        if bloom {
            evaluate_per_row(filter.predicate.node(), expr_arena);
        }
        attach_to_scan(filter.scan, filter.predicate, ir_arena, expr_arena);
        runtime_filters.push(RuntimeFilter {
            key_idx: filter.key_idx,
            pred: filter.pred,
            bloom_keys: bloom.then_some(build_distinct.ceil() as usize),
            probe_distinct: probe_distinct.map(|d| d.ceil() as usize),
        });
    }
    let IR::Join { options, .. } = ir_arena.get_mut(node) else {
        unreachable!()
    };
    let options = Arc::make_mut(options);
    options.args.build_side = Some(match (left, forced) {
        (true, true) => JoinBuildSide::ForceLeft,
        (false, true) => JoinBuildSide::ForceRight,
        (true, false) => JoinBuildSide::PreferLeft,
        (false, false) => JoinBuildSide::PreferRight,
    });
    options.runtime_filters = runtime_filters;
}

/// A side of the join that could be built from.
struct BuildCandidate {
    left: bool,
    /// Its bound when forced, else its estimate.
    rows: f64,
    forced: bool,
}

/// The ways a filtered side may be built from: forced when its bound fits the
/// byte budget, preferred when its estimate fits.
fn build_candidates(left: bool, stats: &NodeStats, width: f64) -> Vec<BuildCandidate> {
    let mut candidates = Vec::new();
    if stats.filtered >= stats.unfiltered {
        return candidates;
    }
    let fits = |rows: f64| rows * width <= BUILD_BYTES;
    if let Some(rows) = stats.max_rows().filter(|b| fits(*b)) {
        candidates.push(BuildCandidate {
            left,
            rows,
            forced: true,
        });
    }
    if fits(stats.filtered) {
        candidates.push(BuildCandidate {
            left,
            rows: stats.filtered,
            forced: false,
        });
    }
    candidates
}

/// A probe key that reached a scan.
struct TracedKey {
    key_idx: usize,
    scan: Node,
    predicate: ExprIR,
    pred: DynamicPred,
}

/// The probe keys whose column reaches a scan that can skip batches, each with the
/// batch-only predicate to put on that scan.
fn trace_probe_keys(
    probe_input: Node,
    probe_keys: Vec<Option<PlSmallStr>>,
    ir_arena: &Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut UnitVec<Node>,
) -> Vec<TracedKey> {
    let probe_schema = ir_arena.get(probe_input).schema(ir_arena).into_owned();
    let mut traced = Vec::new();
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
            traced.push(TracedKey {
                key_idx,
                scan,
                predicate,
                pred,
            });
        }
    }
    traced
}

/// Whether the join may publish a filter or be crossed by one: an inner, semi or
/// anti equi join the streaming engine can run as a hash join that blocks its
/// probe side until the build is done. Sorted inputs may still make it a merge
/// join, which drops the filter. A key that may evaluate differently each time
/// gives no filter, as the join evaluates it again when it builds.
fn is_eligible_join(options: &JoinOptionsIR, expr_arena: &Arena<AExpr>) -> bool {
    let args = &options.args;
    let JoinTypeOptionsIR::Equi { on, .. } = &options.options else {
        return false;
    };
    !on.is_empty()
        && on.iter().all(|(left, right)| {
            !is_inherently_nondeterministic(left.node(), expr_arena)
                && !is_inherently_nondeterministic(right.node(), expr_arena)
        })
        && (args.how == JoinType::Inner || args.how.is_semi_anti())
        && args.maintain_order == MaintainOrderJoin::None
        && !args.nulls_equal
        && args.slice.is_none()
        && !args.validation.needs_checks()
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
                // A preferred side is only read first when the join carries runtime
                // filters; an ordinary preference samples both sides at once.
                let sequential = !options.runtime_filters.is_empty();
                let probe_left = match options.args.build_side {
                    Some(JoinBuildSide::ForceRight) => true,
                    Some(JoinBuildSide::ForceLeft) => false,
                    Some(JoinBuildSide::PreferRight) if sequential => true,
                    Some(JoinBuildSide::PreferLeft) if sequential => false,
                    _ => return None,
                };
                if !is_eligible_join(options, expr_arena) {
                    return None;
                }
                let name = column_name(predicate, expr_arena).clone();
                let schema_left = ir_arena.get(*input_left).schema(ir_arena);
                let schema_right = ir_arena.get(*input_right).schema(ir_arena);
                // A semi or anti join outputs the left columns only.
                let from_right = if options.args.how.is_semi_anti() {
                    None
                } else {
                    join_right_output_names(&schema_left, &schema_right, options)
                        .ok()?
                        .iter()
                        .position(|output| output.as_ref() == Some(&name))
                };
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

/// Let the scan evaluate the dynamic predicate per row, not only by statistics.
fn evaluate_per_row(node: Node, expr_arena: &mut Arena<AExpr>) {
    let AExpr::Function {
        function: IRFunctionExpr::DynamicPred { batch_only, .. },
        ..
    } = expr_arena.get_mut(node)
    else {
        unreachable!()
    };
    *batch_only = false;
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
