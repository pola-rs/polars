use polars_core::prelude::*;

use crate::prelude::*;

mod cache_states;
mod delay_rechunk;

mod cluster_with_columns;
mod collapse_and_project;
mod collapse_joins;
mod collect_members;
mod count_star;
#[cfg(feature = "cse")]
mod cse;
mod flatten_union;
#[cfg(feature = "fused")]
mod fused;
mod join_utils;
mod predicate_pushdown;
mod projection_pushdown;
mod set_order;
mod simplify_expr;
mod slice_pushdown_expr;
mod slice_pushdown_lp;
mod stack_opt;

use collapse_and_project::SimpleProjectionAndCollapse;
use delay_rechunk::DelayRechunk;
use polars_core::config::verbose;
use polars_io::predicates::PhysicalIoExpr;
pub use predicate_pushdown::PredicatePushDown;
pub use projection_pushdown::ProjectionPushDown;
pub use simplify_expr::{SimplifyBooleanRule, SimplifyExprRule};
use slice_pushdown_lp::SlicePushDown;
pub use stack_opt::{OptimizationRule, StackOptimizer};

use self::flatten_union::FlattenUnionRule;
use self::set_order::set_order_flags;
pub use crate::frame::{AllowedOptimizations, OptFlags};
pub use crate::plans::conversion::type_coercion::TypeCoercionRule;
use crate::plans::optimizer::count_star::CountStar;
#[cfg(feature = "cse")]
use crate::plans::optimizer::cse::prune_unused_caches;
#[cfg(feature = "cse")]
use crate::plans::optimizer::cse::CommonSubExprOptimizer;
use crate::plans::optimizer::predicate_pushdown::ExprEval;
#[cfg(feature = "cse")]
use crate::plans::visitor::*;
use crate::prelude::optimizer::collect_members::MemberCollector;

pub trait Optimize {
    fn optimize(&self, logical_plan: DslPlan) -> PolarsResult<DslPlan>;
}

// arbitrary constant to reduce reallocation.
const HASHMAP_SIZE: usize = 16;

pub(crate) fn init_hashmap<K, V>(max_len: Option<usize>) -> PlHashMap<K, V> {
    PlHashMap::with_capacity(std::cmp::min(max_len.unwrap_or(HASHMAP_SIZE), HASHMAP_SIZE))
}

pub fn optimize(
    logical_plan: DslPlan,
    mut opt_state: OptFlags,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut Vec<Node>,
    expr_eval: ExprEval<'_>,
) -> PolarsResult<Node> {
    #[allow(dead_code)]
    let verbose = verbose();

    if opt_state.streaming() {
        polars_warn!(
            Deprecation,
            "\
The old streaming engine is being deprecated and will soon be replaced by the new streaming \
engine. Starting Polars version 1.23.0 and until the new streaming engine is released, the old \
streaming engine may become less usable. For people who rely on the old streaming engine, it is \
suggested to pin your version to before 1.23.0.

More information on the new streaming engine: https://github.com/pola-rs/polars/issues/20947"
        )
    }

    // Gradually fill the rules passed to the optimizer
    let opt = StackOptimizer {};
    let mut rules: Vec<Box<dyn OptimizationRule>> = Vec::with_capacity(8);

    // Unset CSE
    // This can be turned on again during ir-conversion.
    #[allow(clippy::eq_op)]
    #[cfg(feature = "cse")]
    if opt_state.contains(OptFlags::EAGER) {
        opt_state &= !(OptFlags::COMM_SUBEXPR_ELIM | OptFlags::COMM_SUBEXPR_ELIM);
    }
    let mut lp_top = to_alp(logical_plan, expr_arena, lp_arena, &mut opt_state)?;

    // Don't run optimizations that don't make sense on a single node.
    // This keeps eager execution more snappy.
    #[cfg(feature = "cse")]
    let comm_subplan_elim = opt_state.contains(OptFlags::COMM_SUBPLAN_ELIM);

    #[cfg(feature = "cse")]
    let comm_subexpr_elim = opt_state.contains(OptFlags::COMM_SUBEXPR_ELIM);
    #[cfg(not(feature = "cse"))]
    let comm_subexpr_elim = false;

    #[allow(unused_variables)]
    let agg_scan_projection =
        opt_state.contains(OptFlags::FILE_CACHING) && !opt_state.streaming() && !opt_state.eager();

    // During debug we check if the optimizations have not modified the final schema.
    #[cfg(debug_assertions)]
    let prev_schema = lp_arena.get(lp_top).schema(lp_arena).into_owned();

    // Collect members for optimizations that need it.
    let mut members = MemberCollector::new();
    if !opt_state.eager() && (comm_subexpr_elim || opt_state.projection_pushdown()) {
        members.collect(lp_top, lp_arena, expr_arena)
    }

    // Run before slice pushdown
    if opt_state.contains(OptFlags::CHECK_ORDER_OBSERVE)
        && members.has_group_by | members.has_sort | members.has_distinct
    {
        set_order_flags(lp_top, lp_arena, expr_arena, scratch);
    }

    if opt_state.simplify_expr() {
        #[cfg(feature = "fused")]
        rules.push(Box::new(fused::FusedArithmetic {}));
    }

    #[cfg(feature = "cse")]
    let _cse_plan_changed = if comm_subplan_elim
        && members.has_joins_or_unions
        && members.has_duplicate_scans()
        && !members.has_cache
    {
        if verbose {
            eprintln!("found multiple sources; run comm_subplan_elim")
        }
        let (lp, changed, cid2c) = cse::elim_cmn_subplans(lp_top, lp_arena, expr_arena);

        prune_unused_caches(lp_arena, cid2c);

        lp_top = lp;
        members.has_cache |= changed;
        changed
    } else {
        false
    };
    #[cfg(not(feature = "cse"))]
    let _cse_plan_changed = false;

    // Should be run before predicate pushdown.
    if opt_state.projection_pushdown() {
        let mut projection_pushdown_opt = ProjectionPushDown::new(opt_state.new_streaming());
        let alp = lp_arena.take(lp_top);
        let alp = projection_pushdown_opt.optimize(alp, lp_arena, expr_arena)?;
        lp_arena.replace(lp_top, alp);

        if projection_pushdown_opt.is_count_star {
            let mut count_star_opt = CountStar::new();
            count_star_opt.optimize_plan(lp_arena, expr_arena, lp_top)?;
        }
    }

    if opt_state.predicate_pushdown() {
        let mut predicate_pushdown_opt = PredicatePushDown::new(expr_eval);
        let alp = lp_arena.take(lp_top);
        let alp = predicate_pushdown_opt.optimize(alp, lp_arena, expr_arena)?;
        lp_arena.replace(lp_top, alp);
    }

    if opt_state.cluster_with_columns() {
        cluster_with_columns::optimize(lp_top, lp_arena, expr_arena)
    }

    // Make sure it is after predicate pushdown
    if opt_state.collapse_joins() && members.has_filter_with_join_input {
        collapse_joins::optimize(lp_top, lp_arena, expr_arena)
    }

    // Make sure its before slice pushdown.
    if opt_state.fast_projection() {
        rules.push(Box::new(SimpleProjectionAndCollapse::new(
            opt_state.eager(),
        )));
    }

    if !opt_state.eager() {
        rules.push(Box::new(DelayRechunk::new()));
    }

    if opt_state.slice_pushdown() {
        let mut slice_pushdown_opt =
            SlicePushDown::new(opt_state.streaming(), opt_state.new_streaming());
        let alp = lp_arena.take(lp_top);
        let alp = slice_pushdown_opt.optimize(alp, lp_arena, expr_arena)?;

        lp_arena.replace(lp_top, alp);

        // Expressions use the stack optimizer.
        rules.push(Box::new(slice_pushdown_opt));
    }
    // This optimization removes branches, so we must do it when type coercion
    // is completed.
    if opt_state.simplify_expr() {
        rules.push(Box::new(SimplifyBooleanRule {}));
    }

    if !opt_state.eager() {
        rules.push(Box::new(FlattenUnionRule {}));
    }

    lp_top = opt.optimize_loop(&mut rules, expr_arena, lp_arena, lp_top)?;

    if members.has_joins_or_unions && members.has_cache && _cse_plan_changed {
        // We only want to run this on cse inserted caches
        cache_states::set_cache_states(
            lp_top,
            lp_arena,
            expr_arena,
            scratch,
            expr_eval,
            verbose,
            opt_state.new_streaming(),
        )?;
    }

    // This one should run (nearly) last as this modifies the projections
    #[cfg(feature = "cse")]
    if comm_subexpr_elim && !members.has_ext_context {
        let mut optimizer = CommonSubExprOptimizer::new();
        let alp_node = IRNode::new(lp_top);

        lp_top = try_with_ir_arena(lp_arena, expr_arena, |arena| {
            let rewritten = alp_node.rewrite(&mut optimizer, arena)?;
            Ok(rewritten.node())
        })?;
    }

    // During debug we check if the optimizations have not modified the final schema.
    #[cfg(debug_assertions)]
    {
        // only check by names because we may supercast types.
        assert_eq!(
            prev_schema.iter_names().collect::<Vec<_>>(),
            lp_arena
                .get(lp_top)
                .schema(lp_arena)
                .iter_names()
                .collect::<Vec<_>>()
        );
    };

    Ok(lp_top)
}
