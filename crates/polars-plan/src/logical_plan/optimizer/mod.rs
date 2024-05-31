use polars_core::prelude::*;

use crate::prelude::*;

mod cache_states;
mod delay_rechunk;

mod cluster_with_columns;
mod collapse_and_project;
mod collect_members;
mod count_star;
#[cfg(feature = "cse")]
mod cse;
mod flatten_union;
#[cfg(feature = "fused")]
mod fused;
mod predicate_pushdown;
mod projection_pushdown;
mod simplify_expr;
mod simplify_functions;
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
pub use crate::frame::{AllowedOptimizations, OptState};
pub use crate::logical_plan::conversion::type_coercion::TypeCoercionRule;
use crate::logical_plan::optimizer::count_star::CountStar;
#[cfg(feature = "cse")]
use crate::logical_plan::optimizer::cse::prune_unused_caches;
#[cfg(feature = "cse")]
use crate::logical_plan::optimizer::cse::CommonSubExprOptimizer;
use crate::logical_plan::optimizer::predicate_pushdown::HiveEval;
#[cfg(feature = "cse")]
use crate::logical_plan::visitor::*;
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
    opt_state: OptState,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut Vec<Node>,
    hive_partition_eval: HiveEval<'_>,
) -> PolarsResult<Node> {
    #[allow(dead_code)]
    let verbose = verbose();
    // get toggle values
    let cluster_with_columns = opt_state.cluster_with_columns;
    let predicate_pushdown = opt_state.predicate_pushdown;
    let projection_pushdown = opt_state.projection_pushdown;
    let type_coercion = opt_state.type_coercion;
    let simplify_expr = opt_state.simplify_expr;
    let slice_pushdown = opt_state.slice_pushdown;
    let streaming = opt_state.streaming;
    let fast_projection = opt_state.fast_projection;
    // Don't run optimizations that don't make sense on a single node.
    // This keeps eager execution more snappy.
    let eager = opt_state.eager;
    #[cfg(feature = "cse")]
    let comm_subplan_elim = opt_state.comm_subplan_elim && !eager;

    #[cfg(feature = "cse")]
    let comm_subexpr_elim = opt_state.comm_subexpr_elim;
    #[cfg(not(feature = "cse"))]
    let comm_subexpr_elim = false;

    #[allow(unused_variables)]
    let agg_scan_projection = opt_state.file_caching && !streaming && !eager;

    // Gradually fill the rules passed to the optimizer
    let opt = StackOptimizer {};
    let mut rules: Vec<Box<dyn OptimizationRule>> = Vec::with_capacity(8);

    let mut lp_top = to_alp(
        logical_plan,
        expr_arena,
        lp_arena,
        simplify_expr,
        type_coercion,
    )?;
    // During debug we check if the optimizations have not modified the final schema.
    #[cfg(debug_assertions)]
    let prev_schema = lp_arena.get(lp_top).schema(lp_arena).into_owned();

    // Collect members for optimizations that need it.
    let mut members = MemberCollector::new();
    if !eager && (comm_subexpr_elim || projection_pushdown) {
        members.collect(lp_top, lp_arena, expr_arena)
    }

    if simplify_expr {
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
    if projection_pushdown {
        let mut projection_pushdown_opt = ProjectionPushDown::new();
        let alp = lp_arena.take(lp_top);
        let alp = projection_pushdown_opt.optimize(alp, lp_arena, expr_arena)?;
        lp_arena.replace(lp_top, alp);

        if projection_pushdown_opt.is_count_star {
            let mut count_star_opt = CountStar::new();
            count_star_opt.optimize_plan(lp_arena, expr_arena, lp_top);
        }
    }

    if predicate_pushdown {
        let predicate_pushdown_opt = PredicatePushDown::new(hive_partition_eval);
        let alp = lp_arena.take(lp_top);
        let alp = predicate_pushdown_opt.optimize(alp, lp_arena, expr_arena)?;
        lp_arena.replace(lp_top, alp);
    }

    if cluster_with_columns {
        cluster_with_columns::optimize(lp_top, lp_arena, expr_arena)
    }

    // Make sure its before slice pushdown.
    if fast_projection {
        rules.push(Box::new(SimpleProjectionAndCollapse::new(eager)));
    }

    if !eager {
        rules.push(Box::new(DelayRechunk::new()));
    }

    if slice_pushdown {
        let slice_pushdown_opt = SlicePushDown::new(streaming);
        let alp = lp_arena.take(lp_top);
        let alp = slice_pushdown_opt.optimize(alp, lp_arena, expr_arena)?;

        lp_arena.replace(lp_top, alp);

        // Expressions use the stack optimizer.
        rules.push(Box::new(slice_pushdown_opt));
    }
    // This optimization removes branches, so we must do it when type coercion
    // is completed.
    if simplify_expr {
        rules.push(Box::new(SimplifyBooleanRule {}));
    }

    if !eager {
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
            hive_partition_eval,
            verbose,
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
