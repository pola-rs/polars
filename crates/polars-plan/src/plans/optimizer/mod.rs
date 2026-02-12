use polars_core::prelude::*;
use polars_error::feature_gated;

use crate::prelude::*;

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
mod join_utils;
pub(crate) use join_utils::ExprOrigin;
mod expand_datasets;
#[cfg(feature = "python")]
pub use expand_datasets::ExpandedPythonScan;
mod predicate_pushdown;
mod projection_pushdown;
pub mod set_order;
mod simplify_expr;
mod slice_pushdown_expr;
mod slice_pushdown_lp;
mod sortedness;
mod stack_opt;

use collapse_and_project::SimpleProjectionAndCollapse;
#[cfg(feature = "cse")]
pub use cse::NaiveExprMerger;
use delay_rechunk::DelayRechunk;
pub use expand_datasets::ExpandedDataset;
use polars_core::config::verbose;
pub use predicate_pushdown::{DynamicPred, PredicateExpr, PredicatePushDown, TrivialPredicateExpr};
pub use projection_pushdown::ProjectionPushDown;
pub use simplify_expr::{SimplifyBooleanRule, SimplifyExprRule};
use slice_pushdown_lp::SlicePushDown;
pub use sortedness::{AExprSorted, IRSorted, are_keys_sorted_any, is_sorted};
pub use stack_opt::{OptimizationRule, OptimizeExprContext, StackOptimizer};

use self::flatten_union::FlattenUnionRule;
pub use crate::frame::{AllowedOptimizations, OptFlags};
pub use crate::plans::conversion::type_coercion::TypeCoercionRule;
use crate::plans::optimizer::count_star::CountStar;
#[cfg(feature = "cse")]
use crate::plans::optimizer::cse::CommonSubExprOptimizer;
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

pub(crate) fn pushdown_maintain_errors() -> bool {
    std::env::var("POLARS_PUSHDOWN_OPT_MAINTAIN_ERRORS").as_deref() == Ok("1")
}

pub(super) fn run_projection_predicate_pushdown(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    pushdown_maintain_errors: bool,
    opt_flags: &OptFlags,
) -> PolarsResult<()> {
    // Should be run before predicate pushdown.
    if opt_flags.projection_pushdown() {
        let mut projection_pushdown_opt = ProjectionPushDown::new();
        let ir = ir_arena.take(root);
        let ir = projection_pushdown_opt.optimize(ir, ir_arena, expr_arena)?;
        ir_arena.replace(root, ir);

        if projection_pushdown_opt.is_count_star {
            let mut count_star_opt = CountStar::new();
            count_star_opt.optimize_plan(ir_arena, expr_arena, root)?;
        }
    }

    if opt_flags.predicate_pushdown() {
        let mut predicate_pushdown_opt =
            PredicatePushDown::new(pushdown_maintain_errors, opt_flags.new_streaming());
        let ir = ir_arena.take(root);
        let ir = predicate_pushdown_opt.optimize(ir, ir_arena, expr_arena)?;
        ir_arena.replace(root, ir);
    }

    Ok(())
}

pub fn optimize(
    logical_plan: DslPlan,
    mut opt_flags: OptFlags,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut Vec<Node>,
    apply_scan_predicate_to_scan_ir: fn(
        Node,
        &mut Arena<IR>,
        &mut Arena<AExpr>,
    ) -> PolarsResult<()>,
) -> PolarsResult<Node> {
    #[allow(dead_code)]
    let verbose = verbose();

    // Gradually fill the rules passed to the optimizer
    let opt = StackOptimizer {};
    let mut rules: Vec<Box<dyn OptimizationRule>> = Vec::with_capacity(8);

    // Unset CSE
    // This can be turned on again during ir-conversion.
    #[allow(clippy::eq_op)]
    #[cfg(feature = "cse")]
    if opt_flags.contains(OptFlags::EAGER) {
        opt_flags &= !(OptFlags::COMM_SUBEXPR_ELIM | OptFlags::COMM_SUBEXPR_ELIM);
    }
    let mut root = to_alp(logical_plan, expr_arena, ir_arena, &mut opt_flags)?;

    #[allow(unused_assignments)]
    let mut comm_subplan_elim = false;
    // Don't run optimizations that don't make sense on a single node.
    // This keeps eager execution more snappy.
    #[cfg(feature = "cse")]
    {
        comm_subplan_elim = opt_flags.contains(OptFlags::COMM_SUBPLAN_ELIM);
    }

    #[cfg(feature = "cse")]
    let comm_subexpr_elim = opt_flags.contains(OptFlags::COMM_SUBEXPR_ELIM);
    #[cfg(not(feature = "cse"))]
    let comm_subexpr_elim = false;

    // Note: This can be in opt_flags in the future if needed.
    let pushdown_maintain_errors = pushdown_maintain_errors();

    // During debug we check if the optimizations have not modified the final schema.
    #[cfg(debug_assertions)]
    let prev_schema = ir_arena.get(root).schema(ir_arena).into_owned();

    let mut _opt_members: &mut Option<MemberCollector> = &mut None;

    macro_rules! get_or_init_members {
        () => {
            _get_or_init_members(_opt_members, root, ir_arena, expr_arena)
        };
    }

    // Run before slice pushdown
    if opt_flags.simplify_expr() {
        #[cfg(feature = "fused")]
        rules.push(Box::new(fused::FusedArithmetic {}));
    }

    let run_pushdowns = if comm_subplan_elim {
        #[allow(unused_assignments)]
        let mut run_pd = true;

        feature_gated!("cse", {
            let members = get_or_init_members!();
            run_pd = if (members.has_sink_multiple || members.has_joins_or_unions)
                && members.has_duplicate_scans()
                && !members.has_cache
            {
                use self::cse::CommonSubPlanOptimizer;

                if verbose {
                    eprintln!("found multiple sources; run comm_subplan_elim")
                }

                root = CommonSubPlanOptimizer::new().optimize(
                    root,
                    ir_arena,
                    expr_arena,
                    pushdown_maintain_errors,
                    &opt_flags,
                    verbose,
                    scratch,
                )?;
                false
            } else {
                true
            }
        });

        run_pd
    } else {
        true
    };

    if opt_flags.slice_pushdown() {
        let mut slice_pushdown_opt = SlicePushDown::new(
            // We don't maintain errors on slice as the behavior is much more predictable that way.
            //
            // Even if we enable maintain_errors (thereby preventing the slice from being pushed),
            // the new-streaming engine still may not error due to early-stopping.
            false, // maintain_errors
        );
        let ir = ir_arena.take(root);
        let ir = slice_pushdown_opt.optimize(ir, ir_arena, expr_arena)?;

        ir_arena.replace(root, ir);

        // Expressions use the stack optimizer.
        rules.push(Box::new(slice_pushdown_opt));
    }

    if run_pushdowns {
        run_projection_predicate_pushdown(
            root,
            ir_arena,
            expr_arena,
            pushdown_maintain_errors,
            &opt_flags,
        )?;
    }

    if opt_flags.fast_projection() {
        rules.push(Box::new(SimpleProjectionAndCollapse::new(
            opt_flags.eager(),
        )));
    }

    if !opt_flags.eager() {
        rules.push(Box::new(DelayRechunk::new()));
    }

    // This optimization removes branches, so we must do it when type coercion
    // is completed.
    if opt_flags.simplify_expr() {
        rules.push(Box::new(SimplifyBooleanRule {}));
    }

    if !opt_flags.eager() {
        rules.push(Box::new(FlattenUnionRule {}));
    }

    root = opt.optimize_loop(&mut rules, expr_arena, ir_arena, root)?;

    if opt_flags.cluster_with_columns() && get_or_init_members!().with_columns_count > 1 {
        cluster_with_columns::optimize(root, ir_arena, expr_arena)
    }

    // This one should run (nearly) last as this modifies the projections
    #[cfg(feature = "cse")]
    if comm_subexpr_elim && !get_or_init_members!().has_ext_context {
        let mut optimizer =
            CommonSubExprOptimizer::new(opt_flags.contains(OptFlags::NEW_STREAMING));
        let ir_node = IRNode::new_mutate(root);

        root = try_with_ir_arena(ir_arena, expr_arena, |arena| {
            let rewritten = ir_node.rewrite(&mut optimizer, arena)?;
            Ok(rewritten.node())
        })?;
    }

    if opt_flags.contains(OptFlags::CHECK_ORDER_OBSERVE) {
        let members = get_or_init_members!();
        if members.has_group_by
            | members.has_sort
            | members.has_distinct
            | members.has_joins_or_unions
        {
            match ir_arena.get(root) {
                IR::SinkMultiple { inputs } => {
                    let mut roots = inputs.clone();
                    for root in &mut roots {
                        if !matches!(ir_arena.get(*root), IR::Sink { .. }) {
                            *root = ir_arena.add(IR::Sink {
                                input: *root,
                                payload: SinkTypeIR::Memory,
                            });
                        }
                    }
                    set_order::simplify_and_fetch_orderings(&roots, ir_arena, expr_arena);
                },
                ir => {
                    let mut tmp_top = root;
                    if !matches!(ir, IR::Sink { .. }) {
                        tmp_top = ir_arena.add(IR::Sink {
                            input: root,
                            payload: SinkTypeIR::Memory,
                        });
                    }
                    _ = set_order::simplify_and_fetch_orderings(&[tmp_top], ir_arena, expr_arena)
                },
            }
        }
    }

    expand_datasets::expand_datasets(root, ir_arena, expr_arena, apply_scan_predicate_to_scan_ir)?;

    // During debug we check if the optimizations have not modified the final schema.
    #[cfg(debug_assertions)]
    {
        // only check by names because we may supercast types.
        assert_eq!(
            prev_schema.iter_names().collect::<Vec<_>>(),
            ir_arena
                .get(root)
                .schema(ir_arena)
                .iter_names()
                .collect::<Vec<_>>()
        );
    };

    Ok(root)
}

fn _get_or_init_members<'a>(
    opt_members: &'a mut Option<MemberCollector>,
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> &'a mut MemberCollector {
    opt_members.get_or_insert_with(|| {
        let mut members = MemberCollector::new();
        members.collect(root, ir_arena, expr_arena);

        members
    })
}
