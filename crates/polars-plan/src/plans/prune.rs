//! IR pruning. Pruning copies the reachable IR and expressions into a set of destination arenas.

use polars_core::prelude::{InitHashMaps as _, PlHashMap};
use polars_utils::arena::{Arena, Node};
use polars_utils::unique_id::UniqueId;
use recursive::recursive;

use crate::plans::{AExpr, IR, IRPlan, IRPlanRef};

/// Returns a pruned copy of this plan with new arenas (without unreachable nodes).
///
/// The cache hit count is updated based on the number of consumers in the pruned plan.
///
/// The original plan and arenas are not modified.
pub fn prune_plan(ir_plan: IRPlanRef<'_>) -> IRPlan {
    let mut ir_arena = Arena::new();
    let mut expr_arena = Arena::new();
    let [root] = prune(
        &[ir_plan.lp_top],
        ir_plan.lp_arena,
        ir_plan.expr_arena,
        &mut ir_arena,
        &mut expr_arena,
    )
    .try_into()
    .unwrap();
    IRPlan {
        lp_top: root,
        lp_arena: ir_arena,
        expr_arena,
    }
}

/// Prunes a subgraph reachable from the supplied roots into the supplied arenas.
///
/// The returned nodes point to the pruned copies of the supplied roots in the same order.
///
/// The cache hit count is updated based on the number of consumers in the pruned subgraph.
///
/// The original plan and arenas are not modified.
pub fn prune(
    roots: &[Node],
    src_ir: &Arena<IR>,
    src_expr: &Arena<AExpr>,
    dst_ir: &mut Arena<IR>,
    dst_expr: &mut Arena<AExpr>,
) -> Vec<Node> {
    let mut ctx = CopyContext {
        src_ir,
        src_expr,
        dst_ir,
        dst_expr,
        dst_caches: PlHashMap::new(),
        roots: PlHashMap::from_iter(roots.iter().map(|node| (*node, None))),
    };

    let dst_roots: Vec<Node> = roots.iter().map(|&root| ctx.copy_ir(root)).collect();

    assert!(ctx.roots.values().all(|v| v.is_some()));

    dst_roots
}

struct CopyContext<'a> {
    src_ir: &'a Arena<IR>,
    src_expr: &'a Arena<AExpr>,
    dst_ir: &'a mut Arena<IR>,
    dst_expr: &'a mut Arena<AExpr>,
    // Caches and the matching dst nodes.
    dst_caches: PlHashMap<UniqueId, Node>,
    // Root nodes and the matching dst nodes. Needed to ensure they are visited only once,
    // in case they are reachable from other root nodes.
    roots: PlHashMap<Node, Option<Node>>,
}

impl<'a> CopyContext<'a> {
    // Copies the IR subgraph from src to dst.
    #[recursive]
    fn copy_ir(&mut self, src_node: Node) -> Node {
        // If this cache was already visited, bump the cache hits and don't traverse it.
        // This is before the root node check, so that the hit count gets bumped for every visit.
        if let IR::Cache { id, .. } = self.src_ir.get(src_node) {
            if let Some(cache) = self.dst_caches.get(id) {
                return *cache;
            }
        }

        // If this is one of the root nodes and was already visited, don't visit again, just return
        // the matching dst node.
        if let Some(&Some(root_node)) = self.roots.get(&src_node) {
            return root_node;
        }

        let src_ir = self.src_ir.get(src_node);

        let mut dst_ir = src_ir.clone();

        // Recurse into inputs
        dst_ir = dst_ir.with_inputs(src_ir.inputs().map(|input| self.copy_ir(input)));

        // Recurse into expressions
        dst_ir = dst_ir.with_exprs(src_ir.exprs().map(|expr| {
            let mut expr = expr.clone();
            expr.set_node(self.copy_expr(expr.node()));
            expr
        }));

        // Add this node
        let dst_node = self.dst_ir.add(dst_ir);

        // If this is a cache, reset the hit count and store the dst node.
        if let IR::Cache { id, .. } = self.dst_ir.get_mut(dst_node) {
            let prev = self.dst_caches.insert(*id, dst_node);
            assert!(prev.is_none(), "cache {id} was traversed twice");
        }

        // If this is one of the root nodes, store the dst node.
        self.roots.entry(src_node).and_modify(|e| {
            assert!(
                e.replace(dst_node).is_none(),
                "root node was traversed twice"
            )
        });

        dst_node
    }

    /// Copies the expression subgraph from src to dst.
    #[recursive]
    fn copy_expr(&mut self, node: Node) -> Node {
        let expr = self.src_expr.get(node);

        let mut inputs = vec![];
        expr.inputs_rev(&mut inputs);

        for input in &mut inputs {
            *input = self.copy_expr(*input);
        }
        inputs.reverse();

        let mut dst_expr = expr.clone().replace_inputs(&inputs);

        // Fix up eval, the evaluation subtree is not treated as an input,
        // so it needs to be copied manually.
        if let AExpr::Eval { evaluation, .. } = &mut dst_expr {
            *evaluation = self.copy_expr(*evaluation);
        }

        self.dst_expr.add(dst_expr)
    }
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::*;

    use super::*;
    use crate::dsl::{SinkTypeIR, col, lit};
    use crate::plans::{ArenaLpIter as _, ExprToIRContext, to_expr_ir};

    //           SINK[right]
    //               |
    // SINK[left]   SORT   SINK[extra]
    //     |        /        /
    //   CACHE ----+--------+
    //     |
    //  FILTER
    //     |
    //   SCAN
    struct BranchedPlan {
        ir_arena: Arena<IR>,
        expr_arena: Arena<AExpr>,
        scan: Node,
        filter: Node,
        cache: Node,
        left_sink: Node,
        sort: Node,
        right_sink: Node,
        extra_sink: Node,
    }

    #[test]
    fn test_pruned_subgraph_matches() {
        let p = BranchedPlan::new();

        #[rustfmt::skip]
        let cases: &[&[Node]] = &[
            // Single
            &[p.scan],
            &[p.cache],
            &[p.left_sink],
            &[p.right_sink],
            // Multiple
            &[p.left_sink, p.right_sink],
            &[p.left_sink, p.right_sink, p.extra_sink],
            // Duplicate
            &[p.left_sink, p.left_sink],
            &[p.cache, p.cache],
            // A mess
            &[p.filter, p.scan, p.left_sink, p.cache, p.right_sink, p.sort, p.cache, p.right_sink],
        ];

        for &case in cases.iter() {
            let (pruned, arenas) = p.prune(case);
            for (&orig, pruned) in case.iter().zip(pruned) {
                let orig_plan = p.plan(orig);
                let pruned_plan = arenas.plan(pruned);
                assert!(
                    plans_equal(orig_plan, pruned_plan),
                    "orig: {}, pruned: {}",
                    orig_plan.display(),
                    pruned_plan.display()
                );
            }
        }
    }

    #[test]
    fn test_pruned_arena_size() {
        let p = BranchedPlan::new();

        #[rustfmt::skip]
        let cases: &[(&[Node], usize)] = &[
            (&[p.scan], 1),
            (&[p.cache], 3),
            (&[p.cache, p.cache], 3),
            (&[p.left_sink], 4),
            (&[p.left_sink, p.left_sink], 4),
            (&[p.right_sink], 5),
            (&[p.left_sink, p.right_sink], 6),
            (&[p.filter, p.scan, p.left_sink, p.cache, p.right_sink, p.sort, p.cache, p.right_sink], 6),
            (&[p.left_sink, p.right_sink, p.extra_sink], 7),
        ];

        for (i, &(case, expected_arena_size)) in cases.iter().enumerate() {
            let (_, arenas) = p.prune(case);
            assert_eq!(
                arenas.ir.len(),
                expected_arena_size,
                "case: {i}, pruned_ir: {:?}",
                arenas.ir
            );
        }
    }

    fn plans_equal(a: IRPlanRef<'_>, b: IRPlanRef<'_>) -> bool {
        let iter_a = a.lp_arena.iter(a.lp_top);
        let iter_b = b.lp_arena.iter(b.lp_top);
        for ((_, ir_a), (_, ir_b)) in iter_a.zip(iter_b) {
            if std::mem::discriminant(ir_a) != std::mem::discriminant(ir_b)
                || !exprs_equal(ir_a, a.expr_arena, ir_b, b.expr_arena)
            {
                return false;
            }
        }
        true
    }

    fn exprs_equal(ir_a: &IR, arena_a: &Arena<AExpr>, ir_b: &IR, arena_b: &Arena<AExpr>) -> bool {
        let [a, b] = [(ir_a, arena_a), (ir_b, arena_b)].map(|(ir, arena)| {
            ir.exprs()
                .map(|e| (e.output_name_inner().clone(), e.to_expr(arena)))
        });
        a.eq(b)
    }

    impl BranchedPlan {
        pub fn new() -> Self {
            let mut ir_arena = Arena::new();
            let mut expr_arena = Arena::new();
            let schema = Schema::from_iter([Field::new("a".into(), DataType::UInt8)]);

            let scan = ir_arena.add(IR::DataFrameScan {
                df: Arc::new(DataFrame::empty_with_schema(&schema)),
                schema: Arc::new(schema.clone()),
                output_schema: None,
            });

            let mut ctx = ExprToIRContext::new(&mut expr_arena, &schema);
            ctx.allow_unknown = true;
            let filter = ir_arena.add(IR::Filter {
                input: scan,
                predicate: to_expr_ir(col("a").gt_eq(lit(10)), &mut ctx).unwrap(),
            });

            // Throw in an unreachable node
            ir_arena.add(IR::Invalid);

            let cache = ir_arena.add(IR::Cache {
                input: filter,
                id: UniqueId::new(),
            });

            let left_sink = ir_arena.add(IR::Sink {
                input: cache,
                payload: SinkTypeIR::Memory,
            });

            // Throw in an unreachable node
            ir_arena.add(IR::Invalid);

            let mut ctx = ExprToIRContext::new(&mut expr_arena, &schema);
            ctx.allow_unknown = true;
            let sort = ir_arena.add(IR::Sort {
                input: cache,
                by_column: vec![to_expr_ir(col("a"), &mut ctx).unwrap()],
                slice: None,
                sort_options: Default::default(),
            });

            let right_sink = ir_arena.add(IR::Sink {
                input: sort,
                payload: SinkTypeIR::Memory,
            });

            // Throw in an unused sink
            let extra_sink = ir_arena.add(IR::Sink {
                input: cache,
                payload: SinkTypeIR::Memory,
            });

            Self {
                ir_arena,
                expr_arena,
                scan,
                filter,
                cache,
                left_sink,
                sort,
                right_sink,
                extra_sink,
            }
        }

        pub fn prune(&self, roots: &[Node]) -> (Vec<Node>, Arenas) {
            let mut arenas = Arenas {
                ir: Arena::new(),
                expr: Arena::new(),
            };
            let pruned = prune(
                roots,
                &self.ir_arena,
                &self.expr_arena,
                &mut arenas.ir,
                &mut arenas.expr,
            );
            (pruned, arenas)
        }

        pub fn plan(&'_ self, node: Node) -> IRPlanRef<'_> {
            IRPlanRef {
                lp_top: node,
                lp_arena: &self.ir_arena,
                expr_arena: &self.expr_arena,
            }
        }
    }

    struct Arenas {
        ir: Arena<IR>,
        expr: Arena<AExpr>,
    }

    impl Arenas {
        pub fn plan(&'_ self, root: Node) -> IRPlanRef<'_> {
            IRPlanRef {
                lp_top: root,
                lp_arena: &self.ir,
                expr_arena: &self.expr,
            }
        }
    }
}
