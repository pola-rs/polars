use polars_core::CHEAP_SERIES_HASH_LIMIT;
use polars_core::prelude::{PlIndexMap, PlIndexSet};
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_utils::arena::{Arena, Node};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::scratch_vec::ScratchVec;

use crate::constants::CSE_REPLACED;
use crate::plans::aexpr::is_inherently_nondeterministic_excluding_udfs_top_level;
use crate::plans::projection_height::{ExprProjectionHeight, aexpr_projection_height_rec};
use crate::plans::visitor::{
    IRNode, IRNodeArena, RewriteRecursion, RewritingVisitor, TreeWalker as _, VisitRecursion,
    Visitor,
};
use crate::plans::{
    AExpr, CanonicalExprId, CanonicalExprMap, ExprIR, IR, IRBuilder, IRFunctionExpr, LiteralValue,
    OutputName,
};
use crate::prelude::ProjectionOptions;
use crate::prelude::visitor::AexprNode;

/// How a visited node participates in CSE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeRole {
    /// Invalidates its enclosing expression, but descendants may still be candidates.
    Refuse,
    /// May occur within a candidate but is not a candidate itself.
    Member,
    /// May be extracted as a common subexpression.
    Candidate,
}

#[derive(Debug, Clone)]
struct ProjectionExprs {
    expr: Vec<ExprIR>,
    /// offset from the back
    /// `expr[expr.len() - common_sub_offset..]`
    /// are the common sub expressions
    common_sub_offset: usize,
}

impl ProjectionExprs {
    fn default_exprs(&self) -> &[ExprIR] {
        &self.expr[..self.expr.len() - self.common_sub_offset]
    }

    fn cse_exprs(&self) -> &[ExprIR] {
        &self.expr[self.expr.len() - self.common_sub_offset..]
    }

    fn new_with_cse(expr: Vec<ExprIR>, common_sub_offset: usize) -> Self {
        Self {
            expr,
            common_sub_offset,
        }
    }
}

fn cse_column_name(id: CanonicalExprId) -> PlSmallStr {
    format_pl_smallstr!("{}{:#x}", CSE_REPLACED, id.as_u64())
}

/// Canonical expression id maps to the number of occurrences in the current plan node.
type SubExprCount = PlIndexMap<CanonicalExprId, u32>;

/// One frame per active node; `false` means its subtree was invalidated.
type ValidityStack = Vec<bool>;

fn skip_pre_visit(ae: &AExpr, is_groupby: bool, element_wise_select_only: bool) -> bool {
    match ae {
        #[cfg(feature = "dynamic_group_by")]
        AExpr::Rolling { .. } => true,
        AExpr::Over { .. } => true,
        #[cfg(feature = "dtype-struct")]
        AExpr::StructEval { .. } => true,
        AExpr::Eval { .. } => true,
        AExpr::Ternary { .. } => is_groupby,
        ae => {
            if element_wise_select_only {
                if is_groupby {
                    true
                } else {
                    !ae.is_elementwise_top_level()
                }
            } else {
                false
            }
        },
    }
}

/// Counts valid CSE candidates while propagating subtree validity to their ancestors.
struct ExprCandidateVisitor<'a> {
    canonical_map: &'a mut CanonicalExprMap,
    se_count: &'a mut SubExprCount,
    validity_stack: &'a mut ValidityStack,
    // During aggregation we only identify element-wise operations
    is_group_by: bool,
    //
    element_wise_only: bool,
}

impl ExprCandidateVisitor<'_> {
    fn new<'a>(
        canonical_map: &'a mut CanonicalExprMap,
        se_count: &'a mut SubExprCount,
        validity_stack: &'a mut ValidityStack,
        is_group_by: bool,
        element_wise_select_only: bool,
    ) -> ExprCandidateVisitor<'a> {
        ExprCandidateVisitor {
            canonical_map,
            se_count,
            validity_stack,
            is_group_by,
            element_wise_only: element_wise_select_only,
        }
    }

    /// Invalidates the enclosing node, if any.
    fn invalidate_parent(&mut self) {
        if let Some(frame) = self.validity_stack.last_mut() {
            *frame = false;
        }
    }
}

/// Classifies how a visited node may participate in CSE.
fn classify(ae: &AExpr, is_group_by: bool) -> NodeRole {
    match ae {
        // Don't allow this for now, as we can get `null().cast()` in ternary expressions.
        // TODO! Add a typed null
        AExpr::Literal(LiteralValue::Scalar(sc)) if sc.is_null() => NodeRole::Refuse,
        AExpr::Literal(LiteralValue::Series(s)) => {
            let dtype = s.dtype();

            // Object and nested types are harder to hash and compare.
            let allow = !(dtype.is_nested() | dtype.is_object());

            if s.len() < CHEAP_SERIES_HASH_LIMIT && allow {
                NodeRole::Member
            } else {
                NodeRole::Refuse
            }
        },
        AExpr::Literal(_) => NodeRole::Member,
        AExpr::Column(_) => NodeRole::Member,
        AExpr::Len => {
            if is_group_by {
                NodeRole::Refuse
            } else {
                NodeRole::Member
            }
        },
        ae if is_inherently_nondeterministic_excluding_udfs_top_level(ae) => NodeRole::Refuse,
        #[cfg(feature = "rolling_window")]
        AExpr::Function {
            function: IRFunctionExpr::RollingExpr { .. },
            ..
        } => NodeRole::Refuse,
        _ => {
            // Group-by temporaries are evaluated before aggregation, so only elementwise
            // expressions have the correct length.
            if is_group_by {
                if !ae.is_elementwise_top_level() {
                    return NodeRole::Refuse;
                }
                match ae {
                    AExpr::Cast { .. } => NodeRole::Member,
                    _ => NodeRole::Candidate,
                }
            } else {
                NodeRole::Candidate
            }
        },
    }
}

impl Visitor for ExprCandidateVisitor<'_> {
    type Node = AexprNode;
    type Arena = Arena<AExpr>;

    fn pre_visit(
        &mut self,
        node: &Self::Node,
        arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        if skip_pre_visit(
            node.to_aexpr(arena),
            self.is_group_by,
            self.element_wise_only,
        ) {
            // No frame is pushed for this node, so invalidate the parent directly.
            self.invalidate_parent();
            return Ok(VisitRecursion::Skip);
        }

        self.validity_stack.push(true);

        Ok(VisitRecursion::Continue)
    }

    fn post_visit(
        &mut self,
        node: &Self::Node,
        arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        let ae = node.to_aexpr(arena);
        let subtree_is_valid = self.validity_stack.pop().unwrap();

        let role = classify(ae, self.is_group_by);

        if !subtree_is_valid || role == NodeRole::Refuse {
            self.invalidate_parent();
        }

        if subtree_is_valid && role == NodeRole::Candidate {
            let id = self.canonical_map.resolve(node.node(), arena);
            *self.se_count.entry(id).or_insert(0) += 1;
        }

        Ok(VisitRecursion::Continue)
    }
}

struct CommonSubExprRewriter<'a> {
    sub_expr_map: &'a SubExprCount,
    canonical_map: &'a CanonicalExprMap,
    /// keep track of the replaced identifiers.
    replaced_identifiers: &'a mut PlIndexSet<CanonicalExprId>,

    /// Indicates if this expression is rewritten.
    rewritten: bool,
    is_group_by: bool,
    is_element_wise_select_only: bool,
}

impl<'a> CommonSubExprRewriter<'a> {
    fn new(
        canonical_map: &'a CanonicalExprMap,
        sub_expr_map: &'a SubExprCount,
        replaced_identifiers: &'a mut PlIndexSet<CanonicalExprId>,
        is_group_by: bool,
        is_element_wise_select_only: bool,
    ) -> Self {
        Self {
            sub_expr_map,
            canonical_map,
            replaced_identifiers,
            rewritten: false,
            is_group_by,
            is_element_wise_select_only,
        }
    }

    /// Returns the canonical ID and count when this node's equivalence class was accepted as a
    /// CSE candidate for the current plan node.
    fn candidate(&self, node: Node) -> Option<(CanonicalExprId, u32)> {
        let id = self.canonical_map.cached_id(node)?;
        let count = *self.sub_expr_map.get(&id)?;
        Some((id, count))
    }
}

// # Example
// Say we are rewriting `col(foo).sum() + col(foo).sum() * col(bar)`, where `col(foo).sum()` was
// counted twice.
//
//     binary: +
//
//       |                            |
//
//     sum                          binary: *
//
//       |                            |                       |
//
//     col(foo)                     col(bar)                  sum
//
//                                                            |
//
//                                                         col(foo)
//
// call stack
// pre-visit    binary: +   -> counted once      -> no_mutate_and_continue -> visits children
// pre-visit    sum         -> counted twice     -> mutate_and_stop -> does not visit children
// pre-visit    binary: *   -> counted once      -> no_mutate_and_continue -> visits children
// pre-visit    col(bar)    -> not a candidate   -> stop, it is a leaf
// pre-visit    sum         -> counted twice     -> mutate_and_stop -> does not visit children
//
// Both `sum` nodes resolve to the same [`CanonicalExprId`], so both are replaced by a reference
// to the same temporary column.
impl RewritingVisitor for CommonSubExprRewriter<'_> {
    type Node = AexprNode;
    type Arena = Arena<AExpr>;

    fn pre_visit(
        &mut self,
        ae_node: &Self::Node,
        arena: &mut Self::Arena,
    ) -> PolarsResult<RewriteRecursion> {
        let ae = ae_node.to_aexpr(arena);
        if skip_pre_visit(ae, self.is_group_by, self.is_element_wise_select_only) {
            return Ok(RewriteRecursion::Stop);
        }

        if let Some((id, count)) = self.candidate(ae_node.node())
            && count > 1
        {
            self.replaced_identifiers.insert(id);
            return Ok(RewriteRecursion::MutateAndStop);
        }

        let recurse = if ae_node.is_leaf(arena) {
            RewriteRecursion::Stop
        } else {
            RewriteRecursion::NoMutateAndContinue
        };
        Ok(recurse)
    }

    fn mutate(
        &mut self,
        mut node: Self::Node,
        arena: &mut Self::Arena,
    ) -> PolarsResult<Self::Node> {
        let (id, count) = self
            .candidate(node.node())
            .expect("mutated node was not a CSE candidate");
        debug_assert!(count > 1);

        node.assign(AExpr::col(cse_column_name(id)), arena);
        self.rewritten = true;

        Ok(node)
    }
}

pub(crate) struct CommonSubExprOptimizer {
    /// Kept for the whole pass so distinct canonical expressions receive distinct temporary
    /// names plan-wide. Resolved `AExpr` nodes must not be mutated in place.
    canonical_map: CanonicalExprMap,
    // amortize allocations
    // these are cleared per lp node
    se_count: SubExprCount,
    replaced_identifiers: PlIndexSet<CanonicalExprId>,
    // these are cleared per expr node
    validity_stack: ValidityStack,
    // Set by the streaming engine
    // Only supports element-wise CSEE
    // on SELECT/HSTACK
    element_wise_select_only: bool,

    nodes_scratch: ScratchVec<Node>,
    heights_scratch: ScratchVec<ExprProjectionHeight>,
}

impl CommonSubExprOptimizer {
    pub(crate) fn new(element_wise_select_only: bool) -> Self {
        Self {
            canonical_map: CanonicalExprMap::new(),
            se_count: Default::default(),
            validity_stack: Default::default(),
            replaced_identifiers: Default::default(),
            element_wise_select_only,
            nodes_scratch: ScratchVec::default(),
            heights_scratch: ScratchVec::default(),
        }
    }

    fn visit_expression(
        &mut self,
        ae_node: AexprNode,
        is_group_by: bool,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<()> {
        let mut visitor = ExprCandidateVisitor::new(
            &mut self.canonical_map,
            &mut self.se_count,
            &mut self.validity_stack,
            is_group_by,
            self.element_wise_select_only,
        );
        ae_node.visit(&mut visitor, expr_arena).map(|_| ())
    }

    /// Mutate the expression.
    /// Returns a new expression and a `bool` indicating if it was rewritten or not.
    fn mutate_expression(
        &mut self,
        ae_node: AexprNode,
        is_group_by: bool,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<(AexprNode, bool)> {
        let mut rewriter = CommonSubExprRewriter::new(
            &self.canonical_map,
            &self.se_count,
            &mut self.replaced_identifiers,
            is_group_by,
            self.element_wise_select_only,
        );
        ae_node
            .rewrite(&mut rewriter, expr_arena)
            .map(|out| (out, rewriter.rewritten))
    }

    fn find_cse(
        &mut self,
        expr: &[ExprIR],
        expr_arena: &mut Arena<AExpr>,
        is_group_by: bool,
        schema: &Schema,
    ) -> PolarsResult<Option<ProjectionExprs>> {
        // First get all cse's.
        for e in expr {
            // A balanced traversal leaves the stack empty, but an early return from a
            // previous expression might not
            self.validity_stack.clear();

            // Visit expressions and collect sub-expression counts.
            let ae_node = AexprNode::new(e.node());
            self.visit_expression(ae_node, is_group_by, expr_arena)?;
        }

        let has_sub_expr = self.se_count.values().any(|&count| count > 1);

        if has_sub_expr {
            let mut new_expr = Vec::with_capacity((expr.len() as f64 * 1.3) as usize);

            // Then rewrite the expressions that have a cse count > 1.
            for e in expr {
                let ae_node = AexprNode::new(e.node());

                let (out, rewritten) = self.mutate_expression(ae_node, is_group_by, expr_arena)?;

                let out_node = out.node();
                let mut out_e = e.clone();
                let new_node = if !rewritten {
                    out_e
                } else {
                    out_e.set_node(out_node);

                    // If we don't end with an alias we add an alias. Because the normal left-hand
                    // rule we apply for determining the name will not work we now refer to
                    // intermediate temporary names starting with the `CSE_REPLACED` constant.
                    if !e.has_alias() {
                        let name = ae_node.to_field(schema, expr_arena)?.name;
                        out_e.set_alias(name.clone());
                    }
                    out_e
                };
                new_expr.push(new_node)
            }
            // Add the tmp columns
            for &id in self.replaced_identifiers.iter() {
                let node = self.canonical_map.representative(id);

                // Avoid accidentally broadcasting <scalar literal>.<elementwise_ops..>
                if self.element_wise_select_only
                    && !matches!(
                        aexpr_projection_height_rec(
                            node,
                            expr_arena,
                            &mut self.nodes_scratch,
                            &mut self.heights_scratch
                        ),
                        ExprProjectionHeight::Column
                    )
                {
                    return Ok(None);
                }

                let out_e = ExprIR::new(node, OutputName::Alias(cse_column_name(id)));
                new_expr.push(out_e)
            }
            let expr = ProjectionExprs::new_with_cse(new_expr, self.replaced_identifiers.len());
            Ok(Some(expr))
        } else {
            Ok(None)
        }
    }
}

impl RewritingVisitor for CommonSubExprOptimizer {
    type Node = IRNode;
    type Arena = IRNodeArena;

    fn pre_visit(
        &mut self,
        node: &Self::Node,
        arena: &mut Self::Arena,
    ) -> PolarsResult<RewriteRecursion> {
        use IR::*;
        Ok(match node.to_alp(&arena.0) {
            Select { .. } | HStack { .. } | GroupBy { .. } => RewriteRecursion::MutateAndContinue,
            _ => RewriteRecursion::NoMutateAndContinue,
        })
    }

    fn mutate(&mut self, node: Self::Node, arena: &mut Self::Arena) -> PolarsResult<Self::Node> {
        self.se_count.clear();
        self.replaced_identifiers.clear();

        let arena_idx = node.node();
        let alp = arena.0.get(arena_idx);

        match alp {
            IR::Select {
                input,
                expr,
                schema,
                options,
            } => {
                let input_schema = arena.0.get(*input).schema(&arena.0);
                if let Some(expr) =
                    self.find_cse(expr, &mut arena.1, false, input_schema.as_ref().as_ref())?
                {
                    let schema = schema.clone();
                    let options = *options;

                    let lp = IRBuilder::new(*input, &mut arena.1, &mut arena.0)
                        .with_columns(
                            expr.cse_exprs().to_vec(),
                            ProjectionOptions {
                                run_parallel: options.run_parallel,
                                duplicate_check: options.duplicate_check,
                                // These columns might have different
                                // lengths from the dataframe, but
                                // they are only temporaries that will
                                // be removed by the evaluation of the
                                // default_exprs and the subsequent
                                // projection.
                                should_broadcast: false,
                            },
                        )
                        .build();
                    let input = arena.0.add(lp);

                    let lp = IR::Select {
                        input,
                        expr: expr.default_exprs().to_vec(),
                        schema,
                        options,
                    };
                    arena.0.replace(arena_idx, lp);
                }
            },
            IR::HStack {
                input,
                exprs,
                schema,
                options,
            } => {
                let input_schema = arena.0.get(*input).schema(&arena.0);
                if let Some(exprs) =
                    self.find_cse(exprs, &mut arena.1, false, input_schema.as_ref().as_ref())?
                {
                    let schema = schema.clone();
                    let options = *options;
                    let input = *input;

                    let lp = IRBuilder::new(input, &mut arena.1, &mut arena.0)
                        .with_columns(
                            exprs.cse_exprs().to_vec(),
                            // These columns might have different
                            // lengths from the dataframe, but they
                            // are only temporaries that will be
                            // removed by the evaluation of the
                            // default_exprs and the subsequent
                            // projection.
                            ProjectionOptions {
                                run_parallel: options.run_parallel,
                                duplicate_check: options.duplicate_check,
                                should_broadcast: false,
                            },
                        )
                        .with_columns(exprs.default_exprs().to_vec(), options)
                        .build();
                    let input = arena.0.add(lp);

                    let lp = IR::SimpleProjection {
                        input,
                        columns: schema,
                    };
                    arena.0.replace(arena_idx, lp);
                }
            },
            IR::GroupBy {
                input,
                keys,
                aggs,
                options,
                maintain_order,
                apply,
                schema,
            } if !self.element_wise_select_only => {
                let input_schema = arena.0.get(*input).schema(&arena.0);
                if let Some(aggs) =
                    self.find_cse(aggs, &mut arena.1, true, input_schema.as_ref().as_ref())?
                {
                    let keys = keys.clone();
                    let options = options.clone();
                    let schema = schema.clone();
                    let apply = apply.clone();
                    let maintain_order = *maintain_order;
                    let input = *input;

                    let lp = IRBuilder::new(input, &mut arena.1, &mut arena.0)
                        .with_columns(aggs.cse_exprs().to_vec(), Default::default())
                        .build();
                    let input = arena.0.add(lp);

                    let lp = IR::GroupBy {
                        input,
                        keys,
                        aggs: aggs.default_exprs().to_vec(),
                        options,
                        schema,
                        maintain_order,
                        apply,
                    };
                    arena.0.replace(arena_idx, lp);
                }
            },
            _ => {},
        }

        Ok(node)
    }
}
