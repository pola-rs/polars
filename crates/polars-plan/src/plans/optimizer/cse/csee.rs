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

type Accepted = Option<(VisitRecursion, bool)>;
// Don't allow this node in a cse.
const REFUSE_NO_MEMBER: Accepted = Some((VisitRecursion::Continue, false));
// Don't allow this node, but allow as a member of a cse.
const REFUSE_ALLOW_MEMBER: Accepted = Some((VisitRecursion::Continue, true));
const REFUSE_SKIP: Accepted = Some((VisitRecursion::Skip, false));
// Accept this node.
const ACCEPT: Accepted = None;

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

/// Canonical expression id maps to Expr Node and count.
type SubExprCount = PlIndexMap<CanonicalExprId, (Node, u32)>;

#[derive(Debug)]
enum VisitRecord {
    /// entered a new expression
    Entered,
    /// Every visited sub-expression pushes whether it is valid to the stack.
    // This can be `AND` accumulated by the lineage of the expression to determine
    // of the whole expression can be added.
    // For instance a in a group_by we only want to use elementwise operation in cse:
    // - `(col("a") * 2).sum(), (col("a") * 2)` -> we want to do `col("a") * 2` on a `with_columns`
    // - `col("a").sum() * col("a").sum()` -> we don't want `sum` to run on `with_columns`
    // as that doesn't have groups context. If we encounter a `sum` it should be flagged as `false`
    //
    // This should have the following stack
    // id        valid
    // col(a)   true
    // sum      false
    // col(a)   true
    // sum      false
    // binary   true
    // -------------- accumulated
    //          false
    SubExprValid(bool),
}

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

/// Records a [`CanonicalExprId`] for every valid CSE candidate.
///
/// The visitor uses a `visit_stack` to track traversal order.
///
/// # Entering a node
/// When `pre-visit` is called we enter a new (sub)-expression and
/// we add `Entered` to the stack.
/// # Leaving a node
/// On `post_visit`, we pop and combine the validity records belonging to the node's descendants.
/// This determines whether its subtree permits the node to be considered as a CSE candidate.
//
// # Example (this is not a docstring as clippy complains about spacing)
// Say we have the expression: `(col("f00").min() * col("bar")).sum()`
// with the following call tree:
//
//     sum
//
//       |
//
//     binary: *
//
//       |              |
//
//     col(bar)         min
//
//                      |
//
//                      col(f00)
//
// # call order
// function-called              stack                stack-after(pop until E, push V)
// pre-visit: sum                E                        -
// pre-visit: binary: *          EE                       -
// pre-visit: col(bar)           EEE                      -
// post-visit: col(bar)	         EEE                      EEV
// pre-visit: min                EEVE                     -
// pre-visit: col(f00)           EEVEE                    -
// post-visit: col(f00)	         EEVEE                    EEVEV
// post-visit: min	             EEVEV                    EEVV
// post-visit: binary: *         EEVV                     EV
// post-visit: sum               EV                       V
struct ExprIdentifierVisitor<'a> {
    canonical_map: &'a mut CanonicalExprMap,
    se_count: &'a mut SubExprCount,
    visit_stack: &'a mut Vec<VisitRecord>,
    // Whether a repeated CSE candidate was found.
    has_sub_expr: bool,
    // During aggregation we only identify element-wise operations
    is_group_by: bool,
    //
    element_wise_only: bool,
}

impl ExprIdentifierVisitor<'_> {
    fn new<'a>(
        canonical_map: &'a mut CanonicalExprMap,
        se_count: &'a mut SubExprCount,
        visit_stack: &'a mut Vec<VisitRecord>,
        is_group_by: bool,
        element_wise_select_only: bool,
    ) -> ExprIdentifierVisitor<'a> {
        ExprIdentifierVisitor {
            canonical_map,
            se_count,
            visit_stack,
            has_sub_expr: false,
            is_group_by,
            element_wise_only: element_wise_select_only,
        }
    }

    /// Pop all visit-records until an `Entered` is found. We `AND` accumulate the validity of
    /// all `SubExprValid`s and return it.
    /// This works due to the stack.
    /// If we traverse another expression in the mean time, it will get popped of the stack first
    /// so the returned validity belongs to a single sub-expression
    fn pop_until_entered(&mut self) -> bool {
        let mut is_valid_accumulated = true;

        while let Some(item) = self.visit_stack.pop() {
            match item {
                VisitRecord::Entered => return is_valid_accumulated,
                VisitRecord::SubExprValid(valid) => is_valid_accumulated &= valid,
            }
        }
        unreachable!()
    }

    /// return `None` -> node is accepted
    /// return `Some(_)` node is not accepted and apply the given recursion operation
    /// `Some(_, true)` don't accept this node, but can be a member of a cse.
    /// `Some(_,  false)` don't accept this node, and don't allow as a member of a cse.
    fn accept_node_post_visit(&self, ae: &AExpr) -> Accepted {
        match ae {
            // window expressions should `evaluate_on_groups`, not `evaluate`
            // so we shouldn't cache the children as they are evaluated incorrectly
            #[cfg(feature = "dynamic_group_by")]
            AExpr::Rolling { .. } => REFUSE_SKIP,
            AExpr::Over { .. } => REFUSE_SKIP,
            // Don't allow this for now, as we can get `null().cast()` in ternary expressions.
            // TODO! Add a typed null
            AExpr::Literal(LiteralValue::Scalar(sc)) if sc.is_null() => REFUSE_NO_MEMBER,
            AExpr::Literal(s) => {
                match s {
                    LiteralValue::Series(s) => {
                        let dtype = s.dtype();

                        // Object and nested types are harder to hash and compare.
                        let allow = !(dtype.is_nested() | dtype.is_object());

                        if s.len() < CHEAP_SERIES_HASH_LIMIT && allow {
                            REFUSE_ALLOW_MEMBER
                        } else {
                            REFUSE_NO_MEMBER
                        }
                    },
                    _ => REFUSE_ALLOW_MEMBER,
                }
            },
            AExpr::Column(_) => REFUSE_ALLOW_MEMBER,
            AExpr::Len => {
                if self.is_group_by {
                    REFUSE_NO_MEMBER
                } else {
                    REFUSE_ALLOW_MEMBER
                }
            },
            ae if is_inherently_nondeterministic_excluding_udfs_top_level(ae) => REFUSE_NO_MEMBER,
            #[cfg(feature = "rolling_window")]
            AExpr::Function {
                function: IRFunctionExpr::RollingExpr { .. },
                ..
            } => REFUSE_NO_MEMBER,
            _ => {
                // During aggregation we only store elementwise operation in the state
                // other operations we cannot add to the state as they have the output size of the
                // groups, not the original dataframe
                if self.is_group_by {
                    if !ae.is_elementwise_top_level() {
                        return REFUSE_NO_MEMBER;
                    }
                    match ae {
                        AExpr::Cast { .. } => REFUSE_ALLOW_MEMBER,
                        _ => ACCEPT,
                    }
                } else {
                    ACCEPT
                }
            },
        }
    }
}

impl Visitor for ExprIdentifierVisitor<'_> {
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
            // Still add to the stack so that a parent becomes invalidated.
            self.visit_stack.push(VisitRecord::SubExprValid(false));
            return Ok(VisitRecursion::Skip);
        }

        self.visit_stack.push(VisitRecord::Entered);

        Ok(VisitRecursion::Continue)
    }

    fn post_visit(
        &mut self,
        node: &Self::Node,
        arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        let ae = node.to_aexpr(arena);

        let is_valid_accumulated = self.pop_until_entered();

        if !is_valid_accumulated {
            self.visit_stack.push(VisitRecord::SubExprValid(false));
            return Ok(VisitRecursion::Continue);
        }

        // If we don't store this node
        // we only push the visit_stack, so the parents know the trail.
        if let Some((recurse, local_is_valid)) = self.accept_node_post_visit(ae) {
            self.visit_stack
                .push(VisitRecord::SubExprValid(local_is_valid));
            return Ok(recurse);
        }

        let id = self.canonical_map.resolve(node.node(), arena);

        // We popped until entered, push this node's validity on the stack so the trail
        // is available for the parent expression.
        self.visit_stack.push(VisitRecord::SubExprValid(true));

        let (_, se_count) = self.se_count.entry(id).or_insert((node.node(), 0));

        *se_count += 1;
        self.has_sub_expr |= *se_count > 1;

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
        let (_, count) = self.sub_expr_map.get(&id)?;
        Some((id, *count))
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
        // `mutate` is only reached through `MutateAndStop`, so this is the very node `pre_visit`
        // accepted, and it still carries its original arena node.
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
    visit_stack: Vec<VisitRecord>,
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
            visit_stack: Default::default(),
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
        element_wise_select_only: bool,
    ) -> PolarsResult<bool> {
        let mut visitor = ExprIdentifierVisitor::new(
            &mut self.canonical_map,
            &mut self.se_count,
            &mut self.visit_stack,
            is_group_by,
            element_wise_select_only,
        );
        ae_node.visit(&mut visitor, expr_arena).map(|_| ())?;
        Ok(visitor.has_sub_expr)
    }

    /// Mutate the expression.
    /// Returns a new expression and a `bool` indicating if it was rewritten or not.
    fn mutate_expression(
        &mut self,
        ae_node: AexprNode,
        is_group_by: bool,
        expr_arena: &mut Arena<AExpr>,
        element_wise_select_only: bool,
    ) -> PolarsResult<(AexprNode, bool)> {
        let mut rewriter = CommonSubExprRewriter::new(
            &self.canonical_map,
            &self.se_count,
            &mut self.replaced_identifiers,
            is_group_by,
            element_wise_select_only,
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
        element_wise_select_only: bool,
    ) -> PolarsResult<Option<ProjectionExprs>> {
        let mut has_sub_expr = false;

        // First get all cse's.
        for e in expr {
            // An early return may leave records from the previous expression on the stack.
            self.visit_stack.clear();

            // Visit expressions and collect sub-expression counts.
            let ae_node = AexprNode::new(e.node());
            has_sub_expr |=
                self.visit_expression(ae_node, is_group_by, expr_arena, element_wise_select_only)?;
        }

        if has_sub_expr {
            let mut new_expr = Vec::with_capacity((expr.len() as f64 * 1.3) as usize);

            // Then rewrite the expressions that have a cse count > 1.
            for e in expr {
                let ae_node = AexprNode::new(e.node());

                let (out, rewritten) = self.mutate_expression(
                    ae_node,
                    is_group_by,
                    expr_arena,
                    element_wise_select_only,
                )?;

                let out_node = out.node();
                let mut out_e = e.clone();
                let new_node = if !rewritten {
                    out_e
                } else {
                    out_e.set_node(out_node);

                    // Ensure the function ExprIR's have the proper names.
                    // This is needed for structs to get the proper field
                    // This mutates reconstructed nodes in place. They were freshly allocated by
                    // the rewriter and have not been resolved by `canonical_map`.
                    let mut scratch = vec![];
                    let mut stack = vec![(e.node(), out_node)];
                    while let Some((original, new)) = stack.pop() {
                        // Don't follow identical nodes.
                        if original == new {
                            continue;
                        }
                        scratch.clear();
                        let aes = expr_arena.get_disjoint_mut([original, new]);

                        // Only follow paths that are the same.
                        if std::mem::discriminant(aes[0]) != std::mem::discriminant(aes[1]) {
                            continue;
                        }

                        aes[0].inputs_rev(&mut scratch);
                        let offset = scratch.len();
                        aes[1].inputs_rev(&mut scratch);

                        // If they have a different number of inputs, we don't follow the nodes.
                        if scratch.len() != offset * 2 {
                            continue;
                        }

                        for i in 0..scratch.len() / 2 {
                            stack.push((scratch[i], scratch[i + offset]));
                        }

                        match expr_arena.get_disjoint_mut([original, new]) {
                            [
                                AExpr::Function {
                                    input: input_original,
                                    ..
                                },
                                AExpr::Function {
                                    input: input_new, ..
                                },
                            ] => {
                                for (new, original) in input_new.iter_mut().zip(input_original) {
                                    new.set_alias(original.output_name().clone());
                                }
                            },
                            [
                                AExpr::AnonymousFunction {
                                    input: input_original,
                                    ..
                                },
                                AExpr::AnonymousFunction {
                                    input: input_new, ..
                                },
                            ] => {
                                for (new, original) in input_new.iter_mut().zip(input_original) {
                                    new.set_alias(original.output_name().clone());
                                }
                            },
                            _ => {},
                        }
                    }

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
                let (node, _count) = self.se_count.get(&id).unwrap();

                // Avoid accidentally broadcasting <scalar literal>.<elementwise_ops..>
                if self.element_wise_select_only
                    && !matches!(
                        aexpr_projection_height_rec(
                            *node,
                            expr_arena,
                            &mut self.nodes_scratch,
                            &mut self.heights_scratch
                        ),
                        ExprProjectionHeight::Column
                    )
                {
                    return Ok(None);
                }

                let out_e = ExprIR::new(*node, OutputName::Alias(cse_column_name(id)));
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
                if let Some(expr) = self.find_cse(
                    expr,
                    &mut arena.1,
                    false,
                    input_schema.as_ref().as_ref(),
                    self.element_wise_select_only,
                )? {
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
                if let Some(exprs) = self.find_cse(
                    exprs,
                    &mut arena.1,
                    false,
                    input_schema.as_ref().as_ref(),
                    self.element_wise_select_only,
                )? {
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
                if let Some(aggs) = self.find_cse(
                    aggs,
                    &mut arena.1,
                    true,
                    input_schema.as_ref().as_ref(),
                    self.element_wise_select_only,
                )? {
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
