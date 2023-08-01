use std::rc::Rc;

use polars_utils::vec::CapacityByFactor;

use super::*;
use crate::constants::CSE_REPLACED;
use crate::logical_plan::projection_expr::ProjectionExprs;
use crate::logical_plan::visitor::{RewriteRecursion, VisitRecursion};
use crate::prelude::visitor::{ALogicalPlanNode, AexprNode, RewritingVisitor, TreeWalker, Visitor};

/// Identifier that shows the sub-expression path.
/// Must implement hash and equality and ideally
/// have little collisions
/// We will do a full expression comparison to check if the
/// expressions with equal identifiers are truly equal
// TODO! try to use a hash `usize` for this?
type Identifier = Rc<str>;
/// Identifier maps to Expr Node and count.
type SubExprCount = PlHashMap<Identifier, (Node, usize)>;
/// (post_visit_idx, identifier);
type IdentifierArray = Vec<(usize, Identifier)>;

fn replace_name(id: &str) -> String {
    format!("{}{}", CSE_REPLACED, id)
}

#[derive(Debug)]
enum VisitRecord {
    /// entered a new expression
    Entered(usize),
    /// every visited sub-expression pushes their identifier to the stack
    SubExprId(Identifier),
}

/// Goes through an expression and generates a identifier
///
/// The visitor uses a `visit_stack` to track traversal order.
///
/// # Entering a node
/// When `pre-visit` is called we enter a new (sub)-expression and
/// we add `Entered` to the stack.
/// # Leaving a node
/// On `post-visit` when we leave the node and we pop all `SubExprIds` nodes.
/// Those are considered sub-expression of the leaving node
///
/// We also record an `id_array` that followed the pre-visit order. This
/// is used to cache the `Identifiers`.
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
// function-called              stack                stack-after(pop until E, push I)   # ID
// pre-visit: sum                E                        -
// pre-visit: binary: *          EE                       -
// pre-visit: col(bar)           EEE                      -
// post-visit: col(bar)	         EEE                      EEI                          id: col(bar)
// pre-visit: min                EEIE                     -
// pre-visit: col(f00)           EEIEE                    -
// post-visit: col(f00)	         EEIEE                    EEIEI                        id: col(f00)
// post-visit: min	             EEIEI                    EEII                         id: min!col(f00)
// post-visit: binary: *         EEII                     EI                           id: binary: *!min!col(f00)!col(bar)
// post-visit: sum               EI                       I                            id: sum!binary: *!min!col(f00)!col(bar)
struct ExprIdentifierVisitor<'a> {
    se_count: &'a mut SubExprCount,
    identifier_array: &'a mut IdentifierArray,
    // index in pre-visit traversal order
    pre_visit_idx: usize,
    post_visit_idx: usize,
    visit_stack: &'a mut Vec<VisitRecord>,
    /// Offset in the identifier array
    /// this allows us to use a single `vec` on multiple expressions
    id_array_offset: usize,
    // whether the expression replaced a subexpression
    has_sub_expr: bool,
    // During aggregation we only identify element-wise operations
    is_groupby: bool,
}

impl ExprIdentifierVisitor<'_> {
    fn new<'a>(
        se_count: &'a mut SubExprCount,
        identifier_array: &'a mut IdentifierArray,
        visit_stack: &'a mut Vec<VisitRecord>,
        is_groupby: bool,
    ) -> ExprIdentifierVisitor<'a> {
        let id_array_offset = identifier_array.len();
        ExprIdentifierVisitor {
            se_count,
            identifier_array,
            pre_visit_idx: 0,
            post_visit_idx: 0,
            visit_stack,
            id_array_offset,
            has_sub_expr: false,
            is_groupby,
        }
    }

    /// pop all visit-records until an `Entered` is found. We accumulate a `SubExprId`s
    /// to `id`. Finally we return the expression `idx` and `Identifier`.
    /// This works due to the stack.
    /// If we traverse another expression in the mean time, it will get popped of the stack first
    /// so the returned identifier belongs to a single sub-expression
    fn pop_until_entered(&mut self) -> (usize, Identifier) {
        let mut id = String::new();

        while let Some(item) = self.visit_stack.pop() {
            match item {
                VisitRecord::Entered(idx) => return (idx, Rc::from(id)),
                VisitRecord::SubExprId(s) => {
                    id.push('!');
                    id.push_str(s.as_ref());
                }
            }
        }
        unreachable!()
    }

    /// return `None` -> node is accepted
    /// return `Some(_)` node is not accepted and apply the given recursion operation
    fn accept_node(&self, ae: &AExpr) -> Option<VisitRecursion> {
        match ae {
            // window expressions should `evaluate_on_groups`, not `evaluate`
            // so we shouldn't cache the children as they are evaluated incorrectly
            AExpr::Window { .. } => Some(VisitRecursion::Skip),
            // skip window functions for now until we properly implemented the physical side
            AExpr::Column(_) | AExpr::Count | AExpr::Literal(_) | AExpr::Alias(_, _) => {
                Some(VisitRecursion::Continue)
            }
            #[cfg(feature = "random")]
            AExpr::Function {
                function: FunctionExpr::Random { .. },
                ..
            } => Some(VisitRecursion::Continue),
            _ => {
                // during aggregation we only store elementwise operation in the state
                // other operations we cannot add to the state as they have the output size of the
                // groups, not the original dataframe
                if self.is_groupby {
                    match ae {
                        AExpr::Agg(_) | AExpr::AnonymousFunction { .. } => {
                            Some(VisitRecursion::Continue)
                        }
                        AExpr::Function { options, .. } => {
                            if options.is_groups_sensitive() {
                                Some(VisitRecursion::Continue)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
        }
    }
}

impl Visitor for ExprIdentifierVisitor<'_> {
    type Node = AexprNode;

    fn pre_visit(&mut self, _node: &Self::Node) -> PolarsResult<VisitRecursion> {
        self.visit_stack
            .push(VisitRecord::Entered(self.pre_visit_idx));
        self.pre_visit_idx += 1;

        // implement default placeholders
        self.identifier_array
            .push((self.id_array_offset, "".into()));

        Ok(VisitRecursion::Continue)
    }

    fn post_visit(&mut self, node: &Self::Node) -> PolarsResult<VisitRecursion> {
        let ae = node.to_aexpr();
        self.post_visit_idx += 1;

        let (pre_visit_idx, sub_expr_id) = self.pop_until_entered();

        // if we don't store this node
        // we only push the visit_stack, so the parents know the trail
        if let Some(recurse) = self.accept_node(ae) {
            self.identifier_array[pre_visit_idx + self.id_array_offset].0 = self.post_visit_idx;
            self.visit_stack
                .push(VisitRecord::SubExprId(Rc::from(format!("{:E}", ae))));
            return Ok(recurse);
        }

        // create the id of this node
        let id: Identifier = Rc::from(format!("{:E}{}", ae, sub_expr_id));

        // store the created id
        self.identifier_array[pre_visit_idx + self.id_array_offset] =
            (self.post_visit_idx, id.clone());

        // We popped until entered, push this Id on the stack so the trail
        // is available for the parent expression
        self.visit_stack.push(VisitRecord::SubExprId(id.clone()));

        let (_, se_count) = self.se_count.entry(id).or_insert_with(|| (node.node(), 0));

        *se_count += 1;
        self.has_sub_expr |= *se_count > 1;

        Ok(VisitRecursion::Continue)
    }
}

struct CommonSubExprRewriter<'a> {
    sub_expr_map: &'a SubExprCount,
    identifier_array: &'a IdentifierArray,
    /// keep track of the replaced identifiers
    replaced_identifiers: &'a mut PlHashSet<Identifier>,

    max_post_visit_idx: usize,
    /// index in traversal order in which `identifier_array`
    /// was written. This is the index in `identifier_array`.
    visited_idx: usize,
    /// Offset in the identifier array
    /// this allows us to use a single `vec` on multiple expressions
    id_array_offset: usize,
}

impl<'a> CommonSubExprRewriter<'a> {
    fn new(
        sub_expr_map: &'a SubExprCount,
        identifier_array: &'a IdentifierArray,
        replaced_identifiers: &'a mut PlHashSet<Identifier>,
        id_array_offset: usize,
    ) -> Self {
        Self {
            sub_expr_map,
            identifier_array,
            replaced_identifiers,
            max_post_visit_idx: 0,
            visited_idx: 0,
            id_array_offset,
        }
    }
}

// # Example
// Expression tree with [pre-visit,post-visit] indices
// counted from 1
//     [1,8] binary: +
//
//       |                            |
//
//     [2,2] sum                    [4,7] sum
//
//       |                            |
//
//     [3,1] col(foo)               [5,6] binary: *
//
//                                    |                       |
//
//                                   [6,3] col(bar)        [7,5] sum
//
//                                                            |
//
//                                                         [8,4] col(foo)
//
// in this tree `col(foo).sum()` should be post-visited/mutated
// so if we are at `[2,2]`
//
// call stack
// pre-visit    [1,8] binary    -> no_mutate_and_continue -> visits children
// pre-visit    [2,2] sum       -> mutate_and_stop -> does not visit children
// post-visit   [2,2] sum       -> skip index to [4,7] (because we didn't visit children)
// pre-visit    [4,7] sum       -> no_mutate_and_continue -> visits children
// pre-visit    [5,6] binary    -> no_mutate_and_continue -> visits children
// pre-visit    [6,3] col       -> stop_recursion -> does not mutate
// pre-visit    [7,5] sum       -> mutate_and_stop -> does not visit children
// post-visit   [7,5]           -> skip index to end
impl RewritingVisitor for CommonSubExprRewriter<'_> {
    type Node = AexprNode;

    fn pre_visit(&mut self, ae_node: &Self::Node) -> PolarsResult<RewriteRecursion> {
        if self.visited_idx + self.id_array_offset >= self.identifier_array.len()
            || self.max_post_visit_idx
                > self.identifier_array[self.visited_idx + self.id_array_offset].0
        {
            return Ok(RewriteRecursion::Stop);
        }

        // check if we can accept node
        // we don't traverse those children
        if matches!(ae_node.to_aexpr(), AExpr::Window { .. }) {
            return Ok(RewriteRecursion::Stop);
        }

        let id = &self.identifier_array[self.visited_idx + self.id_array_offset].1;

        // placeholder not overwritten, so we can skip this sub-expression
        if id.is_empty() {
            self.visited_idx += 1;
            let recurse = if ae_node.is_leaf() {
                RewriteRecursion::Stop
            } else {
                // continue visit its children to see
                // if there are cse
                RewriteRecursion::NoMutateAndContinue
            };
            return Ok(recurse);
        }

        let (node, count) = self.sub_expr_map.get(id).unwrap();
        if *count > 1
            // this does a full expression traversal to check if the expression is truly
            // the same
            && ae_node.binary(*node, |l, r| l == r)
        {
            self.replaced_identifiers.insert(id.clone());
            // rewrite this sub-expression, don't visit its children
            Ok(RewriteRecursion::MutateAndStop)
        } else {
            // This is a unique expression
            // visit its children to see if they are cse
            self.visited_idx += 1;
            Ok(RewriteRecursion::NoMutateAndContinue)
        }
    }

    fn mutate(&mut self, mut node: Self::Node) -> PolarsResult<Self::Node> {
        let (post_visit_count, id) =
            &self.identifier_array[self.visited_idx + self.id_array_offset];
        self.visited_idx += 1;

        // TODO!: check if we ever hit this branch
        if *post_visit_count < self.max_post_visit_idx {
            return Ok(node);
        }

        self.max_post_visit_idx = *post_visit_count;
        // DFS, so every post_visit that is smaller than `post_visit_count`
        // is a subexpression of this node and we can skip that
        //
        // `self.visited_idx` will influence recursion strategy in `pre_visit`
        // see call-stack comment above
        while self.visited_idx < self.identifier_array.len() - self.id_array_offset
            && *post_visit_count > self.identifier_array[self.visited_idx + self.id_array_offset].0
        {
            self.visited_idx += 1;
        }

        let name = replace_name(id.as_ref());
        node.assign(AExpr::col(name.as_ref()));

        Ok(node)
    }
}

pub(crate) struct CommonSubExprOptimizer<'a> {
    expr_arena: &'a mut Arena<AExpr>,
    // amortize allocations
    // these are cleared per lp node
    se_count: SubExprCount,
    id_array: IdentifierArray,
    id_array_offsets: Vec<u32>,
    replaced_identifiers: PlHashSet<Identifier>,
    // these are cleared per expr node
    visit_stack: Vec<VisitRecord>,
}

impl<'a> CommonSubExprOptimizer<'a> {
    pub(crate) fn new(expr_arena: &'a mut Arena<AExpr>) -> Self {
        Self {
            expr_arena,
            se_count: Default::default(),
            id_array: Default::default(),
            visit_stack: Default::default(),
            id_array_offsets: Default::default(),
            replaced_identifiers: Default::default(),
        }
    }

    fn visit_expression(
        &mut self,
        ae_node: AexprNode,
        is_groupby: bool,
    ) -> PolarsResult<(usize, bool)> {
        let mut visitor = ExprIdentifierVisitor::new(
            &mut self.se_count,
            &mut self.id_array,
            &mut self.visit_stack,
            is_groupby,
        );
        ae_node.visit(&mut visitor).map(|_| ())?;
        Ok((visitor.id_array_offset, visitor.has_sub_expr))
    }
    fn mutate_expression(
        &mut self,
        ae_node: AexprNode,
        id_array_offset: usize,
    ) -> PolarsResult<AexprNode> {
        let mut rewriter = CommonSubExprRewriter::new(
            &self.se_count,
            &self.id_array,
            &mut self.replaced_identifiers,
            id_array_offset,
        );
        ae_node.rewrite(&mut rewriter)
    }

    fn find_cse(
        &mut self,
        expr: &[Node],
        expr_arena: &mut Arena<AExpr>,
        id_array_offsets: &mut Vec<u32>,
        is_groupby: bool,
    ) -> PolarsResult<Option<ProjectionExprs>> {
        let mut has_sub_expr = false;

        // first get all cse's
        for node in expr {
            // the visitor can return early thus depleted its stack
            // on a previous iteration
            self.visit_stack.clear();

            // visit expressions and collect sub-expression counts
            let (id_array_offset, this_expr_has_se) =
                AexprNode::with_context(*node, expr_arena, |ae_node| {
                    self.visit_expression(ae_node, is_groupby)
                })?;
            id_array_offsets.push(id_array_offset as u32);
            has_sub_expr |= this_expr_has_se;
        }

        if has_sub_expr {
            let mut new_expr = Vec::with_capacity_by_factor(expr.len(), 1.3);

            // then rewrite the expressions that have a cse count > 1
            for (node, offset) in expr.iter().zip(id_array_offsets.iter()) {
                let new_node = AexprNode::with_context(*node, expr_arena, |ae_node| {
                    self.mutate_expression(ae_node, *offset as usize)
                })?;
                new_expr.push(new_node.node())
            }
            // Add the tmp columns
            for id in &self.replaced_identifiers {
                let (node, _count) = self.se_count.get(id).unwrap();
                let name = replace_name(id.as_ref());
                let ae = AExpr::Alias(*node, Arc::from(name));
                let node = expr_arena.add(ae);
                new_expr.push(node)
            }
            let expr = ProjectionExprs::new_with_cse(new_expr, self.replaced_identifiers.len());
            Ok(Some(expr))
        } else {
            Ok(None)
        }
    }
}

impl<'a> RewritingVisitor for CommonSubExprOptimizer<'a> {
    type Node = ALogicalPlanNode;

    fn pre_visit(&mut self, node: &Self::Node) -> PolarsResult<RewriteRecursion> {
        use ALogicalPlan::*;
        Ok(match node.to_alp() {
            Projection { .. } | HStack { .. } | Aggregate { .. } => {
                RewriteRecursion::MutateAndContinue
            }
            _ => RewriteRecursion::NoMutateAndContinue,
        })
    }

    fn mutate(&mut self, mut node: Self::Node) -> PolarsResult<Self::Node> {
        let mut expr_arena = Arena::new();
        std::mem::swap(self.expr_arena, &mut expr_arena);
        let mut id_array_offsets = std::mem::take(&mut self.id_array_offsets);

        self.se_count.clear();
        self.id_array.clear();
        id_array_offsets.clear();
        self.replaced_identifiers.clear();

        match node.to_alp() {
            ALogicalPlan::Projection {
                input,
                expr,
                schema,
            } => {
                if let Some(expr) =
                    self.find_cse(expr, &mut expr_arena, &mut id_array_offsets, false)?
                {
                    let lp = ALogicalPlan::Projection {
                        input: *input,
                        expr,
                        schema: schema.clone(),
                    };
                    node.replace(lp);
                }
            }
            ALogicalPlan::HStack {
                input,
                exprs,
                schema,
            } => {
                if let Some(exprs) =
                    self.find_cse(exprs, &mut expr_arena, &mut id_array_offsets, false)?
                {
                    let lp = ALogicalPlan::HStack {
                        input: *input,
                        exprs,
                        schema: schema.clone(),
                    };
                    node.replace(lp);
                }
            }
            // TODO! activate once fixed
            // ALogicalPlan::Aggregate {
            //     input,
            //     keys,
            //     aggs,
            //     options,
            //     maintain_order,
            //     apply,
            //     schema,
            // } => {
            //     if let Some(aggs) =
            //         self.find_cse(aggs, &mut expr_arena, &mut id_array_offsets, true)?
            //     {
            //         let keys = keys.clone();
            //         let options = options.clone();
            //         let schema = schema.clone();
            //         let apply = apply.clone();
            //         let maintain_order = *maintain_order;
            //         let input = *input;
            //
            //         let input = node.with_arena_mut(|lp_arena| {
            //             let lp = ALogicalPlanBuilder::new(input, &mut expr_arena, lp_arena)
            //                 .with_columns(aggs.cse_exprs().to_vec())
            //                 .build();
            //             lp_arena.add(lp)
            //         });
            //
            //         let lp = ALogicalPlan::Aggregate {
            //             input,
            //             keys,
            //             aggs: aggs.default_exprs().to_vec(),
            //             options,
            //             schema,
            //             maintain_order,
            //             apply,
            //         };
            //         node.replace(lp);
            //     }
            // }
            _ => {}
        };
        std::mem::swap(self.expr_arena, &mut expr_arena);
        self.id_array_offsets = id_array_offsets;
        Ok(node)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cse_replacer() {
        let e = (col("f00").sum() * col("bar")).sum() + col("f00").sum();

        let mut arena = Arena::new();
        let node = to_aexpr(e, &mut arena);

        let mut se_count = Default::default();

        // Pre-fill `id_array` with a value to also check if we deal with the offset correct;
        let mut id_array = vec![(0, Rc::from("")); 1];
        let id_array_offset = id_array.len();
        let mut visit_stack = vec![];
        let mut visitor =
            ExprIdentifierVisitor::new(&mut se_count, &mut id_array, &mut visit_stack, false);

        AexprNode::with_context(node, &mut arena, |ae_node| ae_node.visit(&mut visitor)).unwrap();

        let mut replaced_ids = Default::default();
        let mut rewriter =
            CommonSubExprRewriter::new(&se_count, &id_array, &mut replaced_ids, id_array_offset);
        let ae_node =
            AexprNode::with_context(node, &mut arena, |ae_node| ae_node.rewrite(&mut rewriter))
                .unwrap();

        let e = node_to_expr(ae_node.node(), &arena);
        assert_eq!(
            format!("{}", e),
            r#"[(col("__POLARS_CSER_sum!col(f00)")) + ([(col("bar")) * (col("__POLARS_CSER_sum!col(f00)"))].sum())]"#
        );
    }

    #[test]
    fn test_lp_cse_replacer() {
        let df = df![
            "a" => [1, 2, 3],
            "b" => [4, 5, 6],
        ]
        .unwrap();

        let e = col("a").sum();

        let lp = LogicalPlanBuilder::from_existing_df(df)
            .project(vec![
                e.clone() * col("b"),
                e.clone() * col("b") + e,
                col("b"),
            ])
            .build();

        let (node, mut lp_arena, mut expr_arena) = lp.to_alp().unwrap();
        let mut optimizer = CommonSubExprOptimizer::new(&mut expr_arena);

        let out = ALogicalPlanNode::with_context(node, &mut lp_arena, |alp_node| {
            alp_node.rewrite(&mut optimizer)
        })
        .unwrap();

        let ALogicalPlan::Projection { expr, .. } = out.to_alp() else {
            unreachable!()
        };

        let default = expr.default_exprs();
        assert_eq!(default.len(), 3);
        assert_eq!(
            format!("{}", node_to_expr(default[0], &expr_arena)),
            r#"[(col("b")) * (col("__POLARS_CSER_sum!col(a)"))]"#
        );
        assert_eq!(
            format!("{}", node_to_expr(default[1], &expr_arena)),
            r#"[(col("__POLARS_CSER_sum!col(a)")) + ([(col("b")) * (col("__POLARS_CSER_sum!col(a)"))])]"#
        );
        assert_eq!(
            format!("{}", node_to_expr(default[2], &expr_arena)),
            r#"col("b")"#
        );

        let cse = expr.cse_exprs();
        assert_eq!(cse.len(), 1);
        assert_eq!(
            format!("{}", node_to_expr(cse[0], &expr_arena)),
            r#"col("a").sum().alias("__POLARS_CSER_sum!col(a)")"#
        );
    }
}
