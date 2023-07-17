use std::rc::Rc;

use super::*;
use crate::constants::CSE_REPLACED;
use crate::logical_plan::projection_expr::ProjectionExprs;
use crate::logical_plan::visitor::{RewriteRecursion, VisitRecursion};
use crate::prelude::visitor::{AexprNode, ALogicalPlanNode, RewritingVisitor, Visitor, TreeWalker};

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
///
/// # Example
/// Say we have the expression: `(col("f00").min() * col("bar")).sum()`
/// with the following call tree:
///
///     sum
///
///       |
///
///     binary: *
///
///       |              |
///
///     col(bar)         min
///
///                      |
///
///                      col(f00)
///
/// # call order
/// function-called             stack                stack-after(pop until E, push I)   # ID
/// pre-visit: sum                E                        -
/// pre-visit: binary: *          EE                       -
/// pre-visit: col(bar)           EEE                      -
/// post-visit: col(bar)	      EEE                      EEI                          id: col(bar)
/// pre-visit: min                EEIE                     -
/// pre-visit: col(f00)           EEIEE                    -
/// post-visit: col(f00)	      EEIEE                    EEIEI                        id: col(f00)
/// post-visit: min	              EEIEI                    EEII                         id: min!col(f00)
/// post-visit: binary: *	      EEII                     EI                           id: binary: *!min!col(f00)!col(bar)
/// post-visit: sum               EI                       I                            id: sum!binary: *!min!col(f00)!col(bar)
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
    has_sub_expr: bool
}

impl ExprIdentifierVisitor<'_> {
    fn new<'a>(
        se_count: &'a mut SubExprCount,
        identifier_array: &'a mut IdentifierArray,
        visit_stack: &'a mut Vec<VisitRecord>,
    ) -> ExprIdentifierVisitor<'a> {
        let id_array_offset = identifier_array.len();
        ExprIdentifierVisitor {
            se_count,
            identifier_array,
            pre_visit_idx: 0,
            post_visit_idx: 0,
            visit_stack,
            id_array_offset,
            has_sub_expr: false
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

    fn accept_node(&self, ae: &AExpr) -> bool {
        !matches!(ae, AExpr::Column(_) | AExpr::Count | AExpr::Literal(_))
    }
}

impl Visitor for ExprIdentifierVisitor<'_> {
    type Node = AexprNode;

    fn pre_visit(&mut self, _node: &Self::Node) -> PolarsResult<VisitRecursion> {
        self.visit_stack
            .push(VisitRecord::Entered(self.pre_visit_idx));
        self.pre_visit_idx += 1;

        // implement default placeholders
        self.identifier_array.push((self.id_array_offset, "".into()));

        Ok(VisitRecursion::Continue)
    }

    fn post_visit(&mut self, node: &Self::Node) -> PolarsResult<VisitRecursion> {
        let ae = node.to_aexpr();
        self.post_visit_idx += 1;

        let (pre_visit_idx, sub_expr_id) = self.pop_until_entered();

        // if we don't store this node
        // we only push the visit_stack, so the parents know the trail
        if !self.accept_node(ae) {
            self.identifier_array[pre_visit_idx + self.id_array_offset].0 = self.post_visit_idx;
            self.visit_stack
                .push(VisitRecord::SubExprId(Rc::from(format!("{:E}", ae))));
            return Ok(VisitRecursion::Continue);
        }

        // create the id of this node
        let id: Identifier = Rc::from(format!("{:E}{}", ae, sub_expr_id));

        // store the created id
        self.identifier_array[pre_visit_idx + self.id_array_offset] = (self.post_visit_idx, id.clone());

        // We popped until entered, push this Id on the stack so the trail
        // is available for the parent expression
        self.visit_stack.push(VisitRecord::SubExprId(id.clone()));

        let (_, se_count) = self.se_count
            .entry(id)
            .or_insert_with(|| (node.node(), 0));

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

    max_series_number: usize,
    /// index in traversal order in which `identifier_array`
    /// was written. This is the index in `identifier_array`.
    visited_idx: usize,
    /// Offset in the identifier array
    /// this allows us to use a single `vec` on multiple expressions
    id_array_offset: usize
}

impl<'a> CommonSubExprRewriter<'a> {
    fn new(
        sub_expr_map: &'a SubExprCount,
        identifier_array: &'a IdentifierArray,
        replaced_identifiers: &'a mut PlHashSet<Identifier>,
        id_array_offset: usize
    ) -> Self {
        Self {
            sub_expr_map,
            identifier_array,
            replaced_identifiers,
            max_series_number: 0,
            visited_idx: 0,
            id_array_offset
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
        if self.visited_idx >= self.identifier_array.len() - self.id_array_offset {
            return Ok(RewriteRecursion::Stop);
        }

        let id = &self.identifier_array[self.visited_idx + self.id_array_offset].1;

        // placeholder not overwritten, so we can skip this sub-expression
        if id.is_empty() {
            self.visited_idx += 1;
            return Ok(RewriteRecursion::Stop);
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
        let (post_visit_count, id) = &self.identifier_array[self.visited_idx + self.id_array_offset];
        self.visited_idx += 1;

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

struct CommonSubExprOptimizer<'a> {
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
    fn new(
        expr_arena: &'a mut Arena<AExpr>
    ) -> Self {
        Self {
            expr_arena,
            se_count: Default::default(),
            id_array: Default::default(),
            visit_stack: Default::default(),
            id_array_offsets: Default::default(),
            replaced_identifiers: Default::default()
        }
    }

    fn visit_expression(&mut self, ae_node: AexprNode) -> PolarsResult<(usize, bool)> {
        let mut visitor = ExprIdentifierVisitor::new(&mut self.se_count, &mut self.id_array, &mut self.visit_stack);
        ae_node.visit(&mut visitor).map(|_| ())?;
        Ok((visitor.id_array_offset, visitor.has_sub_expr))
    }
    fn mutate_expression(&mut self, ae_node: AexprNode, id_array_offset: usize) -> PolarsResult<AexprNode>{
        let mut rewriter = CommonSubExprRewriter::new(&self.se_count, &self.id_array, &mut self.replaced_identifiers, id_array_offset);
        ae_node.rewrite(&mut rewriter)
    }
}

impl<'a> RewritingVisitor for CommonSubExprOptimizer<'a> {
    type Node = ALogicalPlanNode;

    fn mutate(&mut self, mut node: Self::Node) -> PolarsResult<Self::Node> {
        let mut expr_arena = Arena::new();
        std::mem::swap(self.expr_arena, &mut expr_arena);
        let mut id_array_offsets = std::mem::take(&mut self.id_array_offsets);

        self.se_count.clear();
        self.id_array.clear();
        id_array_offsets.clear();
        self.replaced_identifiers.clear();

        // todo! check if we can take mut
        let out = match node.to_alp() {
            ALogicalPlan::Projection {
                input,
                expr,
                schema
            } => {
                debug_assert!(!expr.has_sub_exprs());

                let mut has_sub_expr = false;

                // first get all cse's
                for node in expr {
                    let (id_array_offset, this_expr_has_se) = AexprNode::with_context(*node, &mut expr_arena, |ae_node| {
                        self.visit_expression(ae_node)
                    })?;
                    self.id_array_offsets.push(id_array_offset as u32);
                    has_sub_expr |= this_expr_has_se;
                    debug_assert!(self.visit_stack.is_empty());
                }


                if has_sub_expr {
                    let mut new_expr = Vec::with_capacity(expr.len() * 2);

                    // then rewrite the expressions that have a cse count > 1
                    for (node, offset) in expr.iter().zip(id_array_offsets.iter()) {
                        let new_node = AexprNode::with_context(*node, &mut expr_arena, |ae_node| {
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
                    let lp = ALogicalPlan::Projection {
                        input: *input,
                        expr,
                        schema: schema.clone()
                    };

                    node.assign(lp);
                }
                Ok(node)

            },
            _ => Ok(node)
        };
        std::mem::swap(self.expr_arena, &mut expr_arena);
        self.id_array_offsets = id_array_offsets;
        out
    }
}

// struct CommonSubExprElimination {
//     processed: PlHashSet<Node>,
//     sub_expr_map: SubExprMap,
// }
//
// fn process_expression(
//     node: Node,
//     expr_arena: &mut Arena<AExpr>,
//     sub_expr_map: &mut SubExprMap,
//     schema: &Schema,
// ) -> PolarsResult<IdentifierArray> {
//     let mut id_array = vec![];
//     let mut visitor = ExprIdentifierVisitor::new(sub_expr_map, &mut id_array, schema);
//     AexprNode::with_context(node, expr_arena, |ae_node| ae_node.visit(&mut visitor))?;
//
//     Ok(id_array)
// }
//
// impl CommonSubExprElimination {
//     pub fn new() -> Self {
//         Self {
//             processed: Default::default(),
//             sub_expr_map: Default::default(),
//         }
//     }
// }
//
// impl OptimizationRule for CommonSubExprElimination {
//     fn optimize_plan(
//         &mut self,
//         lp_arena: &mut Arena<ALogicalPlan>,
//         expr_arena: &mut Arena<AExpr>,
//         node: Node,
//     ) -> Option<ALogicalPlan> {
//         if !self.processed.insert(node) {
//             return None;
//         }
//
//         self.sub_expr_map.clear();
//         match lp_arena.get(node) {
//             ALogicalPlan::Projection {
//                 expr,
//                 input,
//                 schema,
//             } => {
//                 expr.into_iter().map(|node| {
//                     let array =
//                         process_expression(*node, expr_arena, &mut self.sub_expr_map, schema)
//                             .unwrap();
//                 });
//
//                 None
//             }
//             _ => None,
//         }
//     }
// }

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
        let mut visitor = ExprIdentifierVisitor::new(&mut se_count, &mut id_array, &mut visit_stack);

        AexprNode::with_context(node, &mut arena, |ae_node| ae_node.visit(&mut visitor)).unwrap();

        let mut replaced_ids = Default::default();
        let mut rewriter =
            CommonSubExprRewriter::new(&mut se_count, &mut id_array, &mut replaced_ids, id_array_offset);
        let ae_node =
            AexprNode::with_context(node, &mut arena, |ae_node| ae_node.rewrite(&mut rewriter))
                .unwrap();

        let e = node_to_expr(ae_node.node(), &arena);
        assert_eq!(
            format!("{}", e),
            r#"[(col("__POLARS_CSER_sum!col(f00)")) + ([(col("bar")) * (col("__POLARS_CSER_sum!col(f00)"))].sum())]"#
        );

        for k in replaced_ids {
            let (node, count) = se_count.get(&k).unwrap();
            dbg!(node_to_expr(*node, &arena), count);
        }
    }
}
