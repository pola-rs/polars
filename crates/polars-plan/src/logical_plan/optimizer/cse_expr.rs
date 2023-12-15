use polars_utils::vec::CapacityByFactor;

use super::*;
use crate::constants::CSE_REPLACED;
use crate::logical_plan::projection_expr::ProjectionExprs;
use crate::logical_plan::visitor::{RewriteRecursion, VisitRecursion};
use crate::prelude::visitor::{ALogicalPlanNode, AexprNode, RewritingVisitor, TreeWalker, Visitor};

// We use hashes to get an Identifier
// but this is very hard to debug, so we also have a version that
// uses a string trail.
#[cfg(test)]
mod identifier_impl {
    use std::hash::{Hash, Hasher};

    use super::*;
    /// Identifier that shows the sub-expression path.
    /// Must implement hash and equality and ideally
    /// have little collisions
    /// We will do a full expression comparison to check if the
    /// expressions with equal identifiers are truly equal
    #[derive(Clone, Debug)]
    pub struct Identifier {
        inner: String,
        last_node: Option<AexprNode>,
    }

    impl PartialEq for Identifier {
        fn eq(&self, other: &Self) -> bool {
            self.inner == other.inner && self.last_node == other.last_node
        }
    }

    impl Eq for Identifier {}

    impl Hash for Identifier {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.inner.hash(state)
        }
    }

    impl Identifier {
        pub fn new() -> Self {
            Self {
                inner: String::new(),
                last_node: None,
            }
        }

        pub fn ae_node(&self) -> AexprNode {
            self.last_node.unwrap()
        }

        pub fn is_valid(&self) -> bool {
            !self.inner.is_empty()
        }

        pub fn materialize(&self) -> String {
            format!("{}{}", CSE_REPLACED, self.inner)
        }

        pub fn combine(&mut self, other: &Identifier) {
            self.inner.push('!');
            self.inner.push_str(&other.inner);
        }

        pub fn add_ae_node(&self, ae: &AexprNode) -> Self {
            let inner = format!("{:E}{}", ae.to_aexpr(), self.inner);
            Self {
                inner,
                last_node: Some(*ae),
            }
        }
    }
}

#[cfg(not(test))]
mod identifier_impl {
    use std::hash::{Hash, Hasher};

    use ahash::RandomState;
    use polars_core::hashing::_boost_hash_combine;

    use super::*;
    /// Identifier that shows the sub-expression path.
    /// Must implement hash and equality and ideally
    /// have little collisions
    /// We will do a full expression comparison to check if the
    /// expressions with equal identifiers are truly equal
    #[derive(Clone, Debug)]
    pub struct Identifier {
        inner: Option<u64>,
        last_node: Option<AexprNode>,
        hb: RandomState,
    }

    impl PartialEq<Self> for Identifier {
        fn eq(&self, other: &Self) -> bool {
            self.inner == other.inner && self.last_node == other.last_node
        }
    }

    impl Eq for Identifier {}

    impl Hash for Identifier {
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_u64(self.inner.unwrap_or(0))
        }
    }

    impl Identifier {
        pub fn new() -> Self {
            Self {
                inner: None,
                last_node: None,
                hb: RandomState::with_seed(0),
            }
        }

        pub fn ae_node(&self) -> AexprNode {
            self.last_node.unwrap()
        }

        pub fn is_valid(&self) -> bool {
            self.inner.is_some()
        }

        pub fn materialize(&self) -> String {
            format!("{}{}", CSE_REPLACED, self.inner.unwrap_or(0))
        }

        pub fn combine(&mut self, other: &Identifier) {
            let inner = match (self.inner, other.inner) {
                (Some(l), Some(r)) => _boost_hash_combine(l, r),
                (None, Some(r)) => r,
                (Some(l), None) => l,
                _ => return,
            };
            self.inner = Some(inner);
        }

        pub fn add_ae_node(&self, ae: &AexprNode) -> Self {
            let hashed = self.hb.hash_one(ae.to_aexpr());
            let inner = Some(
                self.inner
                    .map_or(hashed, |l| _boost_hash_combine(l, hashed)),
            );
            Self {
                inner,
                last_node: Some(*ae),
                hb: self.hb.clone(),
            }
        }
    }
}
use identifier_impl::*;

/// Identifier maps to Expr Node and count.
type SubExprCount = PlHashMap<Identifier, (Node, usize)>;
/// (post_visit_idx, identifier);
type IdentifierArray = Vec<(usize, Identifier)>;

#[derive(Debug)]
enum VisitRecord {
    /// entered a new expression
    Entered(usize),
    /// Every visited sub-expression pushes their identifier to the stack.
    // The `bool` indicates if this expression is valid.
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
    SubExprId(Identifier, bool),
}

fn skip_pre_visit(ae: &AExpr, is_groupby: bool) -> bool {
    match ae {
        AExpr::Window { .. } => true,
        AExpr::Ternary { .. } => is_groupby,
        _ => false,
    }
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
    is_group_by: bool,
}

type Accepted = Option<(VisitRecursion, bool)>;

impl ExprIdentifierVisitor<'_> {
    fn new<'a>(
        se_count: &'a mut SubExprCount,
        identifier_array: &'a mut IdentifierArray,
        visit_stack: &'a mut Vec<VisitRecord>,
        is_group_by: bool,
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
            is_group_by,
        }
    }

    /// pop all visit-records until an `Entered` is found. We accumulate a `SubExprId`s
    /// to `id`. Finally we return the expression `idx` and `Identifier`.
    /// This works due to the stack.
    /// If we traverse another expression in the mean time, it will get popped of the stack first
    /// so the returned identifier belongs to a single sub-expression
    fn pop_until_entered(&mut self) -> (usize, Identifier, bool) {
        let mut id = Identifier::new();
        let mut is_valid_accumulated = true;

        while let Some(item) = self.visit_stack.pop() {
            match item {
                VisitRecord::Entered(idx) => return (idx, id, is_valid_accumulated),
                VisitRecord::SubExprId(s, valid) => {
                    id.combine(&s);
                    is_valid_accumulated &= valid
                },
            }
        }
        unreachable!()
    }

    /// return `None` -> node is accepted
    /// return `Some(_)` node is not accepted and apply the given recursion operation
    /// `Some(_, true)` don't accept this node, but can be a member of a cse.
    /// `Some(_,  false)` don't accept this node, and don't allow as a member of a cse.
    fn accept_node_post_visit(&self, ae: &AExpr) -> Accepted {
        // Don't allow this node in a cse.
        const REFUSE_NO_MEMBER: Accepted = Some((VisitRecursion::Continue, false));
        // Don't allow this node, but allow as a member of a cse.
        const REFUSE_ALLOW_MEMBER: Accepted = Some((VisitRecursion::Continue, true));
        const REFUSE_SKIP: Accepted = Some((VisitRecursion::Skip, false));
        // Accept this node.
        const ACCEPT: Accepted = None;

        match ae {
            // window expressions should `evaluate_on_groups`, not `evaluate`
            // so we shouldn't cache the children as they are evaluated incorrectly
            AExpr::Window { .. } => REFUSE_SKIP,
            // Don't allow this for now, as we can get `null().cast()` in ternary expressions.
            // TODO! Add a typed null
            AExpr::Literal(LiteralValue::Null) => REFUSE_NO_MEMBER,
            AExpr::Column(_) | AExpr::Literal(_) | AExpr::Alias(_, _) => REFUSE_ALLOW_MEMBER,
            AExpr::Count => {
                if self.is_group_by {
                    REFUSE_NO_MEMBER
                } else {
                    REFUSE_ALLOW_MEMBER
                }
            },
            #[cfg(feature = "random")]
            AExpr::Function {
                function: FunctionExpr::Random { .. },
                ..
            } => REFUSE_NO_MEMBER,
            #[cfg(feature = "rolling_window")]
            AExpr::Function {
                function: FunctionExpr::RollingExpr { .. },
                ..
            } => REFUSE_NO_MEMBER,
            AExpr::AnonymousFunction { .. } => REFUSE_NO_MEMBER,
            _ => {
                // During aggregation we only store elementwise operation in the state
                // other operations we cannot add to the state as they have the output size of the
                // groups, not the original dataframe
                if self.is_group_by {
                    if ae.groups_sensitive() {
                        return REFUSE_NO_MEMBER;
                    }
                    match ae {
                        AExpr::AnonymousFunction { .. } | AExpr::Filter { .. } => REFUSE_NO_MEMBER,
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

    fn pre_visit(&mut self, node: &Self::Node) -> PolarsResult<VisitRecursion> {
        if skip_pre_visit(node.to_aexpr(), self.is_group_by) {
            return Ok(VisitRecursion::Skip);
        }

        self.visit_stack
            .push(VisitRecord::Entered(self.pre_visit_idx));
        self.pre_visit_idx += 1;

        // implement default placeholders
        self.identifier_array
            .push((self.id_array_offset, Identifier::new()));

        Ok(VisitRecursion::Continue)
    }

    fn post_visit(&mut self, node: &Self::Node) -> PolarsResult<VisitRecursion> {
        let ae = node.to_aexpr();
        self.post_visit_idx += 1;

        let (pre_visit_idx, sub_expr_id, is_valid_accumulated) = self.pop_until_entered();
        // create the id of this node
        let id: Identifier = sub_expr_id.add_ae_node(node);

        if !is_valid_accumulated {
            self.identifier_array[pre_visit_idx + self.id_array_offset].0 = self.post_visit_idx;
            self.visit_stack.push(VisitRecord::SubExprId(id, false));
            return Ok(VisitRecursion::Continue);
        }

        // if we don't store this node
        // we only push the visit_stack, so the parents know the trail
        if let Some((recurse, local_is_valid)) = self.accept_node_post_visit(ae) {
            self.identifier_array[pre_visit_idx + self.id_array_offset].0 = self.post_visit_idx;

            self.visit_stack
                .push(VisitRecord::SubExprId(id, local_is_valid));
            return Ok(recurse);
        }

        // store the created id
        self.identifier_array[pre_visit_idx + self.id_array_offset] =
            (self.post_visit_idx, id.clone());

        // We popped until entered, push this Id on the stack so the trail
        // is available for the parent expression
        self.visit_stack
            .push(VisitRecord::SubExprId(id.clone(), true));

        let (_, se_count) = self.se_count.entry(id).or_insert_with(|| (node.node(), 0));

        *se_count += 1;
        self.has_sub_expr |= *se_count > 1;

        Ok(VisitRecursion::Continue)
    }
}

struct CommonSubExprRewriter<'a> {
    sub_expr_map: &'a SubExprCount,
    identifier_array: &'a IdentifierArray,
    /// keep track of the replaced identifiers.
    replaced_identifiers: &'a mut PlHashSet<Identifier>,

    max_post_visit_idx: usize,
    /// index in traversal order in which `identifier_array`
    /// was written. This is the index in `identifier_array`.
    visited_idx: usize,
    /// Offset in the identifier array.
    /// This allows us to use a single `vec` on multiple expressions
    id_array_offset: usize,
    /// Indicates if this expression is rewritten.
    rewritten: bool,
    is_group_by: bool,
}

impl<'a> CommonSubExprRewriter<'a> {
    fn new(
        sub_expr_map: &'a SubExprCount,
        identifier_array: &'a IdentifierArray,
        replaced_identifiers: &'a mut PlHashSet<Identifier>,
        id_array_offset: usize,
        is_group_by: bool,
    ) -> Self {
        Self {
            sub_expr_map,
            identifier_array,
            replaced_identifiers,
            max_post_visit_idx: 0,
            visited_idx: 0,
            id_array_offset,
            rewritten: false,
            is_group_by,
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
        let ae = ae_node.to_aexpr();
        if self.visited_idx + self.id_array_offset >= self.identifier_array.len()
            || self.max_post_visit_idx
                > self.identifier_array[self.visited_idx + self.id_array_offset].0
            || skip_pre_visit(ae, self.is_group_by)
        {
            return Ok(RewriteRecursion::Stop);
        }

        let id = &self.identifier_array[self.visited_idx + self.id_array_offset].1;

        // placeholder not overwritten, so we can skip this sub-expression
        if !id.is_valid() {
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

        // Because some expressions don't have hash / equality guarantee (e.g. floats)
        // we can get none here. This must be changed later.
        let Some((_, count)) = self.sub_expr_map.get(id) else {
            self.visited_idx += 1;
            return Ok(RewriteRecursion::NoMutateAndContinue);
        };
        if *count > 1 {
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
        // If this is not true, the traversal order in the visitor was different from the rewriter.
        debug_assert!(node.binary(id.ae_node().node(), |l, r| l == r));

        let name = id.materialize();
        node.assign(AExpr::col(name.as_ref()));
        self.rewritten = true;

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
        is_group_by: bool,
    ) -> PolarsResult<(usize, bool)> {
        let mut visitor = ExprIdentifierVisitor::new(
            &mut self.se_count,
            &mut self.id_array,
            &mut self.visit_stack,
            is_group_by,
        );
        ae_node.visit(&mut visitor).map(|_| ())?;
        Ok((visitor.id_array_offset, visitor.has_sub_expr))
    }

    /// Mutate the expression.
    /// Returns a new expression and a `bool` indicating if it was rewritten or not.
    fn mutate_expression(
        &mut self,
        ae_node: AexprNode,
        id_array_offset: usize,
        is_group_by: bool,
    ) -> PolarsResult<(AexprNode, bool)> {
        let mut rewriter = CommonSubExprRewriter::new(
            &self.se_count,
            &self.id_array,
            &mut self.replaced_identifiers,
            id_array_offset,
            is_group_by,
        );
        ae_node
            .rewrite(&mut rewriter)
            .map(|out| (out, rewriter.rewritten))
    }

    fn find_cse(
        &mut self,
        expr: &[Node],
        expr_arena: &mut Arena<AExpr>,
        id_array_offsets: &mut Vec<u32>,
        is_group_by: bool,
        schema: &Schema,
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
                    self.visit_expression(ae_node, is_group_by)
                })?;
            id_array_offsets.push(id_array_offset as u32);
            has_sub_expr |= this_expr_has_se;
        }

        if has_sub_expr {
            let mut new_expr = Vec::with_capacity_by_factor(expr.len(), 1.3);

            // then rewrite the expressions that have a cse count > 1
            for (node, offset) in expr.iter().zip(id_array_offsets.iter()) {
                let new_node =
                    AexprNode::with_context_and_arena(*node, expr_arena, |ae_node, expr_arena| {
                        let (out, rewritten) =
                            self.mutate_expression(ae_node, *offset as usize, is_group_by)?;

                        let mut out_node = out.node();
                        if !rewritten {
                            return Ok(out_node);
                        }

                        let ae = expr_arena.get(out_node);
                        // If we don't end with an alias we add an alias. Because the normal left-hand
                        // rule we apply for determining the name will not work we now refer to
                        // intermediate temporary names starting with the `CSE_REPLACED` constant.
                        if !matches!(ae, AExpr::Alias(_, _)) {
                            let name = ae_node.to_field(schema)?.name;
                            out_node =
                                expr_arena.add(AExpr::Alias(out_node, Arc::from(name.as_str())))
                        }

                        PolarsResult::Ok(out_node)
                    })?;
                new_expr.push(new_node)
            }
            // Add the tmp columns
            for id in &self.replaced_identifiers {
                let (node, _count) = self.se_count.get(id).unwrap();
                let name = id.materialize();
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
            },
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

        let arena_idx = node.node();
        node.with_arena_mut(|lp_arena| {
            let alp = lp_arena.get(arena_idx);

            match alp {
                ALogicalPlan::Projection {
                    input,
                    expr,
                    schema,
                    options,
                } => {
                    let input_schema = lp_arena.get(*input).schema(lp_arena);
                    if let Some(expr) = self.find_cse(
                        expr,
                        &mut expr_arena,
                        &mut id_array_offsets,
                        false,
                        input_schema.as_ref().as_ref(),
                    )? {
                        let lp = ALogicalPlan::Projection {
                            input: *input,
                            expr,
                            schema: schema.clone(),
                            options: *options,
                        };
                        lp_arena.replace(arena_idx, lp);
                    }
                },
                ALogicalPlan::HStack {
                    input,
                    exprs,
                    schema,
                    options,
                } => {
                    let input_schema = lp_arena.get(*input).schema(lp_arena);
                    if let Some(exprs) = self.find_cse(
                        exprs,
                        &mut expr_arena,
                        &mut id_array_offsets,
                        false,
                        input_schema.as_ref().as_ref(),
                    )? {
                        let lp = ALogicalPlan::HStack {
                            input: *input,
                            exprs,
                            schema: schema.clone(),
                            options: *options,
                        };
                        lp_arena.replace(arena_idx, lp);
                    }
                },
                ALogicalPlan::Aggregate {
                    input,
                    keys,
                    aggs,
                    options,
                    maintain_order,
                    apply,
                    schema,
                } => {
                    let input_schema = lp_arena.get(*input).schema(lp_arena);
                    if let Some(aggs) = self.find_cse(
                        aggs,
                        &mut expr_arena,
                        &mut id_array_offsets,
                        true,
                        input_schema.as_ref().as_ref(),
                    )? {
                        let keys = keys.clone();
                        let options = options.clone();
                        let schema = schema.clone();
                        let apply = apply.clone();
                        let maintain_order = *maintain_order;
                        let input = *input;

                        let lp = ALogicalPlanBuilder::new(input, &mut expr_arena, lp_arena)
                            .with_columns(aggs.cse_exprs().to_vec(), Default::default())
                            .build();
                        let input = lp_arena.add(lp);

                        let lp = ALogicalPlan::Aggregate {
                            input,
                            keys,
                            aggs: aggs.default_exprs().to_vec(),
                            options,
                            schema,
                            maintain_order,
                            apply,
                        };
                        lp_arena.replace(arena_idx, lp);
                    }
                },
                _ => {},
            }
            PolarsResult::Ok(())
        })?;

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
        let e = (col("foo").sum() * col("bar")).sum() + col("foo").sum();

        let mut arena = Arena::new();
        let node = to_aexpr(e, &mut arena);

        let mut se_count = Default::default();

        // Pre-fill `id_array` with a value to also check if we deal with the offset correct;
        let mut id_array = vec![(0, Identifier::new()); 1];
        let id_array_offset = id_array.len();
        let mut visit_stack = vec![];
        let mut visitor =
            ExprIdentifierVisitor::new(&mut se_count, &mut id_array, &mut visit_stack, false);

        AexprNode::with_context(node, &mut arena, |ae_node| ae_node.visit(&mut visitor)).unwrap();

        let mut replaced_ids = Default::default();
        let mut rewriter = CommonSubExprRewriter::new(
            &se_count,
            &id_array,
            &mut replaced_ids,
            id_array_offset,
            false,
        );
        let ae_node =
            AexprNode::with_context(node, &mut arena, |ae_node| ae_node.rewrite(&mut rewriter))
                .unwrap();

        let e = node_to_expr(ae_node.node(), &arena);
        assert_eq!(
            format!("{}", e),
            r#"[([(col("__POLARS_CSER_sum!col(foo)")) * (col("bar"))].sum()) + (col("__POLARS_CSER_sum!col(foo)"))]"#
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
            .project(
                vec![e.clone() * col("b"), e.clone() * col("b") + e, col("b")],
                Default::default(),
            )
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
            r#"col("__POLARS_CSER_binary: *!sum!col(a)!col(b)").alias("a")"#
        );
        assert_eq!(
            format!("{}", node_to_expr(default[1], &expr_arena)),
            r#"[(col("__POLARS_CSER_binary: *!sum!col(a)!col(b)")) + (col("__POLARS_CSER_sum!col(a)"))].alias("a")"#
        );
        assert_eq!(
            format!("{}", node_to_expr(default[2], &expr_arena)),
            r#"col("b")"#
        );

        let cse = expr.cse_exprs();
        assert_eq!(cse.len(), 2);

        // Hashmap can change the order of the cse's.
        let mut cse = cse
            .iter()
            .map(|node| format!("{}", node_to_expr(*node, &expr_arena)))
            .collect::<Vec<_>>();
        cse.sort();
        assert_eq!(
            cse[0],
            r#"[(col("a").sum()) * (col("b"))].alias("__POLARS_CSER_binary: *!sum!col(a)!col(b)")"#
        );
        assert_eq!(
            cse[1],
            r#"col("a").sum().alias("__POLARS_CSER_sum!col(a)")"#
        );
    }
}
