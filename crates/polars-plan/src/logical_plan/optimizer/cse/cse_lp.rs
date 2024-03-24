use super::*;
use crate::prelude::visitor::ALogicalPlanNode;

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
    #[derive(Clone)]
    pub(super) struct Identifier {
        inner: Option<u64>,
        last_node: Option<ALogicalPlanNode>,
        hb: RandomState,
        expr_arena: *const Arena<AExpr>,
    }

    impl PartialEq<Self> for Identifier {
        fn eq(&self, other: &Self) -> bool {
            self.inner == other.inner
                && match (self.last_node, other.last_node) {
                    (None, None) => true,
                    (Some(l), Some(r)) => {
                        let expr_arena = unsafe { &*self.expr_arena };
                        l.hashable_and_cmp(expr_arena) == r.hashable_and_cmp(expr_arena)
                    },
                    _ => false,
                }
        }
    }

    impl Eq for Identifier {}

    impl Hash for Identifier {
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_u64(self.inner.unwrap_or(0))
        }
    }

    impl Identifier {
        /// # Safety
        ///
        /// The arena must be a valid pointer and there should be no `&mut` to this arena.
        pub unsafe fn new(expr_arena: *const Arena<AExpr>) -> Self {
            Self {
                inner: None,
                last_node: None,
                hb: RandomState::with_seed(0),
                expr_arena,
            }
        }

        pub fn is_valid(&self) -> bool {
            self.inner.is_some()
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

        pub fn add_alp_node(&self, alp: &ALogicalPlanNode) -> Self {
            let expr_arena = unsafe { &*self.expr_arena };
            let hashed = self.hb.hash_one(alp.hashable_and_cmp(expr_arena));
            let inner = Some(
                self.inner
                    .map_or(hashed, |l| _boost_hash_combine(l, hashed)),
            );
            Self {
                inner,
                last_node: Some(*alp),
                hb: self.hb.clone(),
                expr_arena: self.expr_arena,
            }
        }
    }
}
use identifier_impl::*;

/// Identifier maps to Expr Node and count.
type SubPlanCount = PlHashMap<Identifier, (Node, usize)>;
/// (post_visit_idx, identifier);
type IdentifierArray = Vec<(usize, Identifier)>;

/// See Expr based CSE for explanations.
enum VisitRecord {
    /// Entered a new plan node
    Entered(usize),
    SubPlanId(Identifier, bool),
}

struct LpIdentifierVisitor<'a> {
    sp_count: &'a mut SubPlanCount,
    identifier_array: &'a mut IdentifierArray,
    // Index in pre-visit traversal order.
    pre_visit_idx: usize,
    post_visit_idx: usize,
    visit_stack: Vec<VisitRecord>,
    has_subplan: bool,
    expr_arena: &'a Arena<AExpr>,
}

impl LpIdentifierVisitor<'_> {
    fn new<'a>(
        sp_count: &'a mut SubPlanCount,
        identifier_array: &'a mut IdentifierArray,
        expr_arena: &'a Arena<AExpr>,
    ) -> LpIdentifierVisitor<'a> {
        LpIdentifierVisitor {
            sp_count,
            identifier_array,
            pre_visit_idx: 0,
            post_visit_idx: 0,
            visit_stack: vec![],
            has_subplan: false,
            expr_arena,
        }
    }

    fn pop_until_entered(&mut self) -> (usize, Identifier, bool) {
        // SAFETY:
        // we keep pointer valid and will not create mutable refs.
        let mut id = unsafe { Identifier::new(self.expr_arena as *const _) };
        let mut is_valid_accumulated = true;

        while let Some(item) = self.visit_stack.pop() {
            match item {
                VisitRecord::Entered(idx) => return (idx, id, is_valid_accumulated),
                VisitRecord::SubPlanId(s, valid) => {
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
    fn accept_node_post_visit(&self) -> Accepted {
        ACCEPT
    }
}

impl Visitor for LpIdentifierVisitor<'_> {
    type Node = ALogicalPlanNode;

    fn pre_visit(&mut self, _node: &Self::Node) -> PolarsResult<VisitRecursion> {
        self.visit_stack
            .push(VisitRecord::Entered(self.pre_visit_idx));
        self.pre_visit_idx += 1;

        // SAFETY:
        // we keep pointer valid and will not create mutable refs.
        self.identifier_array
            .push((0, unsafe { Identifier::new(self.expr_arena as *const _) }));
        Ok(VisitRecursion::Continue)
    }

    fn post_visit(&mut self, node: &Self::Node) -> PolarsResult<VisitRecursion> {
        self.post_visit_idx += 1;

        let (pre_visit_idx, sub_plan_id, is_valid_accumulated) = self.pop_until_entered();

        // Create the Id of this node.
        let id: Identifier = sub_plan_id.add_alp_node(node);

        if !is_valid_accumulated {
            self.identifier_array[pre_visit_idx].0 = self.post_visit_idx;
            self.visit_stack.push(VisitRecord::SubPlanId(id, false));
            return Ok(VisitRecursion::Continue);
        }

        // If we don't store this node
        // we only push the visit_stack, so the parents know the trail.
        if let Some((recurse, local_is_valid)) = self.accept_node_post_visit() {
            self.identifier_array[pre_visit_idx].0 = self.post_visit_idx;

            self.visit_stack
                .push(VisitRecord::SubPlanId(id, local_is_valid));
            return Ok(recurse);
        }

        // Store the created id.
        self.identifier_array[pre_visit_idx] = (self.post_visit_idx, id.clone());

        // We popped until entered, push this Id on the stack so the trail
        // is available for the parent plan.
        self.visit_stack
            .push(VisitRecord::SubPlanId(id.clone(), true));

        let (_, sp_count) = self.sp_count.entry(id).or_insert_with(|| (node.node(), 0));
        *sp_count += 1;
        self.has_subplan |= *sp_count > 1;
        Ok(VisitRecursion::Continue)
    }
}

struct CommonSubPlanRewriter<'a> {
    sp_count: &'a SubPlanCount,
    identifier_array: &'a IdentifierArray,
    /// keep track of the replaced identifiers.
    replaced_identifiers: &'a mut PlHashSet<Identifier>,

    max_post_visit_idx: usize,
    /// index in traversal order in which `identifier_array`
    /// was written. This is the index in `identifier_array`.
    visited_idx: usize,
    /// Indicates if this expression is rewritten.
    rewritten: bool,
    cache_id: PlHashMap<Identifier, usize>,
}

impl<'a> CommonSubPlanRewriter<'a> {
    fn new(
        sp_count: &'a SubPlanCount,
        identifier_array: &'a IdentifierArray,
        replaced_identifiers: &'a mut PlHashSet<Identifier>,
    ) -> Self {
        Self {
            sp_count,
            identifier_array,
            replaced_identifiers,
            max_post_visit_idx: 0,
            visited_idx: 0,
            rewritten: false,
            cache_id: Default::default(),
        }
    }
}

impl RewritingVisitor for CommonSubPlanRewriter<'_> {
    type Node = ALogicalPlanNode;

    fn pre_visit(&mut self, _lp_node: &Self::Node) -> PolarsResult<RewriteRecursion> {
        if self.visited_idx >= self.identifier_array.len()
            || self.max_post_visit_idx > self.identifier_array[self.visited_idx].0
        {
            return Ok(RewriteRecursion::Stop);
        }

        let id = &self.identifier_array[self.visited_idx].1;

        // Id placeholder not overwritten, so we can skip this sub-expression.
        if !id.is_valid() {
            self.visited_idx += 1;
            return Ok(RewriteRecursion::MutateAndContinue);
        }

        let Some((_, count)) = self.sp_count.get(id) else {
            self.visited_idx += 1;
            return Ok(RewriteRecursion::NoMutateAndContinue);
        };

        if *count > 1 {
            self.replaced_identifiers.insert(id.clone());
            // Rewrite this sub-plan, don't visit its children
            Ok(RewriteRecursion::MutateAndStop)
        } else {
            // This is a unique plan
            // visit its children to see if they are cse
            self.visited_idx += 1;
            Ok(RewriteRecursion::NoMutateAndContinue)
        }
    }

    fn mutate(&mut self, mut node: Self::Node) -> PolarsResult<Self::Node> {
        let (post_visit_count, id) = &self.identifier_array[self.visited_idx];
        self.visited_idx += 1;

        if *post_visit_count < self.max_post_visit_idx {
            return Ok(node);
        }
        self.max_post_visit_idx = *post_visit_count;
        while self.visited_idx < self.identifier_array.len()
            && *post_visit_count > self.identifier_array[self.visited_idx].0
        {
            self.visited_idx += 1;
        }

        let cache_id = self.cache_id.len();
        let cache_id = *self.cache_id.entry(id.clone()).or_insert(cache_id);
        let count = self.sp_count.get(id).unwrap().1 - 1;

        let cache_node = ALogicalPlan::Cache {
            input: node.node(),
            id: cache_id,
            count,
        };
        node.assign(cache_node);
        self.rewritten = true;
        Ok(node)
    }
}

pub(crate) fn elim_cmn_subplans(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
) -> (Node, bool) {
    let mut sp_count = Default::default();
    let mut id_array = Default::default();

    let changed = ALogicalPlanNode::with_context(root, lp_arena, |lp_node| {
        let mut visitor = LpIdentifierVisitor::new(&mut sp_count, &mut id_array, expr_arena);

        lp_node.visit(&mut visitor).map(|_| ()).unwrap();

        let mut replaced_ids = Default::default();
        let mut rewriter = CommonSubPlanRewriter::new(&sp_count, &id_array, &mut replaced_ids);
        lp_node.rewrite(&mut rewriter).unwrap();
        rewriter.rewritten
    });

    (root, changed)
}
