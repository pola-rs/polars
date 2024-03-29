use super::*;
use crate::prelude::visitor::ALogicalPlanNode;

mod identifier_impl {
    use std::fmt::{Debug, Formatter};
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

    impl Debug for Identifier {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.last_node.as_ref().map(|n| n.to_alp()))
        }
    }

    impl PartialEq<Self> for Identifier {
        fn eq(&self, other: &Self) -> bool {
            self.inner == other.inner
                && match (self.last_node, other.last_node) {
                    (None, None) => true,
                    (Some(l), Some(r)) => {
                        let expr_arena = unsafe { &*self.expr_arena };
                        // We ignore caches as they are inserted on the node locations.
                        // In that case we don't want to cmp the cache (as we just inserted it),
                        // but the input node of the cache.
                        l.hashable_and_cmp(expr_arena).ignore_caches()
                            == r.hashable_and_cmp(expr_arena).ignore_caches()
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
type SubPlanCount = PlHashMap<Identifier, (Node, u32)>;
/// (post_visit_idx, identifier);
type IdentifierArray = Vec<(usize, Identifier)>;

/// See Expr based CSE for explanations.
enum VisitRecord {
    /// Entered a new plan node
    Entered(usize),
    SubPlanId(Identifier),
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

    fn pop_until_entered(&mut self) -> (usize, Identifier) {
        // SAFETY:
        // we keep pointer valid and will not create mutable refs.
        let mut id = unsafe { Identifier::new(self.expr_arena as *const _) };

        while let Some(item) = self.visit_stack.pop() {
            match item {
                VisitRecord::Entered(idx) => return (idx, id),
                VisitRecord::SubPlanId(s) => {
                    id.combine(&s);
                },
            }
        }
        unreachable!()
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

        let (pre_visit_idx, sub_plan_id) = self.pop_until_entered();

        // Create the Id of this node.
        let id: Identifier = sub_plan_id.add_alp_node(node);

        // Store the created id.
        self.identifier_array[pre_visit_idx] = (self.post_visit_idx, id.clone());

        // We popped until entered, push this Id on the stack so the trail
        // is available for the parent plan.
        self.visit_stack.push(VisitRecord::SubPlanId(id.clone()));

        let (_, sp_count) = self.sp_count.entry(id).or_insert_with(|| (node.node(), 0));
        *sp_count += 1;
        self.has_subplan |= *sp_count > 1;
        Ok(VisitRecursion::Continue)
    }
}

pub(super) type CacheId2Caches = PlHashMap<usize, (u32, Vec<Node>)>;

struct CommonSubPlanRewriter<'a> {
    sp_count: &'a SubPlanCount,
    identifier_array: &'a IdentifierArray,

    max_post_visit_idx: usize,
    /// index in traversal order in which `identifier_array`
    /// was written. This is the index in `identifier_array`.
    visited_idx: usize,
    /// Indicates if this expression is rewritten.
    rewritten: bool,
    cache_id: PlHashMap<Identifier, usize>,
    // Maps cache_id : (cache_count and cache_nodes)
    cache_id_to_caches: CacheId2Caches,
}

impl<'a> CommonSubPlanRewriter<'a> {
    fn new(sp_count: &'a SubPlanCount, identifier_array: &'a IdentifierArray) -> Self {
        Self {
            sp_count,
            identifier_array,
            max_post_visit_idx: 0,
            visited_idx: 0,
            rewritten: false,
            cache_id: Default::default(),
            cache_id_to_caches: Default::default(),
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
        let cache_count = self.sp_count.get(id).unwrap().1;

        let cache_node = ALogicalPlan::Cache {
            input: node.node(),
            id: cache_id,
            cache_hits: cache_count - 1,
        };
        node.assign(cache_node);
        let (_count, nodes) = self
            .cache_id_to_caches
            .entry(cache_id)
            .or_insert_with(|| (cache_count, vec![]));
        nodes.push(node.node());
        self.rewritten = true;
        Ok(node)
    }
}

pub(crate) fn elim_cmn_subplans(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
) -> (Node, bool, CacheId2Caches) {
    let mut sp_count = Default::default();
    let mut id_array = Default::default();

    let (changed, cache_id_to_caches) = ALogicalPlanNode::with_context(root, lp_arena, |lp_node| {
        let mut visitor = LpIdentifierVisitor::new(&mut sp_count, &mut id_array, expr_arena);

        lp_node.visit(&mut visitor).map(|_| ()).unwrap();

        let mut rewriter = CommonSubPlanRewriter::new(&sp_count, &id_array);
        lp_node.rewrite(&mut rewriter).unwrap();

        (rewriter.rewritten, rewriter.cache_id_to_caches)
    });

    (root, changed, cache_id_to_caches)
}

/// Prune unused caches.
/// In the query below the query will be insert cache 0 with a count of 2 on `lf.select`
/// and cache 1 with a count of 3 on `lf`. But because cache 0 is higher in the chain cache 1
/// will never be used. So we prune caches that don't fit their count.
///
/// `conctat([lf.select(), lf.select(), lf])`
///
pub(crate) fn prune_unused_caches(lp_arena: &mut Arena<ALogicalPlan>, cid2c: CacheId2Caches) {
    for (count, nodes) in cid2c.values() {
        if *count == nodes.len() as u32 {
            continue;
        }

        for node in nodes {
            let ALogicalPlan::Cache { input, .. } = lp_arena.get(*node) else {
                unreachable!()
            };
            lp_arena.swap(*input, *node)
        }
    }
}
