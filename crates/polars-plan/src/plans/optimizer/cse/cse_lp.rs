use hashbrown::hash_map::RawEntryMut;

use super::*;
use crate::prelude::visitor::IRNode;

mod identifier_impl {
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
        last_node: Option<IRNode>,
        hb: PlRandomState,
    }

    impl Identifier {
        pub fn hash(&self) -> u64 {
            self.inner.unwrap_or(0)
        }

        pub fn is_equal(
            &self,
            other: &Self,
            lp_arena: &Arena<IR>,
            expr_arena: &Arena<AExpr>,
        ) -> bool {
            self.inner == other.inner
                && match (self.last_node, other.last_node) {
                    (None, None) => true,
                    (Some(l), Some(r)) => {
                        // We ignore caches as they are inserted on the node locations.
                        // In that case we don't want to cmp the cache (as we just inserted it),
                        // but the input node of the cache.
                        l.hashable_and_cmp(lp_arena, expr_arena).ignore_caches()
                            == r.hashable_and_cmp(lp_arena, expr_arena).ignore_caches()
                    },
                    _ => false,
                }
        }
        pub fn new() -> Self {
            Self {
                inner: None,
                last_node: None,
                hb: PlRandomState::with_seed(0),
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

        pub fn add_alp_node(
            &self,
            alp: &IRNode,
            lp_arena: &Arena<IR>,
            expr_arena: &Arena<AExpr>,
        ) -> Self {
            let hashed = self.hb.hash_one(alp.hashable_and_cmp(lp_arena, expr_arena));
            let inner = Some(
                self.inner
                    .map_or(hashed, |l| _boost_hash_combine(l, hashed)),
            );
            Self {
                inner,
                last_node: Some(*alp),
                hb: self.hb.clone(),
            }
        }
    }
}
use identifier_impl::*;

#[derive(Default)]
struct IdentifierMap<V> {
    inner: PlHashMap<Identifier, V>,
}

impl<V> IdentifierMap<V> {
    fn get(&self, id: &Identifier, lp_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Option<&V> {
        self.inner
            .raw_entry()
            .from_hash(id.hash(), |k| k.is_equal(id, lp_arena, expr_arena))
            .map(|(_k, v)| v)
    }

    fn entry<F: FnOnce() -> V>(
        &mut self,
        id: Identifier,
        v: F,
        lp_arena: &Arena<IR>,
        expr_arena: &Arena<AExpr>,
    ) -> &mut V {
        let h = id.hash();
        match self
            .inner
            .raw_entry_mut()
            .from_hash(h, |k| k.is_equal(&id, lp_arena, expr_arena))
        {
            RawEntryMut::Occupied(entry) => entry.into_mut(),
            RawEntryMut::Vacant(entry) => {
                let (_, v) = entry.insert_with_hasher(h, id, v(), |id| id.hash());
                v
            },
        }
    }
}

/// Identifier maps to Expr Node and count.
type SubPlanCount = IdentifierMap<(Node, u32)>;
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
}

impl LpIdentifierVisitor<'_> {
    fn new<'a>(
        sp_count: &'a mut SubPlanCount,
        identifier_array: &'a mut IdentifierArray,
    ) -> LpIdentifierVisitor<'a> {
        LpIdentifierVisitor {
            sp_count,
            identifier_array,
            pre_visit_idx: 0,
            post_visit_idx: 0,
            visit_stack: vec![],
            has_subplan: false,
        }
    }

    fn pop_until_entered(&mut self) -> (usize, Identifier) {
        let mut id = Identifier::new();

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

fn skip_children(lp: &IR) -> bool {
    match lp {
        // Don't visit all the files in a `scan *` operation.
        // Put an arbitrary limit to 20 files now.
        IR::Union {
            options, inputs, ..
        } => options.from_partitioned_ds && inputs.len() > 20,
        _ => false,
    }
}

impl<'a> Visitor for LpIdentifierVisitor<'a> {
    type Node = IRNode;
    type Arena = IRNodeArena;

    fn pre_visit(
        &mut self,
        node: &Self::Node,
        arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        self.visit_stack
            .push(VisitRecord::Entered(self.pre_visit_idx));
        self.pre_visit_idx += 1;

        self.identifier_array.push((0, Identifier::new()));

        if skip_children(node.to_alp(&arena.0)) {
            Ok(VisitRecursion::Skip)
        } else {
            Ok(VisitRecursion::Continue)
        }
    }

    fn post_visit(
        &mut self,
        node: &Self::Node,
        arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        self.post_visit_idx += 1;

        let (pre_visit_idx, sub_plan_id) = self.pop_until_entered();

        // Create the Id of this node.
        let id: Identifier = sub_plan_id.add_alp_node(node, &arena.0, &arena.1);

        // Store the created id.
        self.identifier_array[pre_visit_idx] = (self.post_visit_idx, id.clone());

        // We popped until entered, push this Id on the stack so the trail
        // is available for the parent plan.
        self.visit_stack.push(VisitRecord::SubPlanId(id.clone()));

        let (_, sp_count) = self
            .sp_count
            .entry(id, || (node.node(), 0), &arena.0, &arena.1);
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
    cache_id: IdentifierMap<usize>,
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

impl<'a> RewritingVisitor for CommonSubPlanRewriter<'a> {
    type Node = IRNode;
    type Arena = IRNodeArena;

    fn pre_visit(
        &mut self,
        lp_node: &Self::Node,
        arena: &mut Self::Arena,
    ) -> PolarsResult<RewriteRecursion> {
        if self.visited_idx >= self.identifier_array.len()
            || self.max_post_visit_idx > self.identifier_array[self.visited_idx].0
        {
            return Ok(RewriteRecursion::Stop);
        }

        let id = &self.identifier_array[self.visited_idx].1;

        // Id placeholder not overwritten, so we can skip this sub-expression.
        if !id.is_valid() {
            self.visited_idx += 1;
            return Ok(RewriteRecursion::NoMutateAndContinue);
        }

        let Some((_, count)) = self.sp_count.get(id, &arena.0, &arena.1) else {
            self.visited_idx += 1;
            return Ok(RewriteRecursion::NoMutateAndContinue);
        };

        if *count > 1 {
            // Rewrite this sub-plan, don't visit its children
            Ok(RewriteRecursion::MutateAndStop)
        }
        // Never mutate if count <= 1. The post-visit will search for the node, and not be able to find it
        else {
            // Don't traverse the children.
            if skip_children(lp_node.to_alp(&arena.0)) {
                return Ok(RewriteRecursion::Stop);
            }
            // This is a unique plan
            // visit its children to see if they are cse
            self.visited_idx += 1;
            Ok(RewriteRecursion::NoMutateAndContinue)
        }
    }

    fn mutate(
        &mut self,
        mut node: Self::Node,
        arena: &mut Self::Arena,
    ) -> PolarsResult<Self::Node> {
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

        let cache_id = self.cache_id.inner.len();
        let cache_id = *self
            .cache_id
            .entry(id.clone(), || cache_id, &arena.0, &arena.1);
        let cache_count = self.sp_count.get(id, &arena.0, &arena.1).unwrap().1;

        let cache_node = IR::Cache {
            input: node.node(),
            id: cache_id,
            cache_hits: cache_count - 1,
        };
        node.assign(cache_node, &mut arena.0);
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
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> (Node, bool, CacheId2Caches) {
    let mut sp_count = Default::default();
    let mut id_array = Default::default();

    with_ir_arena(lp_arena, expr_arena, |arena| {
        let lp_node = IRNode::new(root);
        let mut visitor = LpIdentifierVisitor::new(&mut sp_count, &mut id_array);

        lp_node.visit(&mut visitor, arena).map(|_| ()).unwrap();

        let mut rewriter = CommonSubPlanRewriter::new(&sp_count, &id_array);
        lp_node.rewrite(&mut rewriter, arena).unwrap();

        (root, rewriter.rewritten, rewriter.cache_id_to_caches)
    })
}

/// Prune unused caches.
/// In the query below the query will be insert cache 0 with a count of 2 on `lf.select`
/// and cache 1 with a count of 3 on `lf`. But because cache 0 is higher in the chain cache 1
/// will never be used. So we prune caches that don't fit their count.
///
/// `conctat([lf.select(), lf.select(), lf])`
///
pub(crate) fn prune_unused_caches(lp_arena: &mut Arena<IR>, cid2c: CacheId2Caches) {
    for (count, nodes) in cid2c.values() {
        if *count == nodes.len() as u32 {
            continue;
        }

        for node in nodes {
            let IR::Cache { input, .. } = lp_arena.get(*node) else {
                unreachable!()
            };
            lp_arena.swap(*input, *node)
        }
    }
}
