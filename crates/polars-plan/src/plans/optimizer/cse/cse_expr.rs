use hashbrown::hash_map::RawEntryMut;
use polars_utils::vec::CapacityByFactor;

use super::*;
use crate::constants::CSE_REPLACED;
use crate::prelude::visitor::AexprNode;

const SERIES_LIMIT: usize = 1000;

use polars_core::hashing::_boost_hash_combine;

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

/// Identifier that shows the sub-expression path.
/// Must implement hash and equality and ideally
/// have little collisions
/// We will do a full expression comparison to check if the
/// expressions with equal identifiers are truly equal
#[derive(Clone, Debug)]
pub(super) struct Identifier {
    inner: Option<u64>,
    last_node: Option<AexprNode>,
    hb: PlRandomState,
}

impl Identifier {
    fn new() -> Self {
        Self {
            inner: None,
            last_node: None,
            hb: PlRandomState::with_seed(0),
        }
    }

    fn hash(&self) -> u64 {
        self.inner.unwrap_or(0)
    }

    fn ae_node(&self) -> AexprNode {
        self.last_node.unwrap()
    }

    fn is_equal(&self, other: &Self, arena: &Arena<AExpr>) -> bool {
        self.inner == other.inner
            && self.last_node.map(|v| v.hashable_and_cmp(arena))
                == other.last_node.map(|v| v.hashable_and_cmp(arena))
    }

    fn is_valid(&self) -> bool {
        self.inner.is_some()
    }

    fn materialize(&self) -> String {
        format!("{}{:#x}", CSE_REPLACED, self.materialized_hash())
    }

    fn materialized_hash(&self) -> u64 {
        self.inner.unwrap_or(0)
    }

    fn combine(&mut self, other: &Identifier) {
        let inner = match (self.inner, other.inner) {
            (Some(l), Some(r)) => _boost_hash_combine(l, r),
            (None, Some(r)) => r,
            (Some(l), None) => l,
            _ => return,
        };
        self.inner = Some(inner);
    }

    fn add_ae_node(&self, ae: &AexprNode, arena: &Arena<AExpr>) -> Self {
        let hashed = self.hb.hash_one(ae.to_aexpr(arena));
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

#[derive(Default)]
struct IdentifierMap<V> {
    inner: PlHashMap<Identifier, V>,
}

impl<V> IdentifierMap<V> {
    fn get(&self, id: &Identifier, arena: &Arena<AExpr>) -> Option<&V> {
        self.inner
            .raw_entry()
            .from_hash(id.hash(), |k| k.is_equal(id, arena))
            .map(|(_k, v)| v)
    }

    fn entry<'a, F: FnOnce() -> V>(
        &'a mut self,
        id: Identifier,
        v: F,
        arena: &Arena<AExpr>,
    ) -> &'a mut V {
        let h = id.hash();
        match self
            .inner
            .raw_entry_mut()
            .from_hash(h, |k| k.is_equal(&id, arena))
        {
            RawEntryMut::Occupied(entry) => entry.into_mut(),
            RawEntryMut::Vacant(entry) => {
                let (_, v) = entry.insert_with_hasher(h, id, v(), |id| id.hash());
                v
            },
        }
    }
    fn insert(&mut self, id: Identifier, v: V, arena: &Arena<AExpr>) {
        self.entry(id, || v, arena);
    }

    fn iter(&self) -> impl Iterator<Item = (&Identifier, &V)> {
        self.inner.iter()
    }
}

/// Identifier maps to Expr Node and count.
type SubExprCount = IdentifierMap<(Node, u32)>;
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
    /// Materialized `CSE` materialized (name) hashes can collide. So we validate that all CSE counts
    /// match name hash counts.
    name_validation: &'a mut PlHashMap<u64, u32>,
    identifier_array: &'a mut IdentifierArray,
    // Index in pre-visit traversal order.
    pre_visit_idx: usize,
    post_visit_idx: usize,
    visit_stack: &'a mut Vec<VisitRecord>,
    /// Offset in the identifier array
    /// this allows us to use a single `vec` on multiple expressions
    id_array_offset: usize,
    // Whether the expression replaced a subexpression.
    has_sub_expr: bool,
    // During aggregation we only identify element-wise operations
    is_group_by: bool,
}

impl ExprIdentifierVisitor<'_> {
    fn new<'a>(
        se_count: &'a mut SubExprCount,
        identifier_array: &'a mut IdentifierArray,
        visit_stack: &'a mut Vec<VisitRecord>,
        is_group_by: bool,
        name_validation: &'a mut PlHashMap<u64, u32>,
    ) -> ExprIdentifierVisitor<'a> {
        let id_array_offset = identifier_array.len();
        ExprIdentifierVisitor {
            se_count,
            name_validation,
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
        match ae {
            // window expressions should `evaluate_on_groups`, not `evaluate`
            // so we shouldn't cache the children as they are evaluated incorrectly
            AExpr::Window { .. } => REFUSE_SKIP,
            // Don't allow this for now, as we can get `null().cast()` in ternary expressions.
            // TODO! Add a typed null
            AExpr::Literal(LiteralValue::Null) => REFUSE_NO_MEMBER,
            AExpr::Literal(s) => {
                match s {
                    LiteralValue::Series(s) => {
                        let dtype = s.dtype();

                        // Object and nested types are harder to hash and compare.
                        let allow = !(dtype.is_nested() | dtype.is_object());

                        if s.len() < SERIES_LIMIT && allow {
                            REFUSE_ALLOW_MEMBER
                        } else {
                            REFUSE_NO_MEMBER
                        }
                    },
                    _ => REFUSE_ALLOW_MEMBER,
                }
            },
            AExpr::Column(_) | AExpr::Alias(_, _) => REFUSE_ALLOW_MEMBER,
            AExpr::Len => {
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
    type Arena = Arena<AExpr>;

    fn pre_visit(
        &mut self,
        node: &Self::Node,
        arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        if skip_pre_visit(node.to_aexpr(arena), self.is_group_by) {
            // Still add to the stack so that a parent becomes invalidated.
            self.visit_stack
                .push(VisitRecord::SubExprId(Identifier::new(), false));
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

    fn post_visit(
        &mut self,
        node: &Self::Node,
        arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        let ae = node.to_aexpr(arena);
        self.post_visit_idx += 1;

        let (pre_visit_idx, sub_expr_id, is_valid_accumulated) = self.pop_until_entered();
        // Create the Id of this node.
        let id: Identifier = sub_expr_id.add_ae_node(node, arena);

        if !is_valid_accumulated {
            self.identifier_array[pre_visit_idx + self.id_array_offset].0 = self.post_visit_idx;
            self.visit_stack.push(VisitRecord::SubExprId(id, false));
            return Ok(VisitRecursion::Continue);
        }

        // If we don't store this node
        // we only push the visit_stack, so the parents know the trail.
        if let Some((recurse, local_is_valid)) = self.accept_node_post_visit(ae) {
            self.identifier_array[pre_visit_idx + self.id_array_offset].0 = self.post_visit_idx;

            self.visit_stack
                .push(VisitRecord::SubExprId(id, local_is_valid));
            return Ok(recurse);
        }

        // Store the created id.
        self.identifier_array[pre_visit_idx + self.id_array_offset] =
            (self.post_visit_idx, id.clone());

        // We popped until entered, push this Id on the stack so the trail
        // is available for the parent expression.
        self.visit_stack
            .push(VisitRecord::SubExprId(id.clone(), true));

        let mat_h = id.materialized_hash();
        let (_, se_count) = self.se_count.entry(id, || (node.node(), 0), arena);

        *se_count += 1;
        *self.name_validation.entry(mat_h).or_insert(0) += 1;
        self.has_sub_expr |= *se_count > 1;

        Ok(VisitRecursion::Continue)
    }
}

struct CommonSubExprRewriter<'a> {
    sub_expr_map: &'a SubExprCount,
    identifier_array: &'a IdentifierArray,
    /// keep track of the replaced identifiers.
    replaced_identifiers: &'a mut IdentifierMap<()>,

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
        replaced_identifiers: &'a mut IdentifierMap<()>,
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
    type Arena = Arena<AExpr>;

    fn pre_visit(
        &mut self,
        ae_node: &Self::Node,
        arena: &mut Self::Arena,
    ) -> PolarsResult<RewriteRecursion> {
        let ae = ae_node.to_aexpr(arena);
        if self.visited_idx + self.id_array_offset >= self.identifier_array.len()
            || self.max_post_visit_idx
                > self.identifier_array[self.visited_idx + self.id_array_offset].0
            || skip_pre_visit(ae, self.is_group_by)
        {
            return Ok(RewriteRecursion::Stop);
        }

        let id = &self.identifier_array[self.visited_idx + self.id_array_offset].1;

        // Id placeholder not overwritten, so we can skip this sub-expression.
        if !id.is_valid() {
            self.visited_idx += 1;
            let recurse = if ae_node.is_leaf(arena) {
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
        let Some((_, count)) = self.sub_expr_map.get(id, arena) else {
            self.visited_idx += 1;
            return Ok(RewriteRecursion::NoMutateAndContinue);
        };
        if *count > 1 {
            self.replaced_identifiers.insert(id.clone(), (), arena);
            // rewrite this sub-expression, don't visit its children
            Ok(RewriteRecursion::MutateAndStop)
        } else {
            // This is a unique expression
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
        debug_assert_eq!(
            node.hashable_and_cmp(arena),
            id.ae_node().hashable_and_cmp(arena)
        );

        let name = id.materialize();
        node.assign(AExpr::col(name.as_ref()), arena);
        self.rewritten = true;

        Ok(node)
    }
}

pub(crate) struct CommonSubExprOptimizer {
    // amortize allocations
    // these are cleared per lp node
    se_count: SubExprCount,
    id_array: IdentifierArray,
    id_array_offsets: Vec<u32>,
    replaced_identifiers: IdentifierMap<()>,
    // these are cleared per expr node
    visit_stack: Vec<VisitRecord>,
    name_validation: PlHashMap<u64, u32>,
}

impl CommonSubExprOptimizer {
    pub(crate) fn new() -> Self {
        Self {
            se_count: Default::default(),
            id_array: Default::default(),
            visit_stack: Default::default(),
            id_array_offsets: Default::default(),
            replaced_identifiers: Default::default(),
            name_validation: Default::default(),
        }
    }

    fn visit_expression(
        &mut self,
        ae_node: AexprNode,
        is_group_by: bool,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<(usize, bool)> {
        let mut visitor = ExprIdentifierVisitor::new(
            &mut self.se_count,
            &mut self.id_array,
            &mut self.visit_stack,
            is_group_by,
            &mut self.name_validation,
        );
        ae_node.visit(&mut visitor, expr_arena).map(|_| ())?;
        Ok((visitor.id_array_offset, visitor.has_sub_expr))
    }

    /// Mutate the expression.
    /// Returns a new expression and a `bool` indicating if it was rewritten or not.
    fn mutate_expression(
        &mut self,
        ae_node: AexprNode,
        id_array_offset: usize,
        is_group_by: bool,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<(AexprNode, bool)> {
        let mut rewriter = CommonSubExprRewriter::new(
            &self.se_count,
            &self.id_array,
            &mut self.replaced_identifiers,
            id_array_offset,
            is_group_by,
        );
        ae_node
            .rewrite(&mut rewriter, expr_arena)
            .map(|out| (out, rewriter.rewritten))
    }

    fn find_cse(
        &mut self,
        expr: &[ExprIR],
        expr_arena: &mut Arena<AExpr>,
        id_array_offsets: &mut Vec<u32>,
        is_group_by: bool,
        schema: &Schema,
    ) -> PolarsResult<Option<ProjectionExprs>> {
        let mut has_sub_expr = false;

        // First get all cse's.
        for e in expr {
            // The visitor can return early thus depleted its stack
            // on a previous iteration.
            self.visit_stack.clear();

            // Visit expressions and collect sub-expression counts.
            let ae_node = AexprNode::new(e.node());
            let (id_array_offset, this_expr_has_se) =
                self.visit_expression(ae_node, is_group_by, expr_arena)?;
            id_array_offsets.push(id_array_offset as u32);
            has_sub_expr |= this_expr_has_se;
        }

        // Ensure that the `materialized hashes` count matches that of the CSE count.
        // It can happen that CSE collide and in that case we fallback and skip CSE.
        for (id, (_, count)) in self.se_count.iter() {
            let mat_h = id.materialized_hash();
            let valid = if let Some(name_count) = self.name_validation.get(&mat_h) {
                *name_count == *count
            } else {
                false
            };

            if !valid {
                if verbose() {
                    eprintln!("materialized names collided in common subexpression elimination.\n backtrace and run without CSE")
                }
                return Ok(None);
            }
        }

        if has_sub_expr {
            let mut new_expr = Vec::with_capacity_by_factor(expr.len(), 1.3);

            // Then rewrite the expressions that have a cse count > 1.
            for (e, offset) in expr.iter().zip(id_array_offsets.iter()) {
                let ae_node = AexprNode::new(e.node());

                let (out, rewritten) =
                    self.mutate_expression(ae_node, *offset as usize, is_group_by, expr_arena)?;

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
                        out_e.set_alias(ColumnName::from(name.as_str()));
                    }
                    out_e
                };
                new_expr.push(new_node)
            }
            // Add the tmp columns
            for id in self.replaced_identifiers.inner.keys() {
                let (node, _count) = self.se_count.get(id, expr_arena).unwrap();
                let name = id.materialize();
                let out_e = ExprIR::new(*node, OutputName::Alias(ColumnName::from(name)));
                new_expr.push(out_e)
            }
            let expr =
                ProjectionExprs::new_with_cse(new_expr, self.replaced_identifiers.inner.len());
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
        let mut id_array_offsets = std::mem::take(&mut self.id_array_offsets);

        self.se_count.inner.clear();
        self.name_validation.clear();
        self.id_array.clear();
        id_array_offsets.clear();
        self.replaced_identifiers.inner.clear();

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
                    &mut id_array_offsets,
                    false,
                    input_schema.as_ref().as_ref(),
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
                    &mut id_array_offsets,
                    false,
                    input_schema.as_ref().as_ref(),
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
            } => {
                let input_schema = arena.0.get(*input).schema(&arena.0);
                if let Some(aggs) = self.find_cse(
                    aggs,
                    &mut arena.1,
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

        self.id_array_offsets = id_array_offsets;
        Ok(node)
    }
}
