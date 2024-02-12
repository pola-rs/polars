use polars_core::prelude::{Field, Schema};
use polars_utils::unitvec;

use super::*;
use crate::prelude::*;

impl TreeWalker for Expr {
    fn apply_children<'a>(
        &'a self,
        op: &mut dyn FnMut(&Self) -> PolarsResult<VisitRecursion>,
    ) -> PolarsResult<VisitRecursion> {
        let mut scratch = unitvec![];

        self.nodes(&mut scratch);

        for &child in scratch.as_slice() {
            match op(child)? {
                // let the recursion continue
                VisitRecursion::Continue | VisitRecursion::Skip => {},
                // early stop
                VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
            }
        }
        Ok(VisitRecursion::Continue)
    }

    fn map_children(self, _op: &mut dyn FnMut(Self) -> PolarsResult<Self>) -> PolarsResult<Self> {
        todo!()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AexprNode {
    node: Node,
    arena: *mut Arena<AExpr>,
}

impl AexprNode {
    /// Don't use this directly, use [`Self::with_context`]
    ///
    /// # Safety
    /// This will keep a pointer to `arena`. The caller must ensure it stays alive.
    unsafe fn new(node: Node, arena: &mut Arena<AExpr>) -> Self {
        Self { node, arena }
    }

    /// # Safety
    /// This will keep a pointer to `arena`. The caller must ensure it stays alive.
    pub(crate) unsafe fn from_raw(node: Node, arena: *mut Arena<AExpr>) -> Self {
        Self { node, arena }
    }

    /// Safe interface. Take the `&mut Arena` only for the duration of `op`.
    pub fn with_context<F, T>(node: Node, arena: &mut Arena<AExpr>, op: F) -> T
    where
        F: FnOnce(AexprNode) -> T,
    {
        // SAFETY: we drop this context before arena is out of scope
        unsafe { op(Self::new(node, arena)) }
    }

    /// Safe interface. Take the `&mut Arena` only for the duration of `op`.
    pub fn with_context_and_arena<F, T>(node: Node, arena: &mut Arena<AExpr>, op: F) -> T
    where
        F: FnOnce(AexprNode, &mut Arena<AExpr>) -> T,
    {
        // SAFETY: we drop this context before arena is out of scope
        unsafe { op(Self::new(node, arena), arena) }
    }

    /// Get the `Node`.
    pub fn node(&self) -> Node {
        self.node
    }

    /// Apply an operation with the underlying `Arena`.
    pub fn with_arena<'a, F, T>(&self, op: F) -> T
    where
        F: FnOnce(&'a Arena<AExpr>) -> T,
    {
        let arena = unsafe { &(*self.arena) };

        op(arena)
    }

    /// Apply an operation with the underlying `Arena`.
    pub fn with_arena_mut<'a, F, T>(&mut self, op: F) -> T
    where
        F: FnOnce(&'a mut Arena<AExpr>) -> T,
    {
        let arena = unsafe { &mut (*self.arena) };

        op(arena)
    }

    /// Assign an `AExpr` to underlying arena.
    pub fn assign(&mut self, ae: AExpr) {
        let node = self.with_arena_mut(|arena| arena.add(ae));
        self.node = node
    }

    /// Take a `Node` and convert it an `AExprNode` and call
    /// `F` with `self` and the new created `AExprNode`
    pub fn binary<F, T>(&self, other: Node, op: F) -> T
    where
        F: FnOnce(&AexprNode, &AexprNode) -> T,
    {
        // this is safe as we remain in context
        let other = unsafe { AexprNode::from_raw(other, self.arena) };
        op(self, &other)
    }

    pub fn to_aexpr(&self) -> &AExpr {
        self.with_arena(|arena| arena.get(self.node))
    }

    pub fn to_expr(&self) -> Expr {
        self.with_arena(|arena| node_to_expr(self.node, arena))
    }

    pub fn to_field(&self, schema: &Schema) -> PolarsResult<Field> {
        self.with_arena(|arena| {
            let ae = arena.get(self.node);
            ae.to_field(schema, Context::Default, arena)
        })
    }

    // traverses all nodes and does a full equality check
    fn is_equal(&self, other: &Self, scratch1: &mut Vec<Node>, scratch2: &mut Vec<Node>) -> bool {
        self.with_arena(|arena| {
            let self_ae = self.to_aexpr();
            let other_ae = arena.get(other.node());

            use AExpr::*;
            let this_node_equal = match (self_ae, other_ae) {
                (Alias(_, l), Alias(_, r)) => l == r,
                (Column(l), Column(r)) => l == r,
                (Literal(l), Literal(r)) => l == r,
                (Nth(l), Nth(r)) => l == r,
                (Window { options: l, .. }, Window { options: r, .. }) => l == r,
                (
                    Cast {
                        strict: strict_l,
                        data_type: dtl,
                        ..
                    },
                    Cast {
                        strict: strict_r,
                        data_type: dtr,
                        ..
                    },
                ) => strict_l == strict_r && dtl == dtr,
                (Sort { options: l, .. }, Sort { options: r, .. }) => l == r,
                (Gather { .. }, Gather { .. })
                | (Filter { .. }, Filter { .. })
                | (Ternary { .. }, Ternary { .. })
                | (Len, Len)
                | (Slice { .. }, Slice { .. })
                | (Explode(_), Explode(_)) => true,
                (SortBy { descending: l, .. }, SortBy { descending: r, .. }) => l == r,
                (Agg(l), Agg(r)) => l.equal_nodes(r),
                (
                    Function {
                        function: fl,
                        options: ol,
                        ..
                    },
                    Function {
                        function: fr,
                        options: or,
                        ..
                    },
                ) => fl == fr && ol == or,
                (AnonymousFunction { .. }, AnonymousFunction { .. }) => false,
                (BinaryExpr { op: l, .. }, BinaryExpr { op: r, .. }) => l == r,
                _ => false,
            };

            if !this_node_equal {
                return false;
            }

            self_ae.nodes(scratch1);
            other_ae.nodes(scratch2);

            loop {
                match (scratch1.pop(), scratch2.pop()) {
                    (Some(l), Some(r)) => {
                        // SAFETY: we can pass a *mut pointer
                        // the equality operation will not access mutable
                        let l = unsafe { AexprNode::from_raw(l, self.arena) };
                        let r = unsafe { AexprNode::from_raw(r, self.arena) };

                        if !l.is_equal(&r, scratch1, scratch2) {
                            return false;
                        }
                    },
                    (None, None) => return true,
                    _ => return false,
                }
            }
        })
    }

    #[cfg(feature = "cse")]
    pub(crate) fn is_leaf(&self) -> bool {
        matches!(self.to_aexpr(), AExpr::Column(_) | AExpr::Literal(_))
    }
}

impl PartialEq for AexprNode {
    fn eq(&self, other: &Self) -> bool {
        let mut scratch1 = vec![];
        let mut scratch2 = vec![];
        self.is_equal(other, &mut scratch1, &mut scratch2)
    }
}

impl TreeWalker for AexprNode {
    fn apply_children<'a>(
        &'a self,
        op: &mut dyn FnMut(&Self) -> PolarsResult<VisitRecursion>,
    ) -> PolarsResult<VisitRecursion> {
        let mut scratch = vec![];

        self.to_aexpr().nodes(&mut scratch);
        for node in scratch {
            let aenode = AexprNode {
                node,
                arena: self.arena,
            };
            match op(&aenode)? {
                // let the recursion continue
                VisitRecursion::Continue | VisitRecursion::Skip => {},
                // early stop
                VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
            }
        }
        Ok(VisitRecursion::Continue)
    }

    fn map_children(
        mut self,
        op: &mut dyn FnMut(Self) -> PolarsResult<Self>,
    ) -> PolarsResult<Self> {
        let mut scratch = vec![];

        let ae = self.to_aexpr();
        ae.nodes(&mut scratch);

        // rewrite the nodes
        for node in &mut scratch {
            let aenode = AexprNode {
                node: *node,
                arena: self.arena,
            };
            *node = op(aenode)?.node;
        }

        let ae = ae.clone().replace_inputs(&scratch);
        let node = self.with_arena_mut(move |arena| arena.add(ae));
        self.node = node;
        Ok(self)
    }
}
