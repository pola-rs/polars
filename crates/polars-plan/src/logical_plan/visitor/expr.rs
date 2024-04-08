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

    fn map_children(self, mut f: &mut dyn FnMut(Self) -> PolarsResult<Self>) -> PolarsResult<Self> {
        use polars_utils::functions::try_arc_map as am;
        use AggExpr::*;
        use Expr::*;
        #[rustfmt::skip]
        let ret = match self {
            Alias(l, r) => Alias(am(l, f)?, r),
            Column(_) => self,
            Columns(_) => self,
            DtypeColumn(_) => self,
            Literal(_) => self,
            BinaryExpr { left, op, right } => {
                BinaryExpr { left: am(left, &mut f)? , op, right: am(right, f)?}
            },
            Cast { expr, data_type, strict } => Cast { expr: am(expr, f)?, data_type, strict },
            Sort { expr, options } => Sort { expr: am(expr, f)?, options },
            Gather { expr, idx, returns_scalar } => Gather { expr: am(expr, &mut f)?, idx: am(idx, f)?, returns_scalar },
            SortBy { expr, by, descending } => SortBy { expr: am(expr, &mut f)?, by: by.into_iter().map(f).collect::<Result<_, _>>()?, descending },
            Agg(agg_expr) => Agg(match agg_expr {
                Min { input, propagate_nans } => Min { input: am(input, f)?, propagate_nans },
                Max { input, propagate_nans } => Max { input: am(input, f)?, propagate_nans },
                Median(x) => Median(am(x, f)?),
                NUnique(x) => NUnique(am(x, f)?),
                First(x) => First(am(x, f)?),
                Last(x) => Last(am(x, f)?),
                Mean(x) => Mean(am(x, f)?),
                Implode(x) => Implode(am(x, f)?),
                Count(x, nulls) => Count(am(x, f)?, nulls),
                Quantile { expr, quantile, interpol } => Quantile { expr: am(expr, &mut f)?, quantile: am(quantile, f)?, interpol },
                Sum(x) => Sum(am(x, f)?),
                AggGroups(x) => AggGroups(am(x, f)?),
                Std(x, ddf) => Std(am(x, f)?, ddf),
                Var(x, ddf) => Var(am(x, f)?, ddf),
            }),
            Ternary { predicate, truthy, falsy } => Ternary { predicate: am(predicate, &mut f)?, truthy: am(truthy, &mut f)?, falsy: am(falsy, f)? },
            Function { input, function, options } => Function { input: input.into_iter().map(f).collect::<Result<_, _>>()?, function, options },
            Explode(expr) => Explode(am(expr, f)?),
            Filter { input, by } => Filter { input: am(input, &mut f)?, by: am(by, f)? },
            Window { function, partition_by, options } => {
                let partition_by = partition_by.into_iter().map(&mut f).collect::<Result<_, _>>()?;
                Window { function: am(function, f)?, partition_by, options }
            },
            Wildcard => Wildcard,
            Slice { input, offset, length } => Slice { input: am(input, &mut f)?, offset: am(offset, &mut f)?, length: am(length, f)? },
            Exclude(expr, excluded) => Exclude(am(expr, f)?, excluded),
            KeepName(expr) => KeepName(am(expr, f)?),
            Len => Len,
            Nth(_) => self,
            RenameAlias { function, expr } => RenameAlias { function, expr: am(expr, f)? },
            AnonymousFunction { input, function, output_type, options } => {
                AnonymousFunction { input: input.into_iter().map(f).collect::<Result<_, _>>()?, function, output_type, options }
            },
            SubPlan(_, _) => self,
            Selector(_) => self,
        };
        Ok(ret)
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

    // Check single node on equality
    fn is_equal(&self, other: &Self) -> bool {
        self.with_arena(|arena| {
            let self_ae = self.to_aexpr();
            let other_ae = arena.get(other.node());

            use AExpr::*;
            match (self_ae, other_ae) {
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
                        input: il,
                        function: fl,
                        options: ol,
                        ..
                    },
                    Function {
                        input: ir,
                        function: fr,
                        options: or,
                    },
                ) => {
                    fl == fr && ol == or && {
                        let mut all_same_name = true;
                        for (l, r) in il.iter().zip(ir) {
                            all_same_name &= l.output_name() == r.output_name()
                        }

                        all_same_name
                    }
                },
                (AnonymousFunction { .. }, AnonymousFunction { .. }) => false,
                (BinaryExpr { op: l, .. }, BinaryExpr { op: r, .. }) => l == r,
                _ => false,
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

        scratch1.push(self.node);
        scratch2.push(other.node);

        loop {
            match (scratch1.pop(), scratch2.pop()) {
                (Some(l), Some(r)) => {
                    // SAFETY: we can pass a *mut pointer
                    // the equality operation will not access mutable
                    let l = unsafe { AexprNode::from_raw(l, self.arena) };
                    let r = unsafe { AexprNode::from_raw(r, self.arena) };

                    if !l.is_equal(&r) {
                        return false;
                    }

                    l.to_aexpr().nodes(&mut scratch1);
                    r.to_aexpr().nodes(&mut scratch2);
                },
                (None, None) => return true,
                _ => return false,
            }
        }
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
