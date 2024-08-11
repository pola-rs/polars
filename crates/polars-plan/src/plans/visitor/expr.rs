use std::fmt::Debug;
#[cfg(feature = "cse")]
use std::fmt::Formatter;

use polars_core::prelude::{Field, Schema};
use polars_utils::unitvec;

use super::*;
use crate::prelude::*;

impl TreeWalker for Expr {
    type Arena = ();

    fn apply_children<F: FnMut(&Self, &Self::Arena) -> PolarsResult<VisitRecursion>>(
        &self,
        op: &mut F,
        arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        let mut scratch = unitvec![];

        self.nodes(&mut scratch);

        for &child in scratch.as_slice() {
            match op(child, arena)? {
                // let the recursion continue
                VisitRecursion::Continue | VisitRecursion::Skip => {},
                // early stop
                VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
            }
        }
        Ok(VisitRecursion::Continue)
    }

    fn map_children<F: FnMut(Self, &mut Self::Arena) -> PolarsResult<Self>>(
        self,
        f: &mut F,
        _arena: &mut Self::Arena,
    ) -> PolarsResult<Self> {
        use polars_utils::functions::try_arc_map as am;
        let mut f = |expr| f(expr, &mut ());
        use AggExpr::*;
        use Expr::*;
        #[rustfmt::skip]
        let ret = match self {
            Alias(l, r) => Alias(am(l, f)?, r),
            Column(_) => self,
            Columns(_) => self,
            DtypeColumn(_) => self,
            IndexColumn(_) => self,
            Literal(_) => self,
            #[cfg(feature = "dtype-struct")]
            Field(_) => self,
            BinaryExpr { left, op, right } => {
                BinaryExpr { left: am(left, &mut f)? , op, right: am(right, f)?}
            },
            Cast { expr, data_type, options: strict } => Cast { expr: am(expr, f)?, data_type, options: strict },
            Sort { expr, options } => Sort { expr: am(expr, f)?, options },
            Gather { expr, idx, returns_scalar } => Gather { expr: am(expr, &mut f)?, idx: am(idx, f)?, returns_scalar },
            SortBy { expr, by, sort_options } => SortBy { expr: am(expr, &mut f)?, by: by.into_iter().map(f).collect::<Result<_, _>>()?, sort_options },
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
            Window { function, partition_by, order_by, options } => {
                let partition_by = partition_by.into_iter().map(&mut f).collect::<Result<_, _>>()?;
                Window { function: am(function, f)?, partition_by, order_by, options }
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
}

impl AexprNode {
    pub fn new(node: Node) -> Self {
        Self { node }
    }

    /// Get the `Node`.
    pub fn node(&self) -> Node {
        self.node
    }

    pub fn to_aexpr<'a>(&self, arena: &'a Arena<AExpr>) -> &'a AExpr {
        arena.get(self.node)
    }

    pub fn to_expr(&self, arena: &Arena<AExpr>) -> Expr {
        node_to_expr(self.node, arena)
    }

    pub fn to_field(&self, schema: &Schema, arena: &Arena<AExpr>) -> PolarsResult<Field> {
        let aexpr = arena.get(self.node);
        aexpr.to_field(schema, Context::Default, arena)
    }

    pub fn assign(&mut self, ae: AExpr, arena: &mut Arena<AExpr>) {
        let node = arena.add(ae);
        self.node = node;
    }

    #[cfg(feature = "cse")]
    pub(crate) fn is_leaf(&self, arena: &Arena<AExpr>) -> bool {
        matches!(self.to_aexpr(arena), AExpr::Column(_) | AExpr::Literal(_))
    }

    #[cfg(feature = "cse")]
    pub(crate) fn hashable_and_cmp<'a>(&self, arena: &'a Arena<AExpr>) -> AExprArena<'a> {
        AExprArena {
            node: self.node,
            arena,
        }
    }
}

#[cfg(feature = "cse")]
pub struct AExprArena<'a> {
    node: Node,
    arena: &'a Arena<AExpr>,
}

#[cfg(feature = "cse")]
impl Debug for AExprArena<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "AexprArena: {}", self.node.0)
    }
}

impl AExpr {
    #[cfg(feature = "cse")]
    fn is_equal_node(&self, other: &Self) -> bool {
        use AExpr::*;
        match (self, other) {
            (Alias(_, l), Alias(_, r)) => l == r,
            (Column(l), Column(r)) => l == r,
            (Literal(l), Literal(r)) => l == r,
            (Window { options: l, .. }, Window { options: r, .. }) => l == r,
            (
                Cast {
                    options: strict_l,
                    data_type: dtl,
                    ..
                },
                Cast {
                    options: strict_r,
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
            (
                SortBy {
                    sort_options: l_sort_options,
                    ..
                },
                SortBy {
                    sort_options: r_sort_options,
                    ..
                },
            ) => l_sort_options == r_sort_options,
            (Agg(l), Agg(r)) => l.equal_nodes(r),
            (
                Function {
                    input: il,
                    function: fl,
                    options: ol,
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
    }
}

#[cfg(feature = "cse")]
impl<'a> AExprArena<'a> {
    fn new(node: Node, arena: &'a Arena<AExpr>) -> Self {
        Self { node, arena }
    }
    fn to_aexpr(&self) -> &'a AExpr {
        self.arena.get(self.node)
    }

    // Check single node on equality
    fn is_equal_single(&self, other: &Self) -> bool {
        let self_ae = self.to_aexpr();
        let other_ae = other.to_aexpr();
        self_ae.is_equal_node(other_ae)
    }
}

#[cfg(feature = "cse")]
impl PartialEq for AExprArena<'_> {
    fn eq(&self, other: &Self) -> bool {
        let mut scratch1 = vec![];
        let mut scratch2 = vec![];

        scratch1.push(self.node);
        scratch2.push(other.node);

        loop {
            match (scratch1.pop(), scratch2.pop()) {
                (Some(l), Some(r)) => {
                    let l = Self::new(l, self.arena);
                    let r = Self::new(r, self.arena);

                    if !l.is_equal_single(&r) {
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
    type Arena = Arena<AExpr>;
    fn apply_children<F: FnMut(&Self, &Self::Arena) -> PolarsResult<VisitRecursion>>(
        &self,
        op: &mut F,
        arena: &Self::Arena,
    ) -> PolarsResult<VisitRecursion> {
        let mut scratch = unitvec![];

        self.to_aexpr(arena).nodes(&mut scratch);
        for node in scratch.as_slice() {
            let aenode = AexprNode::new(*node);
            match op(&aenode, arena)? {
                // let the recursion continue
                VisitRecursion::Continue | VisitRecursion::Skip => {},
                // early stop
                VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
            }
        }
        Ok(VisitRecursion::Continue)
    }

    fn map_children<F: FnMut(Self, &mut Self::Arena) -> PolarsResult<Self>>(
        mut self,
        op: &mut F,
        arena: &mut Self::Arena,
    ) -> PolarsResult<Self> {
        let mut scratch = unitvec![];

        let ae = arena.get(self.node).clone();
        ae.nodes(&mut scratch);

        // rewrite the nodes
        for node in scratch.as_mut_slice() {
            let aenode = AexprNode::new(*node);
            *node = op(aenode, arena)?.node;
        }

        let ae = ae.replace_inputs(&scratch);
        self.node = arena.add(ae);
        Ok(self)
    }
}
