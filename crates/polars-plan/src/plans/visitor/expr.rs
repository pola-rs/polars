use std::fmt::{Debug, Formatter};

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
            Literal(_) => self,
            DataTypeFunction(_) => self,
            #[cfg(feature = "dtype-struct")]
            Field(_) => self,
            BinaryExpr { left, op, right } => {
                BinaryExpr { left: am(left, &mut f)? , op, right: am(right, f)?}
            },
            Cast { expr, dtype, options: strict } => Cast { expr: am(expr, f)?, dtype, options: strict },
            Sort { expr, options } => Sort { expr: am(expr, f)?, options },
            Gather { expr, idx, returns_scalar, null_on_oob } => Gather {
                expr: am(expr, &mut f)?,
                idx: am(idx, f)?,
                returns_scalar,
                null_on_oob,
            },
            SortBy { expr, by, sort_options } => SortBy { expr: am(expr, &mut f)?, by: by.into_iter().map(f).collect::<Result<_, _>>()?, sort_options },
            Agg(agg_expr) => Agg(match agg_expr {
                Min { input, propagate_nans } => Min { input: am(input, f)?, propagate_nans },
                Max { input, propagate_nans } => Max { input: am(input, f)?, propagate_nans },
                Median(x) => Median(am(x, f)?),
                NUnique(x) => NUnique(am(x, f)?),
                First(x) => First(am(x, f)?),
                FirstNonNull(x) => FirstNonNull(am(x, f)?),
                Last(x) => Last(am(x, f)?),
                LastNonNull(x) => LastNonNull(am(x, f)?),
                Item { input, allow_empty } => Item { input: am(input, f)?, allow_empty },
                Mean(x) => Mean(am(x, f)?),
                Implode(x) => Implode(am(x, f)?),
                Count { input, include_nulls } => Count { input: am(input, f)?, include_nulls },
                Quantile { expr, quantile, method: interpol } => Quantile { expr: am(expr, &mut f)?, quantile: am(quantile, f)?, method: interpol },
                Sum(x) => Sum(am(x, f)?),
                AggGroups(x) => AggGroups(am(x, f)?),
                Std(x, ddf) => Std(am(x, f)?, ddf),
                Var(x, ddf) => Var(am(x, f)?, ddf),

            }),
            Ternary { predicate, truthy, falsy } => Ternary { predicate: am(predicate, &mut f)?, truthy: am(truthy, &mut f)?, falsy: am(falsy, f)? },
            Function { input, function } => Function { input: input.into_iter().map(f).collect::<Result<_, _>>()?, function },
            Explode { input, options } => Explode { input: am(input, f)?, options },
            Filter { input, by } => Filter { input: am(input, &mut f)?, by: am(by, f)? },
            #[cfg(feature = "dynamic_group_by")]
            Rolling { function, index_column, period, offset, closed_window  } => Rolling { function: am(function, &mut f)?, index_column: am(index_column, &mut f)?, period, offset, closed_window  },
            Over { function, partition_by, order_by, mapping } => {
                let partition_by = partition_by.into_iter().map(&mut f).collect::<Result<_, _>>()?;
                Over { function: am(function, f)?, partition_by, order_by, mapping }
            },
            Slice { input, offset, length } => Slice { input: am(input, &mut f)?, offset: am(offset, &mut f)?, length: am(length, f)? },
            KeepName(expr) => KeepName(am(expr, f)?),
            Element => Element,
            Len => Len,
            RenameAlias { function, expr } => RenameAlias { function, expr: am(expr, f)? },
            AnonymousAgg { input, function, fmt_str } => {
                AnonymousAgg { input: input.into_iter().map(f).collect::<Result<_, _>>()?, function, fmt_str }
            },
            AnonymousFunction { input, function, options, fmt_str } => {
                AnonymousFunction { input: input.into_iter().map(f).collect::<Result<_, _>>()?, function, options, fmt_str }
            },
            Eval { expr: input, evaluation, variant } => Eval { expr: am(input, &mut f)?, evaluation: am(evaluation, f)?, variant },
            #[cfg(feature = "dtype-struct")]
            StructEval { expr: input, evaluation } => {
                StructEval { expr: am(input, &mut f)?, evaluation: evaluation.into_iter().map(f).collect::<Result<_, _>>()?  }
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
        aexpr.to_field(&ToFieldContext::new(arena, schema))
    }

    pub fn assign(&mut self, ae: AExpr, arena: &mut Arena<AExpr>) {
        let node = arena.add(ae);
        self.node = node;
    }

    pub(crate) fn is_leaf(&self, arena: &Arena<AExpr>) -> bool {
        matches!(self.to_aexpr(arena), AExpr::Column(_) | AExpr::Literal(_))
    }

    pub(crate) fn hashable_and_cmp<'a>(&self, arena: &'a Arena<AExpr>) -> AExprArena<'a> {
        AExprArena {
            node: self.node,
            arena,
        }
    }
}

pub struct AExprArena<'a> {
    node: Node,
    arena: &'a Arena<AExpr>,
}

impl Debug for AExprArena<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "AexprArena: {}", self.node.0)
    }
}

impl AExpr {
    fn is_equal_node(&self, other: &Self) -> bool {
        use AExpr::*;
        match (self, other) {
            (Column(l), Column(r)) => l == r,
            (Literal(l), Literal(r)) => l == r,
            #[cfg(feature = "dynamic_group_by")]
            (
                Rolling {
                    function: _,
                    index_column: _,
                    period: l_period,
                    offset: l_offset,
                    closed_window: l_closed_window,
                },
                Rolling {
                    function: _,
                    index_column: _,
                    period: r_period,
                    offset: r_offset,
                    closed_window: r_closed_window,
                },
            ) => l_period == r_period && l_offset == r_offset && l_closed_window == r_closed_window,
            (Over { mapping: l, .. }, Over { mapping: r, .. }) => l == r,
            (
                Cast {
                    options: strict_l,
                    dtype: dtl,
                    ..
                },
                Cast {
                    options: strict_r,
                    dtype: dtr,
                    ..
                },
            ) => strict_l == strict_r && dtl == dtr,
            (Sort { options: l, .. }, Sort { options: r, .. }) => l == r,
            (Gather { .. }, Gather { .. })
            | (Filter { .. }, Filter { .. })
            | (Ternary { .. }, Ternary { .. })
            | (Len, Len)
            | (Slice { .. }, Slice { .. }) => true,
            (
                Explode {
                    expr: _,
                    options: l_options,
                },
                Explode {
                    expr: _,
                    options: r_options,
                },
            ) => l_options == r_options,
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
            (
                AnonymousFunction {
                    function: l1,
                    options: l2,
                    fmt_str: l3,
                    input: _,
                },
                AnonymousFunction {
                    function: r1,
                    options: r2,
                    fmt_str: r3,
                    input: _,
                },
            ) => {
                l2 == r2 && l3 == r3 && {
                    use LazySerde as L;
                    match (l1, r1) {
                        // We only check the pointers, so this works for python
                        // functions that are on the same address.
                        (L::Deserialized(l0), L::Deserialized(r0)) => l0 == r0,
                        (L::Bytes(l0), L::Bytes(r0)) => l0 == r0,
                        (
                            L::Named {
                                name: l_name,
                                payload: l_payload,
                                value: l_value,
                            },
                            L::Named {
                                name: r_name,
                                payload: r_payload,
                                value: r_value,
                            },
                        ) => l_name == r_name && l_payload == r_payload && l_value == r_value,
                        _ => false,
                    }
                }
            },
            (BinaryExpr { op: l, .. }, BinaryExpr { op: r, .. }) => l == r,
            _ => false,
        }
    }
}

impl<'a> AExprArena<'a> {
    pub fn new(node: Node, arena: &'a Arena<AExpr>) -> Self {
        Self { node, arena }
    }
    pub fn to_aexpr(&self) -> &'a AExpr {
        self.arena.get(self.node)
    }

    // Check single node on equality
    pub fn is_equal_single(&self, other: &Self) -> bool {
        let self_ae = self.to_aexpr();
        let other_ae = other.to_aexpr();
        self_ae.is_equal_node(other_ae)
    }
}

impl PartialEq for AExprArena<'_> {
    fn eq(&self, other: &Self) -> bool {
        let mut scratch1 = unitvec![];
        let mut scratch2 = unitvec![];

        scratch1.push(self.node);
        scratch2.push(other.node);

        loop {
            match (scratch1.pop(), scratch2.pop()) {
                (Some(l), Some(r)) => {
                    let l = Self::new(l, self.arena);
                    let r = Self::new(r, other.arena);

                    if !l.is_equal_single(&r) {
                        return false;
                    }

                    l.to_aexpr().inputs_rev(&mut scratch1);
                    r.to_aexpr().inputs_rev(&mut scratch2);
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

        self.to_aexpr(arena).inputs_rev(&mut scratch);
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
        ae.inputs_rev(&mut scratch);

        // rewrite the nodes
        for node in scratch.as_mut_slice() {
            let aenode = AexprNode::new(*node);
            *node = op(aenode, arena)?.node;
        }

        scratch.as_mut_slice().reverse();
        let ae = ae.replace_inputs(&scratch);
        self.node = arena.add(ae);
        Ok(self)
    }
}
