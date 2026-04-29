use std::ops::ControlFlow;

use polars_utils::itertools::Itertools as _;

use super::*;
use crate::traversal::tree_traversal::{GetNodeInputs, tree_traversal};
use crate::traversal::visitor::NodeVisitor;

impl AExpr {
    /// Iterator that retunrs the child nodes of this aexpr in field declaration order.
    ///
    /// This function and its users must be updated if the field declaration order changes.
    pub fn children_iter(&self) -> AENodesIter<'_> {
        use std::slice;

        use AExpr::*;

        match self {
            Element | Column(_) | Literal(_) | Len => AENodesIter::new_empty(),
            #[cfg(feature = "dtype-struct")]
            StructField(_) => AENodesIter::new_empty(),

            Cast {
                expr,
                dtype: _,
                options: _,
            }
            | Sort { expr, options: _ }
            | Explode { expr, options: _ } => AENodesIter::new_single(expr),

            Gather {
                expr,
                idx,
                returns_scalar: _,
                null_on_oob: _,
            } => AENodesIter::new_double(expr, idx),
            SortBy { expr, by, .. } => {
                AENodesIter::DoubleSlice(slice::from_ref(expr).iter().chain(by.iter()))
            },
            Filter { input, by } => AENodesIter::new_double(input, by),
            Agg(agg) => agg.children_iter(),
            BinaryExpr { left, op: _, right } => AENodesIter::new_double(left, right),
            Ternary {
                predicate,
                truthy,
                falsy,
            } => AENodesIter::new_triple(predicate, truthy, falsy),
            AnonymousFunction { input, .. }
            | Function { input, .. }
            | AnonymousAgg { input, .. } => {
                AENodesIter::ExprIRSlice(input.iter().map(ExprIR::node_ref))
            },

            #[cfg(feature = "dynamic_group_by")]
            Rolling {
                function,
                index_column,
                period: _,
                offset: _,
                closed_window: _,
            } => AENodesIter::new_double(function, index_column),
            Over {
                function,
                partition_by,
                order_by,
                mapping: _,
            } => AENodesIter::TripleSlice(
                std::slice::from_ref(function)
                    .iter()
                    .chain(partition_by.iter())
                    .chain(
                        order_by
                            .as_ref()
                            .map_or(&[][..], |x| std::slice::from_ref(&x.0)),
                    ),
            ),
            Eval {
                expr,
                evaluation,
                variant: _,
            } => AENodesIter::new_double(expr, evaluation),
            #[cfg(feature = "dtype-struct")]
            StructEval { expr, evaluation } => AENodesIter::StructEval(
                std::iter::once(expr).chain(evaluation.iter().map(&ExprIR::node_ref as _)),
            ),
            Slice {
                input,
                offset,
                length,
            } => AENodesIter::new_triple(input, offset, length),
        }
    }

    /// Iterator that returns nodes in order such that the last item of the iterator is the node
    /// from which the output name is sourced. The ordering of non-name nodes is unspecified but
    /// guaranteed to match `AExpr::children_iter_mut_name_last`.
    pub fn children_iter_name_last(&self) -> std::iter::Rev<AENodesIter<'_>> {
        use AExpr::*;

        Iterator::rev(match self {
            Ternary {
                predicate,
                truthy,
                falsy,
            } => AENodesIter::new_triple(predicate, falsy, truthy),
            _ => self.children_iter(),
        })
    }

    /// Iterator that retunrs the child nodes of this aexpr in field declaration order.
    ///
    /// This function and its users must be updated if the field declaration order changes.
    pub fn children_iter_mut(&mut self) -> AENodesIterMut<'_> {
        use std::slice;

        use AExpr::*;

        match self {
            Element | Column(_) | Literal(_) | Len => AENodesIterMut::new_empty(),
            #[cfg(feature = "dtype-struct")]
            StructField(_) => AENodesIterMut::new_empty(),

            Cast {
                expr,
                dtype: _,
                options: _,
            }
            | Sort { expr, options: _ }
            | Explode { expr, options: _ } => AENodesIterMut::new_single(expr),

            Gather {
                expr,
                idx,
                returns_scalar: _,
                null_on_oob: _,
            } => AENodesIterMut::new_double(expr, idx),
            SortBy { expr, by, .. } => {
                AENodesIterMut::DoubleSlice(slice::from_mut(expr).iter_mut().chain(by.iter_mut()))
            },
            Filter { input, by } => AENodesIterMut::new_double(input, by),
            Agg(agg) => agg.children_iter_mut(),
            BinaryExpr { left, op: _, right } => AENodesIterMut::new_double(left, right),
            Ternary {
                predicate,
                truthy,
                falsy,
            } => AENodesIterMut::new_triple(predicate, truthy, falsy),
            AnonymousFunction { input, .. }
            | Function { input, .. }
            | AnonymousAgg { input, .. } => {
                AENodesIterMut::ExprIRSlice(input.iter_mut().map(ExprIR::node_mut))
            },

            #[cfg(feature = "dynamic_group_by")]
            Rolling {
                function,
                index_column,
                period: _,
                offset: _,
                closed_window: _,
            } => AENodesIterMut::new_double(function, index_column),
            Over {
                function,
                partition_by,
                order_by,
                mapping: _,
            } => AENodesIterMut::TripleSlice(
                std::slice::from_mut(function)
                    .iter_mut()
                    .chain(partition_by.iter_mut())
                    .chain(
                        order_by
                            .as_mut()
                            .map_or(&mut [][..], |x| std::slice::from_mut(&mut x.0)),
                    ),
            ),
            Eval {
                expr,
                evaluation,
                variant: _,
            } => AENodesIterMut::new_double(expr, evaluation),
            #[cfg(feature = "dtype-struct")]
            StructEval { expr, evaluation } => AENodesIterMut::StructEval(
                std::iter::once(expr).chain(evaluation.iter_mut().map(&ExprIR::node_mut as _)),
            ),
            Slice {
                input,
                offset,
                length,
            } => AENodesIterMut::new_triple(input, offset, length),
        }
    }

    /// Iterator that returns nodes in order such that the last item of the iterator is the node
    /// from which the output name is sourced. The ordering of non-name nodes is unspecified but
    /// guaranteed to match `AExpr::children_iter_mut_name`.
    pub fn children_iter_mut_name_last(&mut self) -> std::iter::Rev<AENodesIterMut<'_>> {
        use AExpr::*;

        Iterator::rev(match self {
            Ternary {
                predicate,
                truthy,
                falsy,
            } => AENodesIterMut::new_triple(predicate, falsy, truthy),
            _ => self.children_iter_mut(),
        })
    }

    /// Iterator that retunrs the child nodes of this aexpr in field declaration order.
    ///
    /// This is derived from `children_iter`, but skips list / struct eval exprs.
    ///
    /// This function and its users must be updated if the field declaration order changes.
    pub fn inputs_iter(&self) -> AENodesIter<'_> {
        use AExpr::*;

        match self {
            #[cfg(feature = "dtype-struct")]
            StructEval {
                expr,
                evaluation: _,
            } => AENodesIter::new_single(expr),
            Eval {
                expr,
                evaluation: _,
                variant: _,
            } => AENodesIter::new_single(expr),
            ae => ae.children_iter(),
        }
    }

    /// Iterator that retunrs the child nodes of this aexpr in field declaration order.
    ///
    /// This is derived from `children_iter`, but skips list / struct eval exprs.
    ///
    /// This function and its users must be updated if the field declaration order changes.
    pub fn inputs_iter_mut(&mut self) -> AENodesIterMut<'_> {
        use AExpr::*;

        match self {
            #[cfg(feature = "dtype-struct")]
            StructEval {
                expr,
                evaluation: _,
            } => AENodesIterMut::new_single(expr),
            Eval {
                expr,
                evaluation: _,
                variant: _,
            } => AENodesIterMut::new_single(expr),
            ae => ae.children_iter_mut(),
        }
    }

    /// Iterator that returns nodes in order such that the last item of the iterator is the node
    /// from which the output name is sourced. The ordering of non-name nodes is unspecified but
    /// guaranteed to match `AExpr::inputs_iter_mut_name_last`.
    ///
    /// This is derived from `children_iter_name_last`, but skips list / struct eval exprs.
    pub fn inputs_iter_name_last(&self) -> std::iter::Rev<AENodesIter<'_>> {
        use AExpr::*;

        Iterator::rev(match self {
            Ternary {
                predicate,
                truthy,
                falsy,
            } => AENodesIter::new_triple(predicate, falsy, truthy),
            _ => self.inputs_iter(),
        })
    }

    /// Iterator that returns nodes in order such that the last item of the iterator is the node
    /// from which the output name is sourced. The ordering of non-name nodes is unspecified but
    /// guaranteed to match `AExpr::inputs_iter_name_last`.
    ///
    /// This is derived from `children_iter_name_last`, but skips list / struct eval exprs.
    pub fn inputs_iter_mut_name_last(&mut self) -> std::iter::Rev<AENodesIterMut<'_>> {
        use AExpr::*;

        Iterator::rev(match self {
            Ternary {
                predicate,
                truthy,
                falsy,
            } => AENodesIterMut::new_triple(predicate, falsy, truthy),
            _ => self.inputs_iter_mut(),
        })
    }

    /// Replace the inputs of this AExpr. This excludes the list / struct eval exprs.
    ///
    /// # Panics
    /// Panics if the number of provided inputs does not match the number of inputs in this AExpr.
    pub fn replace_inputs(&mut self, inputs: impl IntoIterator<Item = Node>) {
        for (l, r) in self.inputs_iter_mut().zip_eq(inputs) {
            *l = r;
        }
    }

    /// Replace the children of this AExpr.
    ///
    /// # Panics
    /// Panics if the number of provided child nodes does not match the number of child nodes in this AExpr.
    pub fn replace_children(&mut self, children: impl IntoIterator<Item = Node>) {
        for (l, r) in self.children_iter_mut().zip_eq(children) {
            *l = r;
        }
    }
}

impl IRAggExpr {
    pub fn children_iter(&self) -> AENodesIter<'_> {
        use IRAggExpr::*;

        match self {
            Min {
                input,
                propagate_nans: _,
            }
            | Max {
                input,
                propagate_nans: _,
            }
            | Median(input)
            | NUnique(input)
            | First(input)
            | FirstNonNull(input)
            | Last(input)
            | LastNonNull(input)
            | Item {
                input,
                allow_empty: _,
            }
            | Mean(input)
            | Implode {
                input,
                maintain_order: _,
            }
            | Sum(input)
            | Count {
                input,
                include_nulls: _,
            }
            | Std(input, _)
            | Var(input, _)
            | AggGroups(input) => AENodesIter::new_single(input),

            Quantile {
                expr,
                quantile,
                method: _,
            } => AENodesIter::new_double(expr, quantile),
        }
    }

    pub fn children_iter_mut(&mut self) -> AENodesIterMut<'_> {
        use IRAggExpr::*;

        match self {
            Min {
                input,
                propagate_nans: _,
            }
            | Max {
                input,
                propagate_nans: _,
            }
            | Median(input)
            | NUnique(input)
            | First(input)
            | FirstNonNull(input)
            | Last(input)
            | LastNonNull(input)
            | Item {
                input,
                allow_empty: _,
            }
            | Mean(input)
            | Implode {
                input,
                maintain_order: _,
            }
            | Sum(input)
            | Count {
                input,
                include_nulls: _,
            }
            | Std(input, _)
            | Var(input, _)
            | AggGroups(input) => AENodesIterMut::new_single(input),

            Quantile {
                expr,
                quantile,
                method: _,
            } => AENodesIterMut::new_double(expr, quantile),
        }
    }

    pub fn set_input(&mut self, input: Node) {
        use IRAggExpr::*;
        let node = match self {
            Min { input, .. } => input,
            Max { input, .. } => input,
            Median(input) => input,
            NUnique(input) => input,
            First(input) => input,
            FirstNonNull(input) => input,
            Last(input) => input,
            LastNonNull(input) => input,
            Item { input, .. } => input,
            Mean(input) => input,
            Implode { input, .. } => input,
            Quantile { expr, .. } => expr,
            Sum(input) => input,
            Count { input, .. } => input,
            Std(input, _) => input,
            Var(input, _) => input,
            AggGroups(input) => input,
        };
        *node = input;
    }
}

pub enum AENodesIter<'a> {
    Slice(std::slice::Iter<'a, Node>),
    DoubleSlice(std::iter::Chain<std::slice::Iter<'a, Node>, std::slice::Iter<'a, Node>>),
    TripleSlice(
        std::iter::Chain<
            std::iter::Chain<std::slice::Iter<'a, Node>, std::slice::Iter<'a, Node>>,
            std::slice::Iter<'a, Node>,
        >,
    ),
    ExprIRSlice(std::iter::Map<std::slice::Iter<'a, ExprIR>, fn(&'a ExprIR) -> &'a Node>),
    StructEval(
        std::iter::Chain<
            std::iter::Once<&'a Node>,
            std::iter::Map<std::slice::Iter<'a, ExprIR>, &'a dyn Fn(&'a ExprIR) -> &'a Node>,
        >,
    ),
}

impl<'a> AENodesIter<'a> {
    pub fn new_empty() -> Self {
        Self::Slice(Default::default())
    }

    pub fn new_single(node: &'a Node) -> Self {
        Self::Slice(std::slice::from_ref(node).iter())
    }

    pub fn new_double(node1: &'a Node, node2: &'a Node) -> Self {
        use std::slice::from_ref;

        Self::DoubleSlice(from_ref(node1).iter().chain(from_ref(node2)))
    }

    pub fn new_triple(node1: &'a Node, node2: &'a Node, node3: &'a Node) -> Self {
        use std::slice::from_ref;

        Self::TripleSlice(
            from_ref(node1)
                .iter()
                .chain(from_ref(node2))
                .chain(from_ref(node3)),
        )
    }
}

impl<'a> Iterator for AENodesIter<'a> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        use AENodesIter::*;
        match self {
            Slice(v) => v.next(),
            DoubleSlice(v) => v.next(),
            TripleSlice(v) => v.next(),
            ExprIRSlice(v) => v.next(),
            StructEval(v) => v.next(),
        }
        .copied()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        use AENodesIter::*;
        match self {
            Slice(v) => v.size_hint(),
            DoubleSlice(v) => v.size_hint(),
            TripleSlice(v) => v.size_hint(),
            ExprIRSlice(v) => v.size_hint(),
            StructEval(v) => v.size_hint(),
        }
    }
}

impl<'a> ExactSizeIterator for AENodesIter<'a> {}

impl<'a> DoubleEndedIterator for AENodesIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        use AENodesIter::*;
        match self {
            Slice(v) => v.next_back(),
            DoubleSlice(v) => v.next_back(),
            TripleSlice(v) => v.next_back(),
            ExprIRSlice(v) => v.next_back(),
            StructEval(v) => v.next_back(),
        }
        .copied()
    }
}

pub enum AENodesIterMut<'a> {
    Slice(std::slice::IterMut<'a, Node>),
    DoubleSlice(std::iter::Chain<std::slice::IterMut<'a, Node>, std::slice::IterMut<'a, Node>>),
    TripleSlice(
        std::iter::Chain<
            std::iter::Chain<std::slice::IterMut<'a, Node>, std::slice::IterMut<'a, Node>>,
            std::slice::IterMut<'a, Node>,
        >,
    ),
    ExprIRSlice(
        std::iter::Map<std::slice::IterMut<'a, ExprIR>, fn(&'a mut ExprIR) -> &'a mut Node>,
    ),
    StructEval(
        std::iter::Chain<
            std::iter::Once<&'a mut Node>,
            std::iter::Map<
                std::slice::IterMut<'a, ExprIR>,
                &'a dyn Fn(&'a mut ExprIR) -> &'a mut Node,
            >,
        >,
    ),
}

impl<'a> AENodesIterMut<'a> {
    pub fn new_empty() -> Self {
        Self::Slice(Default::default())
    }

    pub fn new_single(node: &'a mut Node) -> Self {
        Self::Slice(std::slice::from_mut(node).iter_mut())
    }

    pub fn new_double(node1: &'a mut Node, node2: &'a mut Node) -> Self {
        use std::slice::from_mut;

        Self::DoubleSlice(from_mut(node1).iter_mut().chain(from_mut(node2)))
    }

    pub fn new_triple(node1: &'a mut Node, node2: &'a mut Node, node3: &'a mut Node) -> Self {
        use std::slice::from_mut;

        Self::TripleSlice(
            from_mut(node1)
                .iter_mut()
                .chain(from_mut(node2))
                .chain(from_mut(node3)),
        )
    }
}

impl<'a> Iterator for AENodesIterMut<'a> {
    type Item = &'a mut Node;

    fn next(&mut self) -> Option<Self::Item> {
        use AENodesIterMut::*;
        match self {
            Slice(v) => v.next(),
            DoubleSlice(v) => v.next(),
            TripleSlice(v) => v.next(),
            ExprIRSlice(v) => v.next(),
            StructEval(v) => v.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        use AENodesIterMut::*;
        match self {
            Slice(v) => v.size_hint(),
            DoubleSlice(v) => v.size_hint(),
            TripleSlice(v) => v.size_hint(),
            ExprIRSlice(v) => v.size_hint(),
            StructEval(v) => v.size_hint(),
        }
    }
}

impl<'a> ExactSizeIterator for AENodesIterMut<'a> {}

impl<'a> DoubleEndedIterator for AENodesIterMut<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        use AENodesIterMut::*;
        match self {
            Slice(v) => v.next_back(),
            DoubleSlice(v) => v.next_back(),
            TripleSlice(v) => v.next_back(),
            ExprIRSlice(v) => v.next_back(),
            StructEval(v) => v.next_back(),
        }
    }
}

pub fn aexpr_tree_traversal<ArenaT, Edge, BreakValue>(
    root_ae_node: Node,
    expr_arena: &mut ArenaT,
    visit_stack: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    visitor: &mut dyn NodeVisitor<Key = Node, Storage = ArenaT, Edge = Edge, BreakValue = BreakValue>,
) -> ControlFlow<BreakValue, Edge>
where
    ArenaT: GetNodeInputs<Node>,
{
    tree_traversal(root_ae_node, expr_arena, visit_stack, edges, visitor)
}

impl GetNodeInputs<Node> for Arena<AExpr> {
    fn get_node_inputs(&self, key: Node, push_fn: &mut dyn FnMut(Node)) {
        for node in self.get(key).children_iter() {
            push_fn(node)
        }
    }

    fn num_inputs(&self, key: Node) -> usize {
        self.get(key).children_iter().len()
    }
}

impl GetNodeInputs<Node> for &Arena<AExpr> {
    fn get_node_inputs(&self, key: Node, push_fn: &mut dyn FnMut(Node)) {
        for node in self.get(key).children_iter() {
            push_fn(node)
        }
    }

    fn num_inputs(&self, key: Node) -> usize {
        self.get(key).children_iter().len()
    }
}

impl GetNodeInputs<Node> for Arena<IR> {
    fn get_node_inputs(&self, key: Node, push_fn: &mut dyn FnMut(Node)) {
        for v in self.get(key).inputs() {
            push_fn(v)
        }
    }

    fn num_inputs(&self, key: Node) -> usize {
        self.get(key).inputs().len()
    }
}

impl GetNodeInputs<Node> for &Arena<IR> {
    fn get_node_inputs(&self, key: Node, push_fn: &mut dyn FnMut(Node)) {
        for v in self.get(key).inputs() {
            push_fn(v)
        }
    }

    fn num_inputs(&self, key: Node) -> usize {
        self.get(key).inputs().len()
    }
}
