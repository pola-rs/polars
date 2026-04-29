use std::ops::ControlFlow;

use super::*;
use crate::traversal::tree_traversal::{GetNodeInputs, tree_traversal};
use crate::traversal::visitor::NodeVisitor;

impl AExpr {
    /// Push the inputs of this node to the given container, in field declaration order.
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
                    .chain(partition_by.as_slice())
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
                std::iter::once(expr)
                    .chain(evaluation.as_slice().iter().map(&ExprIR::node_ref as _)),
            ),
            Slice {
                input,
                offset,
                length,
            } => AENodesIter::new_triple(input, offset, length),
        }
    }

    /// Push the inputs of this node to the given container, in reverse order.
    /// This ensures the primary node responsible for the name is pushed last.
    ///
    /// This is subtly different from `children_rev` as this only includes the input expressions,
    /// not expressions used during evaluation.
    pub fn inputs_rev<E>(&self, container: &mut E)
    where
        E: Extend<Node>,
    {
        use AExpr::*;

        match self {
            Element | Column(_) | Literal(_) | Len => {},
            #[cfg(feature = "dtype-struct")]
            StructField(_) => {},
            BinaryExpr { left, op: _, right } => {
                container.extend([*right, *left]);
            },
            Cast { expr, .. } => container.extend([*expr]),
            Sort { expr, .. } => container.extend([*expr]),
            Gather { expr, idx, .. } => {
                container.extend([*idx, *expr]);
            },
            SortBy { expr, by, .. } => {
                container.extend(by.iter().cloned().rev());
                container.extend([*expr]);
            },
            Filter { input, by } => {
                container.extend([*by, *input]);
            },
            Agg(agg) => container.extend(agg.children_iter().rev().copied()),
            Ternary {
                truthy,
                falsy,
                predicate,
            } => {
                container.extend([*predicate, *falsy, *truthy]);
            },
            AnonymousFunction { input, .. }
            | Function { input, .. }
            | AnonymousAgg { input, .. } => container.extend(input.iter().rev().map(|e| e.node())),
            Explode { expr: e, .. } => container.extend([*e]),
            #[cfg(feature = "dynamic_group_by")]
            Rolling {
                function,
                index_column,
                period: _,
                offset: _,
                closed_window: _,
            } => {
                container.extend([*index_column, *function]);
            },
            Over {
                function,
                partition_by,
                order_by,
                mapping: _,
            } => {
                if let Some((n, _)) = order_by {
                    container.extend([*n]);
                }
                container.extend(partition_by.iter().rev().cloned());
                container.extend([*function]);
            },
            Eval {
                expr,
                evaluation,
                variant: _,
            } => {
                // We don't use the evaluation here because it does not contain inputs.
                _ = evaluation;
                container.extend([*expr]);
            },
            #[cfg(feature = "dtype-struct")]
            StructEval { expr, evaluation } => {
                // Evaluation is included. In case this is not allowed, use `inputs_rev_strict()`.
                container.extend(evaluation.iter().rev().map(ExprIR::node));
                container.extend([*expr]);
            },
            Slice {
                input,
                offset,
                length,
            } => {
                container.extend([*length, *offset, *input]);
            },
        }
    }

    /// Push the inputs of this node to the given container, in reverse order.
    /// This ensures the primary node responsible for the name is pushed last.
    ///
    /// Unlike `inputs_rev`, this excludes Eval expressions. These use an extended schema,
    /// determined by their input, which implies a different traversal order.
    ///
    /// This is subtly different from `children_rev` as this only includes the input expressions,
    /// not expressions used during evaluation.
    pub fn inputs_rev_strict<E>(&self, container: &mut E)
    where
        E: Extend<Node>,
    {
        use AExpr::*;

        match self {
            #[cfg(feature = "dtype-struct")]
            StructEval { expr, evaluation } => {
                // Evaluation is explicitly excluded. It is up to the caller to handle
                // any tree traversal if required.
                _ = evaluation;
                container.extend([*expr]);
            },
            expr => expr.inputs_rev(container),
        }
    }

    /// Push the children of this node to the given container, in reverse order.
    /// This ensures the primary node responsible for the name is pushed last.
    ///
    /// This is subtly different from `input_rev` as this only all expressions included in the
    /// expression not only the input expressions,
    pub fn children_rev<E: Extend<Node>>(&self, container: &mut E) {
        use AExpr::*;

        match self {
            Element | Column(_) | Literal(_) | Len => {},
            #[cfg(feature = "dtype-struct")]
            StructField(_) => {},
            BinaryExpr { left, op: _, right } => {
                container.extend([*right, *left]);
            },
            Cast { expr, .. } => container.extend([*expr]),
            Sort { expr, .. } => container.extend([*expr]),
            Gather { expr, idx, .. } => {
                container.extend([*idx, *expr]);
            },
            SortBy { expr, by, .. } => {
                container.extend(by.iter().cloned().rev());
                container.extend([*expr]);
            },
            Filter { input, by } => {
                container.extend([*by, *input]);
            },
            Agg(agg) => container.extend(agg.children_iter().rev().copied()),
            Ternary {
                truthy,
                falsy,
                predicate,
            } => {
                container.extend([*predicate, *falsy, *truthy]);
            },
            AnonymousFunction { input, .. }
            | Function { input, .. }
            | AnonymousAgg { input, .. } => container.extend(input.iter().rev().map(|e| e.node())),
            Explode { expr: e, .. } => container.extend([*e]),
            #[cfg(feature = "dynamic_group_by")]
            Rolling {
                function,
                index_column,
                period: _,
                offset: _,
                closed_window: _,
            } => {
                container.extend([*index_column, *function]);
            },
            Over {
                function,
                partition_by,
                order_by,
                mapping: _,
            } => {
                if let Some((n, _)) = order_by {
                    container.extend([*n]);
                }
                container.extend(partition_by.iter().rev().cloned());
                container.extend([*function]);
            },
            Eval {
                expr,
                evaluation,
                variant: _,
            } => container.extend([*evaluation, *expr]),
            #[cfg(feature = "dtype-struct")]
            StructEval { expr, evaluation } => {
                container.extend(evaluation.iter().rev().map(ExprIR::node));
                container.extend([*expr]);
            },
            Slice {
                input,
                offset,
                length,
            } => {
                container.extend([*length, *offset, *input]);
            },
        }
    }

    pub fn replace_inputs(mut self, inputs: &[Node]) -> Self {
        use AExpr::*;
        let input = match &mut self {
            Element | Column(_) | Literal(_) | Len => return self,
            #[cfg(feature = "dtype-struct")]
            StructField(_) => return self,
            Cast { expr, .. } => expr,
            Explode { expr, .. } => expr,
            BinaryExpr { left, right, .. } => {
                *left = inputs[0];
                *right = inputs[1];
                return self;
            },
            Gather { expr, idx, .. } => {
                *expr = inputs[0];
                *idx = inputs[1];
                return self;
            },
            Sort { expr, .. } => expr,
            SortBy { expr, by, .. } => {
                *expr = inputs[0];
                by.clear();
                by.extend_from_slice(&inputs[1..]);
                return self;
            },
            Filter { input, by, .. } => {
                *input = inputs[0];
                *by = inputs[1];
                return self;
            },
            Agg(a) => {
                match a {
                    IRAggExpr::Quantile {
                        expr,
                        quantile,
                        method: _,
                    } => {
                        *expr = inputs[0];
                        *quantile = inputs[1];
                    },
                    _ => {
                        a.set_input(inputs[0]);
                    },
                }
                return self;
            },
            Ternary {
                truthy,
                falsy,
                predicate,
            } => {
                *truthy = inputs[0];
                *falsy = inputs[1];
                *predicate = inputs[2];
                return self;
            },
            AnonymousFunction { input, .. }
            | Function { input, .. }
            | AnonymousAgg { input, .. } => {
                assert_eq!(input.len(), inputs.len());
                for (e, node) in input.iter_mut().zip(inputs.iter()) {
                    e.set_node(*node);
                }
                return self;
            },
            Eval {
                expr,
                evaluation,
                variant: _,
            } => {
                *expr = inputs[0];
                _ = evaluation; // Intentional.
                return self;
            },
            #[cfg(feature = "dtype-struct")]
            StructEval { expr, evaluation } => {
                *expr = inputs[0];
                _ = evaluation; // Intentional.
                return self;
            },
            Slice {
                input,
                offset,
                length,
            } => {
                *input = inputs[0];
                *offset = inputs[1];
                *length = inputs[2];
                return self;
            },
            #[cfg(feature = "dynamic_group_by")]
            Rolling {
                function,
                index_column,
                period: _,
                offset: _,
                closed_window: _,
            } => {
                *function = inputs[0];
                *index_column = inputs[1];
                return self;
            },
            Over {
                function,
                partition_by,
                order_by,
                ..
            } => {
                let offset = order_by.is_some() as usize;
                *function = inputs[0];
                partition_by.clear();
                partition_by.extend_from_slice(&inputs[1..inputs.len() - offset]);
                if let Some((_, options)) = order_by {
                    *order_by = Some((*inputs.last().unwrap(), *options));
                }
                return self;
            },
        };
        *input = inputs[0];
        self
    }

    pub fn replace_children(mut self, inputs: &[Node]) -> Self {
        use AExpr::*;
        let input = match &mut self {
            Element | Column(_) | Literal(_) | Len => return self,
            #[cfg(feature = "dtype-struct")]
            StructField(_) => return self,
            Cast { expr, .. } => expr,
            Explode { expr, .. } => expr,
            BinaryExpr { left, right, .. } => {
                *left = inputs[0];
                *right = inputs[1];
                return self;
            },
            Gather { expr, idx, .. } => {
                *expr = inputs[0];
                *idx = inputs[1];
                return self;
            },
            Sort { expr, .. } => expr,
            SortBy { expr, by, .. } => {
                *expr = inputs[0];
                by.clear();
                by.extend_from_slice(&inputs[1..]);
                return self;
            },
            Filter { input, by, .. } => {
                *input = inputs[0];
                *by = inputs[1];
                return self;
            },
            Agg(a) => {
                if let IRAggExpr::Quantile {
                    expr,
                    quantile,
                    method: _,
                } = a
                {
                    *expr = inputs[0];
                    *quantile = inputs[1];
                } else {
                    a.set_input(inputs[0]);
                }
                return self;
            },
            Ternary {
                truthy,
                falsy,
                predicate,
            } => {
                *truthy = inputs[0];
                *falsy = inputs[1];
                *predicate = inputs[2];
                return self;
            },
            AnonymousAgg { input, .. }
            | AnonymousFunction { input, .. }
            | Function { input, .. } => {
                assert_eq!(input.len(), inputs.len());
                for (e, node) in input.iter_mut().zip(inputs.iter()) {
                    e.set_node(*node);
                }
                return self;
            },
            Eval {
                expr,
                evaluation,
                variant: _,
            } => {
                *expr = inputs[0];
                *evaluation = inputs[1];
                return self;
            },
            #[cfg(feature = "dtype-struct")]
            StructEval { expr, evaluation } => {
                assert_eq!(inputs.len(), evaluation.len() + 1);
                *expr = inputs[0];
                for (e, node) in evaluation.iter_mut().zip(inputs[1..].iter()) {
                    e.set_node(*node);
                }
                return self;
            },
            Slice {
                input,
                offset,
                length,
            } => {
                *input = inputs[0];
                *offset = inputs[1];
                *length = inputs[2];
                return self;
            },
            #[cfg(feature = "dynamic_group_by")]
            Rolling {
                function,
                index_column,
                period: _,
                offset: _,
                closed_window: _,
            } => {
                *function = inputs[0];
                *index_column = inputs[1];
                return self;
            },
            Over {
                function,
                partition_by,
                order_by,
                ..
            } => {
                let offset = order_by.is_some() as usize;
                *function = inputs[0];
                partition_by.clear();
                partition_by.extend_from_slice(&inputs[1..inputs.len() - offset]);
                if let Some((_, options)) = order_by {
                    *order_by = Some((*inputs.last().unwrap(), *options));
                }
                return self;
            },
        };
        *input = inputs[0];
        self
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
    type Item = &'a Node;

    fn next(&mut self) -> Option<Self::Item> {
        use AENodesIter::*;
        match self {
            Slice(v) => v.next(),
            DoubleSlice(v) => v.next(),
            TripleSlice(v) => v.next(),
            ExprIRSlice(v) => v.next(),
            StructEval(v) => v.next(),
        }
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

struct ExtendWrap<'a, T>(&'a mut dyn FnMut(T));

impl<'a, T> Extend<T> for ExtendWrap<'a, T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for v in iter.into_iter() {
            (self.0)(v)
        }
    }
}

impl GetNodeInputs<Node> for Arena<AExpr> {
    fn get_node_inputs(&self, key: Node, push_fn: &mut dyn FnMut(Node)) {
        for node in self.get(key).children_iter() {
            push_fn(*node)
        }
    }

    fn num_inputs(&self, key: Node) -> usize {
        self.get(key).children_iter().len()
    }
}

impl GetNodeInputs<Node> for &Arena<AExpr> {
    fn get_node_inputs(&self, key: Node, push_fn: &mut dyn FnMut(Node)) {
        for node in self.get(key).children_iter() {
            push_fn(*node)
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
