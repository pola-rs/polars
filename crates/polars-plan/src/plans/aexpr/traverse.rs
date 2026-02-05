use super::*;

impl AExpr {
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
            Agg(agg_e) => match agg_e.get_input() {
                NodeInputs::Single(node) => container.extend([node]),
                NodeInputs::Many(nodes) => container.extend(nodes.into_iter().rev()),
                NodeInputs::Leaf => {},
            },
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
            Agg(agg_e) => match agg_e.get_input() {
                NodeInputs::Single(node) => container.extend([node]),
                NodeInputs::Many(nodes) => container.extend(nodes.into_iter().rev()),
                NodeInputs::Leaf => {},
            },
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
    pub fn get_input(&self) -> NodeInputs {
        use IRAggExpr::*;
        use NodeInputs::*;

        match self {
            Min { input, .. } => Single(*input),
            Max { input, .. } => Single(*input),
            Median(input) => Single(*input),
            NUnique(input) => Single(*input),
            First(input) => Single(*input),
            FirstNonNull(input) => Single(*input),
            Last(input) => Single(*input),
            LastNonNull(input) => Single(*input),
            Item { input, .. } => Single(*input),
            Mean(input) => Single(*input),
            Implode(input) => Single(*input),
            Quantile { expr, quantile, .. } => Many(vec![*expr, *quantile]),
            Sum(input) => Single(*input),
            Count { input, .. } => Single(*input),
            Std(input, _) => Single(*input),
            Var(input, _) => Single(*input),
            AggGroups(input) => Single(*input),
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
            Implode(input) => input,
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

pub enum NodeInputs {
    Leaf,
    Single(Node),
    Many(Vec<Node>),
}

impl NodeInputs {
    pub fn first(&self) -> Node {
        match self {
            NodeInputs::Single(node) => *node,
            NodeInputs::Many(nodes) => nodes[0],
            NodeInputs::Leaf => panic!(),
        }
    }
}
