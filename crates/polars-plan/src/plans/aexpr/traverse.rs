use super::*;

impl AExpr {
    /// Push the inputs of this node to the given container, in reverse order.
    /// This ensures the primary node responsible for the name is pushed last.
    pub(crate) fn inputs_rev<E>(&self, container: &mut E)
    where
        E: Extend<Node>,
    {
        use AExpr::*;

        match self {
            Column(_) | Literal(_) | Len => {},
            Alias(e, _) => container.extend([*e]),
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
            AnonymousFunction { input, .. } | Function { input, .. } => {
                container.extend(input.iter().rev().map(|e| e.node()))
            },
            Explode(e) => container.extend([*e]),
            Window {
                function,
                partition_by,
                order_by,
                options: _,
            } => {
                if let Some((n, _)) = order_by {
                    container.extend([*n]);
                }
                container.extend(partition_by.iter().rev().cloned());
                container.extend([*function]);
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
            Column(_) | Literal(_) | Len => return self,
            Alias(input, _) => input,
            Cast { expr, .. } => expr,
            Explode(input) => input,
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
                    IRAggExpr::Quantile { expr, quantile, .. } => {
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
            AnonymousFunction { input, .. } | Function { input, .. } => {
                assert_eq!(input.len(), inputs.len());
                for (e, node) in input.iter_mut().zip(inputs.iter()) {
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
            Window {
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
            Last(input) => Single(*input),
            Mean(input) => Single(*input),
            Implode(input) => Single(*input),
            Quantile { expr, quantile, .. } => Many(vec![*expr, *quantile]),
            Sum(input) => Single(*input),
            Count(input, _) => Single(*input),
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
            Last(input) => input,
            Mean(input) => input,
            Implode(input) => input,
            Quantile { expr, .. } => expr,
            Sum(input) => input,
            Count(input, _) => input,
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
