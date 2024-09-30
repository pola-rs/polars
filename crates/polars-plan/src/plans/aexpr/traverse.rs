use super::*;

impl AExpr {
    /// Push nodes at this level to a pre-allocated stack.
    pub(crate) fn nodes<C: PushNode>(&self, container: &mut C) {
        use AExpr::*;

        match self {
            Column(_) | Literal(_) | Len => {},
            Alias(e, _) => container.push_node(*e),
            BinaryExpr { left, op: _, right } => {
                // reverse order so that left is popped first
                container.push_node(*right);
                container.push_node(*left);
            },
            Cast { expr, .. } => container.push_node(*expr),
            Sort { expr, .. } => container.push_node(*expr),
            Gather { expr, idx, .. } => {
                container.push_node(*idx);
                // latest, so that it is popped first
                container.push_node(*expr);
            },
            SortBy { expr, by, .. } => {
                for node in by {
                    container.push_node(*node)
                }
                // latest, so that it is popped first
                container.push_node(*expr);
            },
            Filter { input, by } => {
                container.push_node(*by);
                // latest, so that it is popped first
                container.push_node(*input);
            },
            Agg(agg_e) => match agg_e.get_input() {
                NodeInputs::Single(node) => container.push_node(node),
                NodeInputs::Many(nodes) => container.extend_from_slice(&nodes),
                NodeInputs::Leaf => {},
            },
            Ternary {
                truthy,
                falsy,
                predicate,
            } => {
                container.push_node(*predicate);
                container.push_node(*falsy);
                // latest, so that it is popped first
                container.push_node(*truthy);
            },
            AnonymousFunction { input, .. } | Function { input, .. } =>
            // we iterate in reverse order, so that the lhs is popped first and will be found
            // as the root columns/ input columns by `_suffix` and `_keep_name` etc.
            {
                input
                    .iter()
                    .rev()
                    .for_each(|e| container.push_node(e.node()))
            },
            Explode(e) => container.push_node(*e),
            Window {
                function,
                partition_by,
                order_by,
                options: _,
            } => {
                if let Some((n, _)) = order_by {
                    container.push_node(*n);
                }
                for e in partition_by.iter().rev() {
                    container.push_node(*e);
                }
                // latest so that it is popped first
                container.push_node(*function);
            },
            Slice {
                input,
                offset,
                length,
            } => {
                container.push_node(*length);
                container.push_node(*offset);
                // latest so that it is popped first
                container.push_node(*input);
            },
        }
    }

    pub(crate) fn replace_inputs(mut self, inputs: &[Node]) -> Self {
        use AExpr::*;
        let input = match &mut self {
            Column(_) | Literal(_) | Len => return self,
            Alias(input, _) => input,
            Cast { expr, .. } => expr,
            Explode(input) => input,
            BinaryExpr { left, right, .. } => {
                *right = inputs[0];
                *left = inputs[1];
                return self;
            },
            Gather { expr, idx, .. } => {
                *idx = inputs[0];
                *expr = inputs[1];
                return self;
            },
            Sort { expr, .. } => expr,
            SortBy { expr, by, .. } => {
                *expr = *inputs.last().unwrap();
                by.clear();
                by.extend_from_slice(&inputs[..inputs.len() - 1]);
                return self;
            },
            Filter { input, by, .. } => {
                *by = inputs[0];
                *input = inputs[1];
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
                *predicate = inputs[0];
                *falsy = inputs[1];
                *truthy = inputs[2];
                return self;
            },
            AnonymousFunction { input, .. } | Function { input, .. } => {
                debug_assert_eq!(input.len(), inputs.len());

                // Assign in reverse order as that was the order in which nodes were extracted.
                for (e, node) in input.iter_mut().zip(inputs.iter().rev()) {
                    e.set_node(*node);
                }
                return self;
            },
            Slice {
                input,
                offset,
                length,
            } => {
                *length = inputs[0];
                *offset = inputs[1];
                *input = inputs[2];
                return self;
            },
            Window {
                function,
                partition_by,
                order_by,
                ..
            } => {
                let offset = order_by.is_some() as usize;
                *function = *inputs.last().unwrap();
                partition_by.clear();
                partition_by.extend_from_slice(&inputs[offset..inputs.len() - 1]);

                if let Some((_, options)) = order_by {
                    *order_by = Some((inputs[0], *options));
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
            #[cfg(feature = "bitwise")]
            Bitwise(input, _) => Single(*input),
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
            #[cfg(feature = "bitwise")]
            Bitwise(input, _) => input,
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
