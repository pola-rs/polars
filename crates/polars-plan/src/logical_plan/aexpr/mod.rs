mod hash;
mod schema;

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::legacy::prelude::QuantileInterpolOptions;
use polars_core::frame::group_by::GroupByMethod;
use polars_core::prelude::*;
use polars_core::utils::{get_time_units, try_get_supertype};
use polars_utils::arena::{Arena, Node};
use strum_macros::IntoStaticStr;

use crate::dsl::function_expr::FunctionExpr;
#[cfg(feature = "cse")]
use crate::logical_plan::visitor::AexprNode;
use crate::logical_plan::Context;
use crate::prelude::consts::LEN;
use crate::prelude::*;

#[derive(Clone, Debug, IntoStaticStr)]
pub enum AAggExpr {
    Min {
        input: Node,
        propagate_nans: bool,
    },
    Max {
        input: Node,
        propagate_nans: bool,
    },
    Median(Node),
    NUnique(Node),
    First(Node),
    Last(Node),
    Mean(Node),
    Implode(Node),
    Quantile {
        expr: Node,
        quantile: Node,
        interpol: QuantileInterpolOptions,
    },
    Sum(Node),
    Count(Node, bool),
    Std(Node, u8),
    Var(Node, u8),
    AggGroups(Node),
}

impl Hash for AAggExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Min { propagate_nans, .. } | Self::Max { propagate_nans, .. } => {
                propagate_nans.hash(state)
            },
            Self::Quantile { interpol, .. } => interpol.hash(state),
            Self::Std(_, v) | Self::Var(_, v) => v.hash(state),
            _ => {},
        }
    }
}

impl AAggExpr {
    pub(super) fn equal_nodes(&self, other: &AAggExpr) -> bool {
        use AAggExpr::*;
        match (self, other) {
            (
                Min {
                    propagate_nans: l, ..
                },
                Min {
                    propagate_nans: r, ..
                },
            ) => l == r,
            (
                Max {
                    propagate_nans: l, ..
                },
                Max {
                    propagate_nans: r, ..
                },
            ) => l == r,
            (Quantile { interpol: l, .. }, Quantile { interpol: r, .. }) => l == r,
            (Std(_, l), Std(_, r)) => l == r,
            (Var(_, l), Var(_, r)) => l == r,
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

impl From<AAggExpr> for GroupByMethod {
    fn from(value: AAggExpr) -> Self {
        use AAggExpr::*;
        match value {
            Min { propagate_nans, .. } => {
                if propagate_nans {
                    GroupByMethod::NanMin
                } else {
                    GroupByMethod::Min
                }
            },
            Max { propagate_nans, .. } => {
                if propagate_nans {
                    GroupByMethod::NanMax
                } else {
                    GroupByMethod::Max
                }
            },
            Median(_) => GroupByMethod::Median,
            NUnique(_) => GroupByMethod::NUnique,
            First(_) => GroupByMethod::First,
            Last(_) => GroupByMethod::Last,
            Mean(_) => GroupByMethod::Mean,
            Implode(_) => GroupByMethod::Implode,
            Sum(_) => GroupByMethod::Sum,
            Count(_, include_nulls) => GroupByMethod::Count { include_nulls },
            Std(_, ddof) => GroupByMethod::Std(ddof),
            Var(_, ddof) => GroupByMethod::Var(ddof),
            AggGroups(_) => GroupByMethod::Groups,
            Quantile { .. } => unreachable!(),
        }
    }
}

// AExpr representation of Nodes which are allocated in an Arena
#[derive(Clone, Debug, Default)]
pub enum AExpr {
    Explode(Node),
    Alias(Node, Arc<str>),
    Column(Arc<str>),
    Literal(LiteralValue),
    BinaryExpr {
        left: Node,
        op: Operator,
        right: Node,
    },
    Cast {
        expr: Node,
        data_type: DataType,
        strict: bool,
    },
    Sort {
        expr: Node,
        options: SortOptions,
    },
    Gather {
        expr: Node,
        idx: Node,
        returns_scalar: bool,
    },
    SortBy {
        expr: Node,
        by: Vec<Node>,
        descending: Vec<bool>,
    },
    Filter {
        input: Node,
        by: Node,
    },
    Agg(AAggExpr),
    Ternary {
        predicate: Node,
        truthy: Node,
        falsy: Node,
    },
    AnonymousFunction {
        input: Vec<Node>,
        function: SpecialEq<Arc<dyn SeriesUdf>>,
        output_type: GetOutput,
        options: FunctionOptions,
    },
    Function {
        /// function arguments
        input: Vec<Node>,
        /// function to apply
        function: FunctionExpr,
        options: FunctionOptions,
    },
    Window {
        function: Node,
        partition_by: Vec<Node>,
        options: WindowType,
    },
    #[default]
    Wildcard,
    Slice {
        input: Node,
        offset: Node,
        length: Node,
    },
    Len,
    Nth(i64),
}

impl AExpr {
    #[cfg(feature = "cse")]
    pub(crate) fn is_equal(l: Node, r: Node, arena: &Arena<AExpr>) -> bool {
        let arena = arena as *const Arena<AExpr> as *mut Arena<AExpr>;
        // safety: we can pass a *mut pointer
        // the equality operation will not access mutable
        unsafe {
            let ae_node_l = AexprNode::from_raw(l, arena);
            let ae_node_r = AexprNode::from_raw(r, arena);
            ae_node_l == ae_node_r
        }
    }

    #[cfg(feature = "cse")]
    pub(crate) fn col(name: &str) -> Self {
        AExpr::Column(Arc::from(name))
    }
    /// Any expression that is sensitive to the number of elements in a group
    /// - Aggregations
    /// - Sorts
    /// - Counts
    /// - ..
    pub(crate) fn groups_sensitive(&self) -> bool {
        use AExpr::*;
        match self {
            Function { options, .. } | AnonymousFunction { options, .. } => {
                options.is_groups_sensitive()
            }
            Sort { .. }
            | SortBy { .. }
            | Agg { .. }
            | Window { .. }
            | Len
            | Slice { .. }
            | Gather { .. }
            | Nth(_)
             => true,
            Alias(_, _)
            | Explode(_)
            | Column(_)
            | Literal(_)
            // a caller should traverse binary and ternary
            // to determine if the whole expr. is group sensitive
            | BinaryExpr { .. }
            | Ternary { .. }
            | Wildcard
            | Cast { .. }
            | Filter { .. } => false,
        }
    }

    /// This should be a 1 on 1 copy of the get_type method of Expr until Expr is completely phased out.
    pub fn get_type(
        &self,
        schema: &Schema,
        ctxt: Context,
        arena: &Arena<AExpr>,
    ) -> PolarsResult<DataType> {
        self.to_field(schema, ctxt, arena)
            .map(|f| f.data_type().clone())
    }

    /// Push nodes at this level to a pre-allocated stack
    pub(crate) fn nodes(&self, container: &mut Vec<Node>) {
        use AExpr::*;

        match self {
            Nth(_) | Column(_) | Literal(_) | Wildcard | Len => {},
            Alias(e, _) => container.push(*e),
            BinaryExpr { left, op: _, right } => {
                // reverse order so that left is popped first
                container.push(*right);
                container.push(*left);
            },
            Cast { expr, .. } => container.push(*expr),
            Sort { expr, .. } => container.push(*expr),
            Gather { expr, idx, .. } => {
                container.push(*idx);
                // latest, so that it is popped first
                container.push(*expr);
            },
            SortBy { expr, by, .. } => {
                for node in by {
                    container.push(*node)
                }
                // latest, so that it is popped first
                container.push(*expr);
            },
            Filter { input, by } => {
                container.push(*by);
                // latest, so that it is popped first
                container.push(*input);
            },
            Agg(agg_e) => match agg_e.get_input() {
                NodeInputs::Single(node) => container.push(node),
                NodeInputs::Many(nodes) => container.extend_from_slice(&nodes),
                NodeInputs::Leaf => {},
            },
            Ternary {
                truthy,
                falsy,
                predicate,
            } => {
                container.push(*predicate);
                container.push(*falsy);
                // latest, so that it is popped first
                container.push(*truthy);
            },
            AnonymousFunction { input, .. } | Function { input, .. } =>
            // we iterate in reverse order, so that the lhs is popped first and will be found
            // as the root columns/ input columns by `_suffix` and `_keep_name` etc.
            {
                input
                    .iter()
                    .rev()
                    .copied()
                    .for_each(|node| container.push(node))
            },
            Explode(e) => container.push(*e),
            Window {
                function,
                partition_by,
                options: _,
            } => {
                for e in partition_by.iter().rev() {
                    container.push(*e);
                }
                // latest so that it is popped first
                container.push(*function);
            },
            Slice {
                input,
                offset,
                length,
            } => {
                container.push(*length);
                container.push(*offset);
                // latest so that it is popped first
                container.push(*input);
            },
        }
    }

    pub(crate) fn replace_inputs(mut self, inputs: &[Node]) -> Self {
        use AExpr::*;
        let input = match &mut self {
            Column(_) | Literal(_) | Wildcard | Len | Nth(_) => return self,
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
                    AAggExpr::Quantile { expr, quantile, .. } => {
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
                input.clear();
                input.extend(inputs.iter().rev().copied());
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
                ..
            } => {
                *function = *inputs.last().unwrap();
                partition_by.clear();
                partition_by.extend_from_slice(&inputs[..inputs.len() - 1]);

                return self;
            },
        };
        *input = inputs[0];
        self
    }

    pub(crate) fn is_leaf(&self) -> bool {
        matches!(
            self,
            AExpr::Column(_) | AExpr::Literal(_) | AExpr::Len | AExpr::Nth(_)
        )
    }
}

impl AAggExpr {
    pub fn get_input(&self) -> NodeInputs {
        use AAggExpr::*;
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
        use AAggExpr::*;
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
