mod schema;

use std::sync::Arc;

use polars_arrow::prelude::QuantileInterpolOptions;
use polars_core::prelude::*;
use polars_core::utils::{get_time_units, try_get_supertype};
use polars_utils::arena::{Arena, Node};

use crate::dsl::function_expr::FunctionExpr;
use crate::logical_plan::Context;
use crate::prelude::aexpr::NodeInputs::Single;
use crate::prelude::names::COUNT;
use crate::prelude::*;

#[derive(Clone, Debug)]
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
    List(Node),
    Quantile {
        expr: Node,
        quantile: Node,
        interpol: QuantileInterpolOptions,
    },
    Sum(Node),
    Count(Node),
    Std(Node, u8),
    Var(Node, u8),
    AggGroups(Node),
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
    Take {
        expr: Node,
        idx: Node,
    },
    SortBy {
        expr: Node,
        by: Vec<Node>,
        reverse: Vec<bool>,
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
        order_by: Option<Node>,
        options: WindowOptions,
    },
    #[default]
    Wildcard,
    Slice {
        input: Node,
        offset: Node,
        length: Node,
    },
    Count,
    Nth(i64),
}

impl AExpr {
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
            | Count
            | Slice { .. }
            | Take { .. }
            | Nth(_)
             => true,
            | Alias(_, _)
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

    pub(crate) fn replace_input(self, input: Node) -> Self {
        use AExpr::*;
        match self {
            Alias(_, name) => Alias(input, name),
            Cast {
                expr: _,
                data_type,
                strict,
            } => Cast {
                expr: input,
                data_type,
                strict,
            },
            _ => todo!(),
        }
    }

    pub(crate) fn get_input(&self) -> NodeInputs {
        use AExpr::*;
        use NodeInputs::*;
        match self {
            Alias(input, _) => Single(*input),
            Cast { expr, .. } => Single(*expr),
            Explode(input) => Single(*input),
            Column(_) => Leaf,
            Literal(_) => Leaf,
            BinaryExpr { left, right, .. } => Many(vec![*left, *right]),
            Sort { expr, .. } => Single(*expr),
            Take { expr, .. } => Single(*expr),
            SortBy { expr, by, .. } => {
                let mut many = by.clone();
                many.push(*expr);
                Many(many)
            }
            Filter { input, .. } => Single(*input),
            Agg(a) => a.get_input(),
            Ternary {
                truthy,
                falsy,
                predicate,
            } => Many(vec![*truthy, *falsy, *predicate]),
            AnonymousFunction { input, .. } | Function { input, .. } => match input.len() {
                1 => Single(input[0]),
                _ => Many(input.clone()),
            },
            Window {
                function,
                order_by,
                partition_by,
                ..
            } => {
                let mut out = Vec::with_capacity(partition_by.len() + 2);
                out.push(*function);
                if let Some(a) = order_by {
                    out.push(*a);
                }
                out.extend(partition_by);
                Many(out)
            }
            Wildcard => panic!("no wildcard expected"),
            Slice { input, .. } => Single(*input),
            Count => Leaf,
            Nth(_) => Leaf,
        }
    }
}

impl AAggExpr {
    pub(crate) fn get_input(&self) -> NodeInputs {
        use AAggExpr::*;
        match self {
            Min { input, .. } => Single(*input),
            Max { input, .. } => Single(*input),
            Median(input) => Single(*input),
            NUnique(input) => Single(*input),
            First(input) => Single(*input),
            Last(input) => Single(*input),
            Mean(input) => Single(*input),
            List(input) => Single(*input),
            Quantile { expr, .. } => Single(*expr),
            Sum(input) => Single(*input),
            Count(input) => Single(*input),
            Std(input, _) => Single(*input),
            Var(input, _) => Single(*input),
            AggGroups(input) => Single(*input),
        }
    }
}

pub(crate) enum NodeInputs {
    Leaf,
    Single(Node),
    Many(Vec<Node>),
}

impl NodeInputs {
    pub(crate) fn first(&self) -> Node {
        match self {
            Single(node) => *node,
            NodeInputs::Many(nodes) => nodes[0],
            NodeInputs::Leaf => panic!(),
        }
    }
}
