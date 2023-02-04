mod schema;

use std::sync::Arc;

use polars_arrow::prelude::QuantileInterpolOptions;
use polars_core::prelude::*;
use polars_core::utils::{get_time_units, try_get_supertype};
use polars_utils::arena::{Arena, Node};

use crate::dsl::function_expr::FunctionExpr;
use crate::logical_plan::Context;
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

    pub(crate) fn get_input(&self) -> Node {
        use AExpr::*;
        match self {
            Alias(input, _) => *input,
            Cast { expr, .. } => *expr,
            _ => todo!(),
        }
    }
}
