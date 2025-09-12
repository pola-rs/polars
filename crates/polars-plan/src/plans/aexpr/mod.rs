mod builder;
mod equality;
mod evaluate;
mod function_expr;
#[cfg(feature = "cse")]
mod hash;
mod minterm_iter;
pub mod predicates;
mod scalar;
mod schema;
mod traverse;

use std::hash::{Hash, Hasher};

pub use function_expr::*;
#[cfg(feature = "cse")]
pub(super) use hash::traverse_and_hash_aexpr;
pub use minterm_iter::MintermIter;
use polars_compute::rolling::QuantileMethod;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_core::utils::{get_time_units, try_get_supertype};
use polars_utils::arena::{Arena, Node};
pub use scalar::is_scalar_ae;
use strum_macros::IntoStaticStr;
pub use traverse::*;
mod properties;
pub use aexpr::function_expr::schema::FieldsMapper;
pub use builder::AExprBuilder;
pub use properties::*;

use crate::constants::LEN;
use crate::plans::Context;
use crate::prelude::*;

#[derive(Clone, Debug, IntoStaticStr)]
#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IRAggExpr {
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
        method: QuantileMethod,
    },
    Sum(Node),
    Count {
        input: Node,
        include_nulls: bool,
    },
    Std(Node, u8),
    Var(Node, u8),
    AggGroups(Node),
}

impl Hash for IRAggExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Min {
                input: _,
                propagate_nans,
            }
            | Self::Max {
                input: _,
                propagate_nans,
            } => propagate_nans.hash(state),
            Self::Quantile {
                method: interpol, ..
            } => interpol.hash(state),
            Self::Std(_, v) | Self::Var(_, v) => v.hash(state),
            Self::Count {
                input: _,
                include_nulls,
            } => include_nulls.hash(state),
            _ => {},
        }
    }
}

impl IRAggExpr {
    pub(super) fn equal_nodes(&self, other: &IRAggExpr) -> bool {
        use IRAggExpr::*;
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
            (Quantile { method: l, .. }, Quantile { method: r, .. }) => l == r,
            (Std(_, l), Std(_, r)) => l == r,
            (Var(_, l), Var(_, r)) => l == r,
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

impl From<IRAggExpr> for GroupByMethod {
    fn from(value: IRAggExpr) -> Self {
        use IRAggExpr::*;
        match value {
            Min {
                input: _,
                propagate_nans,
            } => {
                if propagate_nans {
                    GroupByMethod::NanMin
                } else {
                    GroupByMethod::Min
                }
            },
            Max {
                input: _,
                propagate_nans,
            } => {
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
            Count {
                input: _,
                include_nulls,
            } => GroupByMethod::Count { include_nulls },
            Std(_, ddof) => GroupByMethod::Std(ddof),
            Var(_, ddof) => GroupByMethod::Var(ddof),
            AggGroups(_) => GroupByMethod::Groups,
            Quantile { .. } => unreachable!(),
        }
    }
}

/// IR expression node that is allocated in an [`Arena`][polars_utils::arena::Arena].
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AExpr {
    Explode {
        expr: Node,
        skip_empty: bool,
    },
    Column(PlSmallStr),
    Literal(LiteralValue),
    BinaryExpr {
        left: Node,
        op: Operator,
        right: Node,
    },
    Cast {
        expr: Node,
        dtype: DataType,
        options: CastOptions,
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
        sort_options: SortMultipleOptions,
    },
    Filter {
        input: Node,
        by: Node,
    },
    Agg(IRAggExpr),
    Ternary {
        predicate: Node,
        truthy: Node,
        falsy: Node,
    },
    AnonymousFunction {
        input: Vec<ExprIR>,
        function: OpaqueColumnUdf,
        options: FunctionOptions,
        fmt_str: Box<PlSmallStr>,
    },
    /// Evaluates the `evaluation` expression on the output of the `expr`.
    ///
    /// Consequently, `expr` is an input and `evaluation` is not and needs a different schema.
    Eval {
        expr: Node,

        /// An expression that is guaranteed to not contain any column reference beyond
        /// `pl.element()` which refers to `pl.col("")`.
        evaluation: Node,

        variant: EvalVariant,
    },
    Function {
        /// Function arguments
        /// Some functions rely on aliases,
        /// for instance assignment of struct fields.
        /// Therefor we need [`ExprIr`].
        input: Vec<ExprIR>,
        /// function to apply
        function: IRFunctionExpr,
        options: FunctionOptions,
    },
    Window {
        function: Node,
        partition_by: Vec<Node>,
        order_by: Option<(Node, SortOptions)>,
        options: WindowType,
    },
    Slice {
        input: Node,
        offset: Node,
        length: Node,
    },
    #[default]
    Len,
}

impl AExpr {
    #[cfg(feature = "cse")]
    pub(crate) fn col(name: PlSmallStr) -> Self {
        AExpr::Column(name)
    }

    /// This should be a 1 on 1 copy of the get_type method of Expr until Expr is completely phased out.
    pub fn get_dtype(&self, schema: &Schema, arena: &Arena<AExpr>) -> PolarsResult<DataType> {
        self.to_field(schema, arena).map(|f| f.dtype().clone())
    }

    #[recursive::recursive]
    pub fn is_scalar(&self, arena: &Arena<AExpr>) -> bool {
        match self {
            AExpr::Literal(lv) => lv.is_scalar(),
            AExpr::Function { options, input, .. }
            | AExpr::AnonymousFunction { options, input, .. } => {
                if options.flags.contains(FunctionFlags::RETURNS_SCALAR) {
                    true
                } else if options.is_elementwise()
                    || options.flags.contains(FunctionFlags::LENGTH_PRESERVING)
                {
                    input.iter().all(|e| e.is_scalar(arena))
                } else {
                    false
                }
            },
            AExpr::BinaryExpr { left, right, .. } => {
                is_scalar_ae(*left, arena) && is_scalar_ae(*right, arena)
            },
            AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                is_scalar_ae(*predicate, arena)
                    && is_scalar_ae(*truthy, arena)
                    && is_scalar_ae(*falsy, arena)
            },
            AExpr::Agg(_) | AExpr::Len => true,
            AExpr::Cast { expr, .. } => is_scalar_ae(*expr, arena),
            AExpr::Eval { expr, variant, .. } => match variant {
                EvalVariant::List => is_scalar_ae(*expr, arena),
                EvalVariant::Cumulative { .. } => is_scalar_ae(*expr, arena),
            },
            AExpr::Sort { expr, .. } => is_scalar_ae(*expr, arena),
            AExpr::Gather { returns_scalar, .. } => *returns_scalar,
            AExpr::SortBy { expr, .. } => is_scalar_ae(*expr, arena),
            AExpr::Window { function, .. } => is_scalar_ae(*function, arena),
            AExpr::Explode { .. }
            | AExpr::Column(_)
            | AExpr::Filter { .. }
            | AExpr::Slice { .. } => false,
        }
    }
}
