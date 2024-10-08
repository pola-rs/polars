use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

use bytes::Bytes;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::error::feature_gated;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use super::expr_dyn_fn::*;
use crate::prelude::*;

#[derive(PartialEq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AggExpr {
    Min {
        input: Arc<Expr>,
        propagate_nans: bool,
    },
    Max {
        input: Arc<Expr>,
        propagate_nans: bool,
    },
    Median(Arc<Expr>),
    NUnique(Arc<Expr>),
    First(Arc<Expr>),
    Last(Arc<Expr>),
    Mean(Arc<Expr>),
    Implode(Arc<Expr>),
    // include_nulls
    Count(Arc<Expr>, bool),
    Quantile {
        expr: Arc<Expr>,
        quantile: Arc<Expr>,
        interpol: QuantileInterpolOptions,
    },
    Sum(Arc<Expr>),
    AggGroups(Arc<Expr>),
    Std(Arc<Expr>, u8),
    Var(Arc<Expr>, u8),
    #[cfg(feature = "bitwise")]
    Bitwise(Arc<Expr>, super::function_expr::BitwiseAggFunction),
}

impl AsRef<Expr> for AggExpr {
    fn as_ref(&self) -> &Expr {
        use AggExpr::*;
        match self {
            Min { input, .. } => input,
            Max { input, .. } => input,
            Median(e) => e,
            NUnique(e) => e,
            First(e) => e,
            Last(e) => e,
            Mean(e) => e,
            Implode(e) => e,
            Count(e, _) => e,
            Quantile { expr, .. } => expr,
            Sum(e) => e,
            AggGroups(e) => e,
            Std(e, _) => e,
            Var(e, _) => e,
            #[cfg(feature = "bitwise")]
            Bitwise(e, _) => e,
        }
    }
}

/// Expressions that can be used in various contexts.
///
/// Queries consist of multiple expressions.
/// When using the polars lazy API, don't construct an `Expr` directly; instead, create one using
/// the functions in the `polars_lazy::dsl` module. See that module's docs for more info.
#[derive(Clone, PartialEq)]
#[must_use]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Expr {
    Alias(Arc<Expr>, PlSmallStr),
    Column(PlSmallStr),
    Columns(Arc<[PlSmallStr]>),
    DtypeColumn(Vec<DataType>),
    IndexColumn(Arc<[i64]>),
    Literal(LiteralValue),
    BinaryExpr {
        left: Arc<Expr>,
        op: Operator,
        right: Arc<Expr>,
    },
    Cast {
        expr: Arc<Expr>,
        dtype: DataType,
        options: CastOptions,
    },
    Sort {
        expr: Arc<Expr>,
        options: SortOptions,
    },
    Gather {
        expr: Arc<Expr>,
        idx: Arc<Expr>,
        returns_scalar: bool,
    },
    SortBy {
        expr: Arc<Expr>,
        by: Vec<Expr>,
        sort_options: SortMultipleOptions,
    },
    Agg(AggExpr),
    /// A ternary operation
    /// if true then "foo" else "bar"
    Ternary {
        predicate: Arc<Expr>,
        truthy: Arc<Expr>,
        falsy: Arc<Expr>,
    },
    Function {
        /// function arguments
        input: Vec<Expr>,
        /// function to apply
        function: FunctionExpr,
        options: FunctionOptions,
    },
    Explode(Arc<Expr>),
    Filter {
        input: Arc<Expr>,
        by: Arc<Expr>,
    },
    /// Polars flavored window functions.
    Window {
        /// Also has the input. i.e. avg("foo")
        function: Arc<Expr>,
        partition_by: Vec<Expr>,
        order_by: Option<(Arc<Expr>, SortOptions)>,
        options: WindowType,
    },
    Wildcard,
    Slice {
        input: Arc<Expr>,
        /// length is not yet known so we accept negative offsets
        offset: Arc<Expr>,
        length: Arc<Expr>,
    },
    /// Can be used in a select statement to exclude a column from selection
    /// TODO: See if we can replace `Vec<Excluded>` with `Arc<Excluded>`
    Exclude(Arc<Expr>, Vec<Excluded>),
    /// Set root name as Alias
    KeepName(Arc<Expr>),
    Len,
    /// Take the nth column in the `DataFrame`
    Nth(i64),
    RenameAlias {
        function: SpecialEq<Arc<dyn RenameAliasFn>>,
        expr: Arc<Expr>,
    },
    #[cfg(feature = "dtype-struct")]
    Field(Arc<[PlSmallStr]>),
    AnonymousFunction {
        /// function arguments
        input: Vec<Expr>,
        /// function to apply
        function: OpaqueColumnUdf,
        /// output dtype of the function
        output_type: GetOutput,
        options: FunctionOptions,
    },
    SubPlan(SpecialEq<Arc<DslPlan>>, Vec<String>),
    /// Expressions in this node should only be expanding
    /// e.g.
    /// `Expr::Columns`
    /// `Expr::Dtypes`
    /// `Expr::Wildcard`
    /// `Expr::Exclude`
    Selector(super::selector::Selector),
}

pub type OpaqueColumnUdf = LazySerde<SpecialEq<Arc<dyn ColumnsUdf>>>;
pub(crate) fn new_column_udf<F: ColumnsUdf + 'static>(func: F) -> OpaqueColumnUdf {
    LazySerde::Deserialized(SpecialEq::new(Arc::new(func)))
}

#[derive(Clone)]
pub enum LazySerde<T: Clone> {
    Deserialized(T),
    Bytes(Bytes),
}

impl<T: PartialEq + Clone> PartialEq for LazySerde<T> {
    fn eq(&self, other: &Self) -> bool {
        use LazySerde as L;
        match (self, other) {
            (L::Deserialized(a), L::Deserialized(b)) => a == b,
            (L::Bytes(a), L::Bytes(b)) => a.as_ptr() == b.as_ptr() && a.len() == b.len(),
            _ => false,
        }
    }
}

impl<T: Clone> Debug for LazySerde<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes(_) => write!(f, "lazy-serde<Bytes>"),
            Self::Deserialized(_) => write!(f, "lazy-serde<T>"),
        }
    }
}

impl OpaqueColumnUdf {
    pub fn materialize(self) -> PolarsResult<SpecialEq<Arc<dyn ColumnsUdf>>> {
        match self {
            Self::Deserialized(t) => Ok(t),
            Self::Bytes(b) => {
                feature_gated!("serde";"python", {
                    python_udf::PythonUdfExpression::try_deserialize(b.as_ref()).map(SpecialEq::new)
                })
            },
        }
    }
}

#[allow(clippy::derived_hash_with_manual_eq)]
impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let d = std::mem::discriminant(self);
        d.hash(state);
        match self {
            Expr::Column(name) => name.hash(state),
            Expr::Columns(names) => names.hash(state),
            Expr::DtypeColumn(dtypes) => dtypes.hash(state),
            Expr::IndexColumn(indices) => indices.hash(state),
            Expr::Literal(lv) => std::mem::discriminant(lv).hash(state),
            Expr::Selector(s) => s.hash(state),
            Expr::Nth(v) => v.hash(state),
            Expr::Filter { input, by } => {
                input.hash(state);
                by.hash(state);
            },
            Expr::BinaryExpr { left, op, right } => {
                left.hash(state);
                right.hash(state);
                std::mem::discriminant(op).hash(state)
            },
            Expr::Cast {
                expr,
                dtype,
                options: strict,
            } => {
                expr.hash(state);
                dtype.hash(state);
                strict.hash(state)
            },
            Expr::Sort { expr, options } => {
                expr.hash(state);
                options.hash(state);
            },
            Expr::Alias(input, name) => {
                input.hash(state);
                name.hash(state)
            },
            Expr::KeepName(input) => input.hash(state),
            Expr::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                predicate.hash(state);
                truthy.hash(state);
                falsy.hash(state);
            },
            Expr::Function {
                input,
                function,
                options,
            } => {
                input.hash(state);
                std::mem::discriminant(function).hash(state);
                options.hash(state);
            },
            Expr::Gather {
                expr,
                idx,
                returns_scalar,
            } => {
                expr.hash(state);
                idx.hash(state);
                returns_scalar.hash(state);
            },
            // already hashed by discriminant
            Expr::Wildcard | Expr::Len => {},
            Expr::SortBy {
                expr,
                by,
                sort_options,
            } => {
                expr.hash(state);
                by.hash(state);
                sort_options.hash(state);
            },
            Expr::Agg(input) => input.hash(state),
            Expr::Explode(input) => input.hash(state),
            Expr::Window {
                function,
                partition_by,
                order_by,
                options,
            } => {
                function.hash(state);
                partition_by.hash(state);
                order_by.hash(state);
                options.hash(state);
            },
            Expr::Slice {
                input,
                offset,
                length,
            } => {
                input.hash(state);
                offset.hash(state);
                length.hash(state);
            },
            Expr::Exclude(input, excl) => {
                input.hash(state);
                excl.hash(state);
            },
            Expr::RenameAlias { function: _, expr } => expr.hash(state),
            Expr::AnonymousFunction {
                input,
                function: _,
                output_type: _,
                options,
            } => {
                input.hash(state);
                options.hash(state);
            },
            Expr::SubPlan(_, names) => names.hash(state),
            #[cfg(feature = "dtype-struct")]
            Expr::Field(names) => names.hash(state),
        }
    }
}

impl Eq for Expr {}

impl Default for Expr {
    fn default() -> Self {
        Expr::Literal(LiteralValue::Null)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]

pub enum Excluded {
    Name(PlSmallStr),
    Dtype(DataType),
}

impl Expr {
    /// Get Field result of the expression. The schema is the input data.
    pub fn to_field(&self, schema: &Schema, ctxt: Context) -> PolarsResult<Field> {
        // this is not called much and the expression depth is typically shallow
        let mut arena = Arena::with_capacity(5);
        self.to_field_amortized(schema, ctxt, &mut arena)
    }
    pub(crate) fn to_field_amortized(
        &self,
        schema: &Schema,
        ctxt: Context,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Field> {
        let root = to_aexpr(self.clone(), expr_arena)?;
        expr_arena.get(root).to_field(schema, ctxt, expr_arena)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Operator {
    Eq,
    EqValidity,
    NotEq,
    NotEqValidity,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Multiply,
    Divide,
    TrueDivide,
    FloorDivide,
    Modulus,
    And,
    Or,
    Xor,
    LogicalAnd,
    LogicalOr,
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use Operator::*;
        let tkn = match self {
            Eq => "==",
            EqValidity => "==v",
            NotEq => "!=",
            NotEqValidity => "!=v",
            Lt => "<",
            LtEq => "<=",
            Gt => ">",
            GtEq => ">=",
            Plus => "+",
            Minus => "-",
            Multiply => "*",
            Divide => "//",
            TrueDivide => "/",
            FloorDivide => "floor_div",
            Modulus => "%",
            And | LogicalAnd => "&",
            Or | LogicalOr => "|",
            Xor => "^",
        };
        write!(f, "{tkn}")
    }
}

impl Operator {
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Eq
                | Self::NotEq
                | Self::Lt
                | Self::LtEq
                | Self::Gt
                | Self::GtEq
                | Self::And
                | Self::Or
                | Self::Xor
                | Self::EqValidity
                | Self::NotEqValidity
        )
    }

    pub fn swap_operands(self) -> Self {
        match self {
            Operator::Eq => Operator::Eq,
            Operator::Gt => Operator::Lt,
            Operator::GtEq => Operator::LtEq,
            Operator::LtEq => Operator::GtEq,
            Operator::Or => Operator::Or,
            Operator::LogicalAnd => Operator::LogicalAnd,
            Operator::LogicalOr => Operator::LogicalOr,
            Operator::Xor => Operator::Xor,
            Operator::NotEq => Operator::NotEq,
            Operator::EqValidity => Operator::EqValidity,
            Operator::NotEqValidity => Operator::NotEqValidity,
            Operator::Divide => Operator::Multiply,
            Operator::Multiply => Operator::Divide,
            Operator::And => Operator::And,
            Operator::Plus => Operator::Minus,
            Operator::Minus => Operator::Plus,
            Operator::Lt => Operator::Gt,
            _ => unimplemented!(),
        }
    }

    pub fn is_arithmetic(&self) -> bool {
        !(self.is_comparison())
    }
}
