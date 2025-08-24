mod datatype_fn;
mod expr_dyn_fn;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

use bytes::Bytes;
pub use datatype_fn::*;
pub use expr_dyn_fn::*;
use polars_compute::rolling::QuantileMethod;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::error::feature_gated;
use polars_core::prelude::*;
use polars_utils::format_pl_smallstr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde")]
pub mod named_serde;
#[cfg(feature = "serde")]
mod serde_expr;

use super::datatype_expr::DataTypeExpr;
use crate::prelude::*;

#[derive(PartialEq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
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
    Count {
        input: Arc<Expr>,
        include_nulls: bool,
    },
    Quantile {
        expr: Arc<Expr>,
        quantile: Arc<Expr>,
        method: QuantileMethod,
    },
    Sum(Arc<Expr>),
    AggGroups(Arc<Expr>),
    Std(Arc<Expr>, u8),
    Var(Arc<Expr>, u8),
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
            Count { input, .. } => input,
            Quantile { expr, .. } => expr,
            Sum(e) => e,
            AggGroups(e) => e,
            Std(e, _) => e,
            Var(e, _) => e,
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
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum Expr {
    Alias(Arc<Expr>, PlSmallStr),
    Column(PlSmallStr),
    Selector(Selector),
    Literal(LiteralValue),
    DataTypeFunction(DataTypeFunction),
    BinaryExpr {
        left: Arc<Expr>,
        op: Operator,
        right: Arc<Expr>,
    },
    Cast {
        expr: Arc<Expr>,
        dtype: DataTypeExpr,
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
    },
    Explode {
        input: Arc<Expr>,
        skip_empty: bool,
    },
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
    Slice {
        input: Arc<Expr>,
        /// length is not yet known so we accept negative offsets
        offset: Arc<Expr>,
        length: Arc<Expr>,
    },
    /// Set root name as Alias
    KeepName(Arc<Expr>),
    Len,
    #[cfg(feature = "dtype-struct")]
    Field(Arc<[PlSmallStr]>),
    AnonymousFunction {
        /// function arguments
        input: Vec<Expr>,
        /// function to apply
        function: OpaqueColumnUdf,

        options: FunctionOptions,
        /// used for formatting
        #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
        fmt_str: Box<PlSmallStr>,
    },
    /// Evaluates the `evaluation` expression on the output of the `expr`.
    ///
    /// Consequently, `expr` is an input and `evaluation` is not and needs a different schema.
    Eval {
        expr: Arc<Expr>,
        evaluation: Arc<Expr>,
        variant: EvalVariant,
    },
    SubPlan(SpecialEq<Arc<DslPlan>>, Vec<String>),
    RenameAlias {
        function: RenameAliasFn,
        expr: Arc<Expr>,
    },
}

#[derive(Clone)]
pub enum LazySerde<T: Clone> {
    Deserialized(T),
    Bytes(Bytes),
    /// Named functions allow for serializing arbitrary Rust functions as long as both sides know
    /// ahead of time which function it is. There is a registry of functions that both sides know
    /// and every time we need serialize we serialize the function by name in the registry.
    ///
    /// Used by cloud.
    Named {
        // Name and payload are used by the NamedRegistry
        // To load the function `T` at runtime.
        name: String,
        payload: Option<Bytes>,
        // Sometimes we need the function `T` before sending
        // to a different machine, so optionally set it as well.
        value: Option<T>,
    },
}

impl<T: PartialEq + Clone> PartialEq for LazySerde<T> {
    fn eq(&self, other: &Self) -> bool {
        use LazySerde as L;
        match (self, other) {
            (L::Deserialized(a), L::Deserialized(b)) => a == b,
            (L::Bytes(a), L::Bytes(b)) => {
                std::ptr::eq(a.as_ptr(), b.as_ptr()) && a.len() == b.len()
            },
            (
                L::Named {
                    name: l,
                    payload: pl,
                    value: _,
                },
                L::Named {
                    name: r,
                    payload: pr,
                    value: _,
                },
            ) => {
                #[cfg(debug_assertions)]
                {
                    if l == r {
                        assert_eq!(pl, pr, "name should point to unique payload")
                    }
                }
                _ = pl;
                _ = pr;
                l == r
            },
            _ => false,
        }
    }
}

impl<T: Clone> Debug for LazySerde<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes(_) => write!(f, "lazy-serde<Bytes>"),
            Self::Deserialized(_) => write!(f, "lazy-serde<T>"),
            Self::Named {
                name,
                payload: _,
                value: _,
            } => write!(f, "lazy-serde<Named>: {name}"),
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
            // Expr::Columns(names) => names.hash(state),
            // Expr::DtypeColumn(dtypes) => dtypes.hash(state),
            // Expr::IndexColumn(indices) => indices.hash(state),
            Expr::Literal(lv) => std::mem::discriminant(lv).hash(state),
            Expr::Selector(s) => s.hash(state),
            // Expr::Nth(v) => v.hash(state),
            Expr::DataTypeFunction(v) => v.hash(state),
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
            Expr::Function { input, function } => {
                input.hash(state);
                std::mem::discriminant(function).hash(state);
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
            Expr::Len => {},
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
            Expr::Explode { input, skip_empty } => {
                skip_empty.hash(state);
                input.hash(state)
            },
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
            // Expr::Exclude(input, excl) => {
            //     input.hash(state);
            //     excl.hash(state);
            // },
            Expr::RenameAlias { function, expr } => {
                function.hash(state);
                expr.hash(state);
            },
            Expr::AnonymousFunction {
                input,
                function: _,
                options,
                fmt_str,
            } => {
                input.hash(state);
                options.hash(state);
                fmt_str.hash(state);
            },
            Expr::Eval {
                expr: input,
                evaluation,
                variant,
            } => {
                input.hash(state);
                evaluation.hash(state);
                variant.hash(state);
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
        Expr::Literal(LiteralValue::Scalar(Scalar::default()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum Excluded {
    Name(PlSmallStr),
    Dtype(DataType),
}

impl Expr {
    /// Get Field result of the expression. The schema is the input data.
    pub fn to_field(&self, schema: &Schema) -> PolarsResult<Field> {
        // this is not called much and the expression depth is typically shallow
        let mut arena = Arena::with_capacity(5);
        self.to_field_amortized(schema, &mut arena)
    }
    pub(crate) fn to_field_amortized(
        &self,
        schema: &Schema,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Field> {
        let mut ctx = ExprToIRContext::new(expr_arena, schema);
        ctx.allow_unknown = true;
        let expr = to_expr_ir(self.clone(), &mut ctx)?;
        let (node, output_name) = expr.into_inner();
        let dtype = expr_arena.get(node).to_dtype(schema, expr_arena)?;
        Ok(Field::new(output_name.into_inner().unwrap(), dtype))
    }

    pub fn into_selector(self) -> Option<Selector> {
        match self {
            Expr::Column(name) => Some(Selector::ByName {
                names: [name].into(),
                strict: true,
            }),
            Expr::Selector(selector) => Some(selector),
            _ => None,
        }
    }

    pub fn try_into_selector(self) -> PolarsResult<Selector> {
        match self {
            Expr::Column(name) => Ok(Selector::ByName {
                names: [name].into(),
                strict: true,
            }),
            Expr::Selector(selector) => Ok(selector),
            expr => Err(polars_err!(InvalidOperation: "cannot turn `{expr}` into selector")),
        }
    }

    /// Extract a constant usize from an expression.
    pub fn extract_usize(&self) -> PolarsResult<usize> {
        match self {
            Expr::Literal(n) => n.extract_usize(),
            Expr::Cast { expr, dtype, .. } => {
                // lit(x, dtype=...) are Cast expressions. We verify the inner expression is literal.
                if dtype.as_literal().is_some_and(|dt| dt.is_integer()) {
                    expr.extract_usize()
                } else {
                    polars_bail!(InvalidOperation: "expression must be constant literal to extract integer")
                }
            },
            _ => {
                polars_bail!(InvalidOperation: "expression must be constant literal to extract integer")
            },
        }
    }

    pub fn extract_i64(&self) -> PolarsResult<i64> {
        match self {
            Expr::Literal(n) => n.extract_i64(),
            Expr::BinaryExpr { left, op, right } => match op {
                Operator::Minus => {
                    let left = left.extract_i64()?;
                    let right = right.extract_i64()?;
                    Ok(left - right)
                },
                _ => unreachable!(),
            },
            Expr::Cast { expr, dtype, .. } => {
                if dtype.as_literal().is_some_and(|dt| dt.is_integer()) {
                    expr.extract_i64()
                } else {
                    polars_bail!(InvalidOperation: "expression must be constant literal to extract integer")
                }
            },
            _ => {
                polars_bail!(InvalidOperation: "expression must be constant literal to extract integer")
            },
        }
    }

    #[inline]
    pub fn map_unary(self, function: impl Into<FunctionExpr>) -> Self {
        Expr::n_ary(function, vec![self])
    }
    #[inline]
    pub fn map_binary(self, function: impl Into<FunctionExpr>, rhs: Self) -> Self {
        Expr::n_ary(function, vec![self, rhs])
    }

    #[inline]
    pub fn map_ternary(self, function: impl Into<FunctionExpr>, arg1: Expr, arg2: Expr) -> Expr {
        Expr::n_ary(function, vec![self, arg1, arg2])
    }

    #[inline]
    pub fn try_map_n_ary(
        self,
        function: impl Into<FunctionExpr>,
        exprs: impl IntoIterator<Item = PolarsResult<Expr>>,
    ) -> PolarsResult<Expr> {
        let exprs = exprs.into_iter();
        let mut input = Vec::with_capacity(exprs.size_hint().0 + 1);
        input.push(self);
        for e in exprs {
            input.push(e?);
        }
        Ok(Expr::n_ary(function, input))
    }

    #[inline]
    pub fn map_n_ary(
        self,
        function: impl Into<FunctionExpr>,
        exprs: impl IntoIterator<Item = Expr>,
    ) -> Expr {
        let exprs = exprs.into_iter();
        let mut input = Vec::with_capacity(exprs.size_hint().0 + 1);
        input.push(self);
        input.extend(exprs);
        Expr::n_ary(function, input)
    }

    #[inline]
    pub fn n_ary(function: impl Into<FunctionExpr>, input: Vec<Expr>) -> Expr {
        let function = function.into();
        Expr::Function { input, function }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum EvalVariant {
    /// `list.eval`
    List,

    /// `cumulative_eval`
    Cumulative { min_samples: usize },
}

impl EvalVariant {
    pub fn to_name(&self) -> &'static str {
        match self {
            Self::List => "list.eval",
            Self::Cumulative { min_samples: _ } => "cumulative_eval",
        }
    }

    /// Get the `DataType` of the `pl.element()` value.
    pub fn element_dtype<'a>(&self, dtype: &'a DataType) -> PolarsResult<&'a DataType> {
        match (self, dtype) {
            (Self::List, DataType::List(inner)) => Ok(inner.as_ref()),
            (Self::Cumulative { min_samples: _ }, dt) => Ok(dt),
            _ => polars_bail!(op = self.to_name(), dtype),
        }
    }

    pub fn is_elementwise(&self) -> bool {
        match self {
            EvalVariant::List => true,
            EvalVariant::Cumulative { min_samples: _ } => false,
        }
    }

    pub fn is_row_separable(&self) -> bool {
        match self {
            EvalVariant::List => true,
            EvalVariant::Cumulative { min_samples: _ } => false,
        }
    }

    pub fn is_length_preserving(&self) -> bool {
        match self {
            EvalVariant::List | EvalVariant::Cumulative { .. } => true,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
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
                | Self::EqValidity
                | Self::NotEqValidity
        )
    }

    pub fn is_bitwise(&self) -> bool {
        matches!(self, Self::And | Self::Or | Self::Xor)
    }

    pub fn is_comparison_or_bitwise(&self) -> bool {
        self.is_comparison() || self.is_bitwise()
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
        !(self.is_comparison_or_bitwise())
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum RenameAliasFn {
    Prefix(PlSmallStr),
    Suffix(PlSmallStr),
    ToLowercase,
    ToUppercase,
    #[cfg(feature = "python")]
    Python(SpecialEq<Arc<polars_utils::python_function::PythonObject>>),
    #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
    Rust(SpecialEq<Arc<RenameAliasRustFn>>),
}

impl RenameAliasFn {
    pub fn call(&self, name: &PlSmallStr) -> PolarsResult<PlSmallStr> {
        let out = match self {
            Self::Prefix(prefix) => format_pl_smallstr!("{prefix}{name}"),
            Self::Suffix(suffix) => format_pl_smallstr!("{name}{suffix}"),
            Self::ToLowercase => PlSmallStr::from_string(name.to_lowercase()),
            Self::ToUppercase => PlSmallStr::from_string(name.to_uppercase()),
            #[cfg(feature = "python")]
            Self::Python(lambda) => {
                let name = name.as_str();
                pyo3::marker::Python::with_gil(|py| {
                    let out: PlSmallStr = lambda
                        .call1(py, (name,))?
                        .extract::<std::borrow::Cow<str>>(py)?
                        .as_ref()
                        .into();
                    pyo3::PyResult::<_>::Ok(out)
                }).map_err(|e| polars_err!(ComputeError: "Python function in 'name.map' produced an error: {e}."))?
            },
            Self::Rust(f) => f(name)?,
        };
        Ok(out)
    }
}

pub type RenameAliasRustFn =
    dyn Fn(&PlSmallStr) -> PolarsResult<PlSmallStr> + 'static + Send + Sync;
