pub mod anonymous;
mod datatype_fn;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

pub use anonymous::*;
use bytes::Bytes;
pub use datatype_fn::*;
use polars_compute::rolling::QuantileMethod;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::error::feature_gated;
use polars_core::prelude::*;
use polars_utils::format_pl_smallstr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
    FirstNonNull(Arc<Expr>),
    Last(Arc<Expr>),
    LastNonNull(Arc<Expr>),
    Item {
        input: Arc<Expr>,
        /// Give a missing value if there are no values.
        allow_empty: bool,
    },
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
            FirstNonNull(e) => e,
            Last(e) => e,
            LastNonNull(e) => e,
            Item { input, .. } => input,
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
    /// Values in a `eval` context.
    ///
    /// Equivalent of `pl.element()`.
    Element,
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
        null_on_oob: bool,
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
        options: ExplodeOptions,
    },
    Filter {
        input: Arc<Expr>,
        by: Arc<Expr>,
    },
    /// Polars flavored window functions.
    Over {
        /// Also has the input. i.e. avg("foo")
        function: Arc<Expr>,
        partition_by: Vec<Expr>,
        order_by: Option<(Arc<Expr>, SortOptions)>,
        mapping: WindowMapping,
    },
    #[cfg(feature = "dynamic_group_by")]
    Rolling {
        function: Arc<Expr>,
        index_column: Arc<Expr>,
        period: Duration,
        offset: Duration,
        closed_window: ClosedWindow,
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
    AnonymousAgg {
        /// function arguments
        input: Vec<Expr>,
        /// function to apply
        function: OpaqueStreamingAgg,

        /// used for formatting
        #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
        fmt_str: Box<PlSmallStr>,
    },
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
    /// Evaluates the `evaluation` expressions on the output of the `expr`.
    ///
    /// Consequently, `expr` is an input and `evaluation` uses an extended schema that includes this input.
    #[cfg(feature = "dtype-struct")]
    StructEval {
        expr: Arc<Expr>,
        evaluation: Vec<Expr>,
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
                null_on_oob,
            } => {
                expr.hash(state);
                idx.hash(state);
                returns_scalar.hash(state);
                null_on_oob.hash(state);
            },
            // already hashed by discriminant
            Expr::Element | Expr::Len => {},
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
            Expr::Explode { input, options } => {
                options.hash(state);
                input.hash(state)
            },
            #[cfg(feature = "dynamic_group_by")]
            Expr::Rolling {
                function,
                index_column,
                period,
                offset,
                closed_window,
            } => {
                function.hash(state);
                index_column.hash(state);
                period.hash(state);
                offset.hash(state);
                closed_window.hash(state);
            },
            Expr::Over {
                function,
                partition_by,
                order_by,
                mapping,
            } => {
                function.hash(state);
                partition_by.hash(state);
                order_by.hash(state);
                mapping.hash(state);
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
            Expr::AnonymousAgg {
                input,
                function: _,
                fmt_str,
            } => {
                input.hash(state);
                fmt_str.hash(state);
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
            #[cfg(feature = "dtype-struct")]
            Expr::StructEval {
                expr: input,
                evaluation,
            } => {
                input.hash(state);
                evaluation.hash(state);
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
        let mut ctx = ExprToIRContext::new_with_fields(expr_arena, schema);
        ctx.allow_unknown = true;
        let expr = to_expr_ir(self.clone(), &mut ctx)?;
        let (node, output_name) = expr.into_inner();
        let dtype = expr_arena
            .get(node)
            .to_dtype(&ToFieldContext::new(expr_arena, schema))?;
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
    /// `list.agg`
    ListAgg,

    /// `array.eval`
    Array {
        /// If set to true, evaluation can output variable amount of items and output datatype will
        /// be `List`.
        as_list: bool,
    },
    /// `arr.agg`
    ArrayAgg,

    /// `cumulative_eval`
    Cumulative { min_samples: usize },
}

impl EvalVariant {
    pub fn to_name(&self) -> &'static str {
        match self {
            Self::List => "list.eval",
            Self::ListAgg => "list.agg",
            Self::Array { .. } => "array.eval",
            Self::ArrayAgg => "array.agg",
            Self::Cumulative { min_samples: _ } => "cumulative_eval",
        }
    }

    /// Get the `DataType` of the `pl.element()` value.
    pub fn element_dtype<'a>(&self, dtype: &'a DataType) -> PolarsResult<&'a DataType> {
        match (self, dtype) {
            (Self::List | Self::ListAgg, DataType::List(inner)) => Ok(inner.as_ref()),
            #[cfg(feature = "dtype-array")]
            (Self::Array { .. } | Self::ArrayAgg, DataType::Array(inner, _)) => Ok(inner.as_ref()),
            (Self::Cumulative { min_samples: _ }, dt) => Ok(dt),
            _ => polars_bail!(op = self.to_name(), dtype),
        }
    }

    /// Get the output datatype from the output element datatype
    pub fn output_dtype(
        &self,
        dtype: &'_ DataType,
        output_element_dtype: DataType,
        eval_is_scalar: bool,
    ) -> PolarsResult<DataType> {
        match (self, dtype) {
            (Self::List, DataType::List(_)) => Ok(DataType::List(Box::new(output_element_dtype))),
            (Self::ListAgg, DataType::List(_)) => {
                if eval_is_scalar {
                    Ok(output_element_dtype)
                } else {
                    Ok(DataType::List(Box::new(output_element_dtype)))
                }
            },
            #[cfg(feature = "dtype-array")]
            (Self::Array { as_list: false }, DataType::Array(_, width)) => {
                Ok(DataType::Array(Box::new(output_element_dtype), *width))
            },
            #[cfg(feature = "dtype-array")]
            (Self::Array { as_list: true }, DataType::Array(_, _)) => {
                Ok(DataType::List(Box::new(output_element_dtype)))
            },
            #[cfg(feature = "dtype-array")]
            (Self::ArrayAgg, DataType::Array(_, _)) => {
                if eval_is_scalar {
                    Ok(output_element_dtype)
                } else {
                    Ok(DataType::List(Box::new(output_element_dtype)))
                }
            },
            (Self::Cumulative { min_samples: _ }, _) => Ok(output_element_dtype),
            _ => polars_bail!(op = self.to_name(), dtype),
        }
    }

    pub fn is_elementwise(&self) -> bool {
        match self {
            EvalVariant::List | EvalVariant::ListAgg => true,
            EvalVariant::Array { .. } | EvalVariant::ArrayAgg => true,
            EvalVariant::Cumulative { min_samples: _ } => false,
        }
    }

    pub fn is_row_separable(&self) -> bool {
        match self {
            EvalVariant::List | EvalVariant::ListAgg => true,
            EvalVariant::Array { .. } | EvalVariant::ArrayAgg => true,
            EvalVariant::Cumulative { min_samples: _ } => false,
        }
    }

    pub fn is_length_preserving(&self) -> bool {
        match self {
            EvalVariant::List
            | EvalVariant::ListAgg
            | EvalVariant::Array { .. }
            | EvalVariant::ArrayAgg
            | EvalVariant::Cumulative { .. } => true,
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
    /// Rust division semantics, this is what Rust interface `/` fispatches to
    RustDivide,
    /// Python division semantics, converting to floats. This is what python `/` operator dispatches to
    TrueDivide,
    /// Floor division semantics, this is what python `//` dispatches to
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
            RustDivide => "rust_div",
            TrueDivide => "/",
            FloorDivide => "//",
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
            // Operator::Divide requires modifying the right operand: left / right == 1/right * left
            Operator::RustDivide => unimplemented!(),
            Operator::Multiply => Operator::Multiply,
            Operator::And => Operator::And,
            Operator::Plus => Operator::Plus,
            // Operator::Minus requires modifying the right operand: left - right == -right + left
            Operator::Minus => unimplemented!(),
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
    Map(PlanCallback<PlSmallStr, PlSmallStr>),
    Replace {
        pattern: PlSmallStr,
        value: PlSmallStr,
        literal: bool,
    },
}

impl RenameAliasFn {
    pub fn call(&self, name: &PlSmallStr) -> PolarsResult<PlSmallStr> {
        let out = match self {
            Self::Prefix(prefix) => format_pl_smallstr!("{prefix}{name}"),
            Self::Suffix(suffix) => format_pl_smallstr!("{name}{suffix}"),
            Self::ToLowercase => PlSmallStr::from_string(name.to_lowercase()),
            Self::ToUppercase => PlSmallStr::from_string(name.to_uppercase()),
            Self::Map(f) => f.call(name.clone())?,
            Self::Replace {
                pattern,
                value,
                literal,
            } => {
                if *literal {
                    name.replace(pattern.as_str(), value.as_str()).into()
                } else {
                    feature_gated!("regex", {
                        let rx = polars_utils::regex_cache::compile_regex(pattern)?;
                        rx.replace_all(name, value.as_str()).into()
                    })
                }
            },
        };
        Ok(out)
    }
}

pub type RenameAliasRustFn =
    dyn Fn(&PlSmallStr) -> PolarsResult<PlSmallStr> + 'static + Send + Sync;
