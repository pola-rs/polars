use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::{DataType, SortMultipleOptions, SortOptions};
use polars_core::scalar::Scalar;
use polars_utils::IdxSize;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::{AExpr, IRAggExpr, IRBooleanFunction, IRFunctionExpr, RowEncodingVariant};
use crate::dsl::Operator;
use crate::plans::{ExprIR, LiteralValue, OutputName};

#[derive(Clone, Copy)]
pub struct AExprBuilder {
    node: Node,
}

impl AExprBuilder {
    pub fn new_from_node(node: Node) -> Self {
        Self { node }
    }

    pub fn new_from_aexpr(expr: AExpr, arena: &mut Arena<AExpr>) -> Self {
        Self::new_from_node(arena.add(expr))
    }

    pub fn lit(lit: LiteralValue, arena: &mut Arena<AExpr>) -> Self {
        Self::new_from_aexpr(AExpr::Literal(lit), arena)
    }

    pub fn lit_scalar(scalar: Scalar, arena: &mut Arena<AExpr>) -> Self {
        Self::lit(LiteralValue::Scalar(scalar), arena)
    }

    pub fn col(name: impl Into<PlSmallStr>, arena: &mut Arena<AExpr>) -> Self {
        Self::new_from_aexpr(AExpr::Column(name.into()), arena)
    }

    pub fn dataframe_length(arena: &mut Arena<AExpr>) -> Self {
        Self::new_from_aexpr(AExpr::Len, arena)
    }

    pub fn function(
        input: Vec<ExprIR>,
        function: IRFunctionExpr,
        arena: &mut Arena<AExpr>,
    ) -> Self {
        let options = function.function_options();
        Self::new_from_aexpr(
            AExpr::Function {
                input,
                function,
                options,
            },
            arena,
        )
    }

    pub fn row_encode_unary(
        self,
        variant: RowEncodingVariant,
        dtype: DataType,
        arena: &mut Arena<AExpr>,
    ) -> Self {
        Self::function(
            vec![ExprIR::from_node(self.node(), arena)],
            IRFunctionExpr::RowEncode(vec![dtype], variant),
            arena,
        )
    }

    pub fn cast(self, dtype: DataType, arena: &mut Arena<AExpr>) -> Self {
        Self {
            node: arena.add(AExpr::Cast {
                expr: self.node,
                dtype,
                options: CastOptions::Strict,
            }),
        }
    }

    pub fn binary_op(
        self,
        other: impl IntoAExprBuilder,
        op: Operator,
        arena: &mut Arena<AExpr>,
    ) -> Self {
        Self {
            node: arena.add(AExpr::BinaryExpr {
                left: self.node,
                op,
                right: other.into_aexpr_builder().node,
            }),
        }
    }

    pub fn agg(agg: IRAggExpr, arena: &mut Arena<AExpr>) -> Self {
        Self::new_from_aexpr(AExpr::Agg(agg), arena)
    }

    pub fn first(self, arena: &mut Arena<AExpr>) -> Self {
        Self::agg(IRAggExpr::First(self.node()), arena)
    }

    pub fn last(self, arena: &mut Arena<AExpr>) -> Self {
        Self::agg(IRAggExpr::Last(self.node()), arena)
    }

    pub fn min(self, arena: &mut Arena<AExpr>) -> Self {
        Self::agg(
            IRAggExpr::Min {
                input: self.node(),
                propagate_nans: false,
            },
            arena,
        )
    }

    pub fn max(self, arena: &mut Arena<AExpr>) -> Self {
        Self::agg(
            IRAggExpr::Max {
                input: self.node(),
                propagate_nans: false,
            },
            arena,
        )
    }

    pub fn nan_min(self, arena: &mut Arena<AExpr>) -> Self {
        Self::agg(
            IRAggExpr::Min {
                input: self.node(),
                propagate_nans: true,
            },
            arena,
        )
    }

    pub fn nan_max(self, arena: &mut Arena<AExpr>) -> Self {
        Self::agg(
            IRAggExpr::Max {
                input: self.node(),
                propagate_nans: true,
            },
            arena,
        )
    }

    pub fn sum(self, arena: &mut Arena<AExpr>) -> Self {
        Self::agg(IRAggExpr::Sum(self.node()), arena)
    }

    pub fn len(self, arena: &mut Arena<AExpr>) -> Self {
        Self::agg(
            IRAggExpr::Count {
                input: self.node(),
                include_nulls: true,
            },
            arena,
        )
    }

    pub fn count(self, arena: &mut Arena<AExpr>) -> Self {
        Self::agg(
            IRAggExpr::Count {
                input: self.node(),
                include_nulls: false,
            },
            arena,
        )
    }

    pub fn count_opt_nulls(self, include_nulls: bool, arena: &mut Arena<AExpr>) -> Self {
        Self::agg(
            IRAggExpr::Count {
                input: self.node(),
                include_nulls,
            },
            arena,
        )
    }

    pub fn explode_skip_empty(self, arena: &mut Arena<AExpr>) -> Self {
        Self::new_from_aexpr(
            AExpr::Explode {
                expr: self.node(),
                skip_empty: true,
            },
            arena,
        )
    }

    pub fn explode_null_empty(self, arena: &mut Arena<AExpr>) -> Self {
        Self::new_from_aexpr(
            AExpr::Explode {
                expr: self.node(),
                skip_empty: false,
            },
            arena,
        )
    }

    pub fn sort(self, options: SortOptions, arena: &mut Arena<AExpr>) -> Self {
        Self::new_from_aexpr(
            AExpr::Sort {
                expr: self.node(),
                options,
            },
            arena,
        )
    }

    pub fn sort_by(
        self,
        by: Vec<Node>,
        options: SortMultipleOptions,
        arena: &mut Arena<AExpr>,
    ) -> Self {
        Self::new_from_aexpr(
            AExpr::SortBy {
                expr: self.node(),
                by,
                sort_options: options,
            },
            arena,
        )
    }

    pub fn filter(self, by: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        Self::new_from_aexpr(
            AExpr::Filter {
                input: self.node(),
                by: by.into_aexpr_builder().node(),
            },
            arena,
        )
    }

    pub fn when_then_otherwise(
        when: impl IntoAExprBuilder,
        then: impl IntoAExprBuilder,
        otherwise: impl IntoAExprBuilder,
        arena: &mut Arena<AExpr>,
    ) -> Self {
        when.into_aexpr_builder().ternary(then, otherwise, arena)
    }

    pub fn ternary(
        self,
        truthy: impl IntoAExprBuilder,
        falsy: impl IntoAExprBuilder,
        arena: &mut Arena<AExpr>,
    ) -> Self {
        Self::new_from_aexpr(
            AExpr::Ternary {
                predicate: self.into_aexpr_builder().node(),
                truthy: truthy.into_aexpr_builder().node(),
                falsy: falsy.into_aexpr_builder().node(),
            },
            arena,
        )
    }

    pub fn shift(self, periods: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        Self::function(
            vec![
                self.expr_ir_unnamed(),
                periods.into_aexpr_builder().expr_ir_unnamed(),
            ],
            IRFunctionExpr::Shift,
            arena,
        )
    }

    pub fn slice(
        self,
        offset: impl IntoAExprBuilder,
        length: impl IntoAExprBuilder,
        arena: &mut Arena<AExpr>,
    ) -> Self {
        Self::new_from_aexpr(
            AExpr::Slice {
                input: self.into_aexpr_builder().node(),
                offset: offset.into_aexpr_builder().node(),
                length: length.into_aexpr_builder().node(),
            },
            arena,
        )
    }

    #[cfg(feature = "is_in")]
    pub fn is_in(
        self,
        other: impl IntoAExprBuilder,
        nulls_equal: bool,
        arena: &mut Arena<AExpr>,
    ) -> Self {
        Self::function(
            vec![
                self.expr_ir_unnamed(),
                other.into_aexpr_builder().expr_ir_unnamed(),
            ],
            IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { nulls_equal }),
            arena,
        )
    }

    pub fn to_physical(self, arena: &mut Arena<AExpr>) -> Self {
        Self::function(
            vec![self.expr_ir_unnamed()],
            IRFunctionExpr::ToPhysical,
            arena,
        )
    }

    #[cfg(feature = "abs")]
    pub fn abs(self, arena: &mut Arena<AExpr>) -> Self {
        Self::function(vec![self.expr_ir_unnamed()], IRFunctionExpr::Abs, arena)
    }

    pub fn negate(self, arena: &mut Arena<AExpr>) -> Self {
        Self::function(vec![self.expr_ir_unnamed()], IRFunctionExpr::Negate, arena)
    }

    pub fn not(self, arena: &mut Arena<AExpr>) -> Self {
        Self::function(
            vec![self.expr_ir_unnamed()],
            IRFunctionExpr::Boolean(IRBooleanFunction::Not),
            arena,
        )
    }

    pub fn null_count(self, arena: &mut Arena<AExpr>) -> Self {
        Self::function(
            vec![self.expr_ir_unnamed()],
            IRFunctionExpr::NullCount,
            arena,
        )
    }

    pub fn is_null(self, arena: &mut Arena<AExpr>) -> Self {
        Self::function(
            vec![self.expr_ir_unnamed()],
            IRFunctionExpr::Boolean(IRBooleanFunction::IsNull),
            arena,
        )
    }

    pub fn is_not_null(self, arena: &mut Arena<AExpr>) -> Self {
        Self::function(
            vec![self.expr_ir_unnamed()],
            IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull),
            arena,
        )
    }

    pub fn is_nan(self, arena: &mut Arena<AExpr>) -> Self {
        Self::function(
            vec![self.expr_ir_unnamed()],
            IRFunctionExpr::Boolean(IRBooleanFunction::IsNan),
            arena,
        )
    }

    pub fn is_not_nan(self, arena: &mut Arena<AExpr>) -> Self {
        Self::function(
            vec![self.expr_ir_unnamed()],
            IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNan),
            arena,
        )
    }

    pub fn has_no_nulls(self, arena: &mut Arena<AExpr>) -> Self {
        let nc = self.null_count(arena);
        let idx_zero = Self::lit_scalar(Scalar::from(0 as IdxSize), arena);
        nc.eq(idx_zero, arena)
    }

    pub fn has_nulls(self, arena: &mut Arena<AExpr>) -> Self {
        let nc = self.null_count(arena);
        let idx_zero = Self::lit_scalar(Scalar::from(0 as IdxSize), arena);
        nc.gt(idx_zero, arena)
    }

    pub fn drop_nulls(self, arena: &mut Arena<AExpr>) -> Self {
        Self::function(
            vec![self.expr_ir_retain_name(arena)],
            IRFunctionExpr::DropNulls,
            arena,
        )
    }

    pub fn drop_nans(self, arena: &mut Arena<AExpr>) -> Self {
        Self::function(
            vec![self.expr_ir_retain_name(arena)],
            IRFunctionExpr::DropNans,
            arena,
        )
    }

    pub fn eq(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::Eq, arena)
    }

    pub fn eq_validity(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::EqValidity, arena)
    }

    pub fn not_eq(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::NotEq, arena)
    }

    pub fn not_eq_validity(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::NotEqValidity, arena)
    }

    pub fn lt(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::Lt, arena)
    }

    pub fn lt_eq(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::LtEq, arena)
    }

    pub fn gt(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::Gt, arena)
    }

    pub fn gt_eq(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::GtEq, arena)
    }

    pub fn plus(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::Plus, arena)
    }

    pub fn minus(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::Minus, arena)
    }

    pub fn multiply(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::Multiply, arena)
    }

    pub fn divide(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::Divide, arena)
    }

    pub fn true_divide(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::TrueDivide, arena)
    }

    pub fn floor_divide(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::FloorDivide, arena)
    }

    pub fn modulus(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::Modulus, arena)
    }

    pub fn and(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::And, arena)
    }

    pub fn or(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::Or, arena)
    }

    pub fn xor(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::Xor, arena)
    }

    pub fn logical_and(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::LogicalAnd, arena)
    }

    pub fn logical_or(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::LogicalOr, arena)
    }

    pub fn expr_ir(self, name: impl Into<PlSmallStr>) -> ExprIR {
        ExprIR::new(self.node(), OutputName::Alias(name.into()))
    }

    pub fn expr_ir_retain_name(self, arena: &Arena<AExpr>) -> ExprIR {
        ExprIR::from_node(self.node(), arena)
    }

    pub fn expr_ir_unnamed(self) -> ExprIR {
        self.expr_ir(PlSmallStr::EMPTY)
    }

    pub fn node(self) -> Node {
        self.node
    }
}

pub trait IntoAExprBuilder {
    fn into_aexpr_builder(self) -> AExprBuilder;
}

impl IntoAExprBuilder for Node {
    fn into_aexpr_builder(self) -> AExprBuilder {
        AExprBuilder { node: self }
    }
}

impl IntoAExprBuilder for AExprBuilder {
    fn into_aexpr_builder(self) -> AExprBuilder {
        self
    }
}
