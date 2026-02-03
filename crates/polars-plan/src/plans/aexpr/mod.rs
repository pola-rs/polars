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
pub use scalar::{is_length_preserving_ae, is_scalar_ae};
use strum_macros::IntoStaticStr;
pub use traverse::*;
mod properties;
pub use aexpr::function_expr::schema::FieldsMapper;
pub use builder::AExprBuilder;
pub use evaluate::{constant_evaluate, into_column};
pub use properties::*;
pub use schema::ToFieldContext;

use crate::constants::LEN;
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
    Item {
        input: Node,
        /// Return a missing value if there are no values.
        allow_empty: bool,
    },
    First(Node),
    FirstNonNull(Node),
    Last(Node),
    LastNonNull(Node),
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
            FirstNonNull(_) => GroupByMethod::FirstNonNull,
            Last(_) => GroupByMethod::Last,
            LastNonNull(_) => GroupByMethod::LastNonNull,
            Item { allow_empty, .. } => GroupByMethod::Item { allow_empty },
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
            // Multi-input aggregations.
            Quantile { .. } => unreachable!(),
        }
    }
}

/// IR expression node that is allocated in an [`Arena`][polars_utils::arena::Arena].
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AExpr {
    /// Values in a `eval` context.
    ///
    /// Equivalent of `pl.element()`.
    Element,
    Explode {
        expr: Node,
        options: ExplodeOptions,
    },
    Column(PlSmallStr),
    /// Struct field value in a `struct.with_fields` context.
    ///
    /// Equivalent of `pl.field(name)`.
    #[cfg(feature = "dtype-struct")]
    StructField(PlSmallStr),
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
        null_on_oob: bool,
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
    AnonymousAgg {
        input: Vec<ExprIR>,
        fmt_str: Box<PlSmallStr>,
        function: OpaqueStreamingAgg,
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
    #[cfg(feature = "dtype-struct")]
    StructEval {
        expr: Node,
        evaluation: Vec<ExprIR>,
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
    Over {
        function: Node,
        partition_by: Vec<Node>,
        order_by: Option<(Node, SortOptions)>,
        mapping: WindowMapping,
    },
    #[cfg(feature = "dynamic_group_by")]
    Rolling {
        function: Node,
        index_column: Node,
        period: Duration,
        offset: Duration,
        closed_window: ClosedWindow,
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

    #[recursive::recursive]
    pub fn is_scalar(&self, arena: &Arena<AExpr>) -> bool {
        match self {
            AExpr::Element => false,
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
            AExpr::Agg(_) | AExpr::AnonymousAgg { .. } | AExpr::Len => true,
            AExpr::Cast { expr, .. } => is_scalar_ae(*expr, arena),
            AExpr::Eval { expr, variant, .. } => {
                variant.is_length_preserving() && is_scalar_ae(*expr, arena)
            },
            #[cfg(feature = "dtype-struct")]
            AExpr::StructEval { expr, .. } => is_scalar_ae(*expr, arena),
            AExpr::Sort { expr, .. } => is_scalar_ae(*expr, arena),
            AExpr::Gather { returns_scalar, .. } => *returns_scalar,
            AExpr::SortBy { expr, .. } => is_scalar_ae(*expr, arena),

            // Over and Rolling implicitly zip with the context and thus are never scalars
            AExpr::Over { .. } => false,
            #[cfg(feature = "dynamic_group_by")]
            AExpr::Rolling { .. } => false,

            AExpr::Explode { .. }
            | AExpr::Column(_)
            | AExpr::Filter { .. }
            | AExpr::Slice { .. } => false,
            #[cfg(feature = "dtype-struct")]
            AExpr::StructField(_) => false,
        }
    }

    #[recursive::recursive]
    pub fn is_length_preserving(&self, arena: &Arena<AExpr>) -> bool {
        fn broadcasting_input_length_preserving(
            n: impl IntoIterator<Item = Node>,
            arena: &Arena<AExpr>,
        ) -> bool {
            let mut num_items = 0;
            let mut num_length_preserving = 0;
            let mut num_scalar_or_length_preserving = 0;

            for n in n {
                num_items += 1;

                if is_length_preserving_ae(n, arena) {
                    num_length_preserving += 1;
                    num_scalar_or_length_preserving += 1;
                } else if is_scalar_ae(n, arena) {
                    num_scalar_or_length_preserving += 1;
                }
            }

            num_length_preserving > 0 && num_scalar_or_length_preserving == num_items
        }

        match self {
            AExpr::Element => true,
            AExpr::Column(_) => true,
            #[cfg(feature = "dtype-struct")]
            AExpr::StructField(_) => true,

            // Over and Rolling implicitly zip with the context and thus should always be length
            // preserving
            AExpr::Over { mapping, .. } => !matches!(mapping, WindowMapping::Explode),
            #[cfg(feature = "dynamic_group_by")]
            AExpr::Rolling { .. } => true,

            AExpr::AnonymousAgg { .. } | AExpr::Literal(_) | AExpr::Agg(_) | AExpr::Len => false,
            AExpr::Function { options, input, .. }
            | AExpr::AnonymousFunction { options, input, .. } => {
                if options.flags.is_elementwise() {
                    broadcasting_input_length_preserving(input.iter().map(|e| e.node()), arena)
                } else if options.flags.is_length_preserving() {
                    input.iter().all(|e| e.is_length_preserving(arena))
                } else {
                    false
                }
            },
            AExpr::BinaryExpr { left, right, .. } => {
                broadcasting_input_length_preserving([*left, *right], arena)
            },
            AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            } => broadcasting_input_length_preserving([*predicate, *truthy, *falsy], arena),
            AExpr::Cast { expr, .. } => is_length_preserving_ae(*expr, arena),
            AExpr::Eval { expr, variant, .. } => {
                variant.is_length_preserving() && is_length_preserving_ae(*expr, arena)
            },
            #[cfg(feature = "dtype-struct")]
            AExpr::StructEval { expr, .. } => is_length_preserving_ae(*expr, arena),
            AExpr::Sort { expr, .. } => is_length_preserving_ae(*expr, arena),
            AExpr::Gather {
                expr: _,
                idx,
                returns_scalar,
                null_on_oob: _,
            } => !returns_scalar && is_length_preserving_ae(*idx, arena),
            AExpr::SortBy { expr, by, .. } => broadcasting_input_length_preserving(
                std::iter::once(*expr).chain(by.iter().copied()),
                arena,
            ),

            AExpr::Explode { .. } | AExpr::Filter { .. } | AExpr::Slice { .. } => false,
        }
    }

    /// Is the top-level expression fallible based on the data values.
    pub fn is_fallible_top_level(&self, arena: &Arena<AExpr>) -> bool {
        #[allow(clippy::collapsible_match, clippy::match_like_matches_macro)]
        match self {
            AExpr::Function {
                input, function, ..
            } => match function {
                IRFunctionExpr::ListExpr(f) => match f {
                    IRListFunction::Get(false) => true,
                    #[cfg(feature = "list_gather")]
                    IRListFunction::Gather(false) => true,
                    _ => false,
                },
                #[cfg(feature = "dtype-array")]
                IRFunctionExpr::ArrayExpr(f) => match f {
                    IRArrayFunction::Get(false) => true,
                    _ => false,
                },
                #[cfg(feature = "replace")]
                IRFunctionExpr::ReplaceStrict { .. } => true,
                #[cfg(all(feature = "strings", feature = "temporal"))]
                IRFunctionExpr::StringExpr(f) => match f {
                    IRStringFunction::Strptime(_, strptime_options) => {
                        debug_assert!(input.len() <= 2);

                        let ambiguous_arg_is_infallible_scalar = input
                            .get(1)
                            .map(|x| arena.get(x.node()))
                            .is_some_and(|ae| match ae {
                                AExpr::Literal(lv) => {
                                    lv.extract_str().is_some_and(|ambiguous| match ambiguous {
                                        "earliest" | "latest" | "null" => true,
                                        "raise" => false,
                                        v => {
                                            if cfg!(debug_assertions) {
                                                panic!("unhandled parameter to ambiguous: {v}")
                                            }
                                            false
                                        },
                                    })
                                },
                                _ => false,
                            });

                        let ambiguous_is_fallible = !ambiguous_arg_is_infallible_scalar;

                        !matches!(arena.get(input[0].node()), AExpr::Literal(_))
                            && (strptime_options.strict || ambiguous_is_fallible)
                    },
                    _ => false,
                },
                _ => false,
            },
            AExpr::Cast {
                expr,
                dtype: _,
                options: CastOptions::Strict,
            } => !matches!(arena.get(*expr), AExpr::Literal(_)),
            _ => false,
        }
    }
}

#[recursive::recursive]
pub fn deep_clone_ae(ae: Node, arena: &mut Arena<AExpr>) -> Node {
    let slf = arena.get(ae).clone();

    let mut children = vec![];
    slf.children_rev(&mut children);
    for child in &mut children {
        *child = deep_clone_ae(*child, arena);
    }
    children.reverse();

    arena.add(slf.replace_children(&children))
}
