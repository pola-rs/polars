use polars_utils::arena::{Arena, Node};

use crate::dsl::WindowMapping;
use crate::plans::AExpr;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ExprProjectionHeight {
    Column,
    Scalar,
    #[default]
    Unknown,
}

impl ExprProjectionHeight {
    pub fn zip_with(self, other: Self) -> Self {
        use ExprProjectionHeight::*;

        match (self, other) {
            (Scalar, v) | (v, Scalar) => v,
            (Unknown, _) | (_, Unknown) => Unknown,
            (l, r) => {
                if l == r {
                    l
                } else {
                    Unknown
                }
            },
        }
    }

    pub fn zipped_projection_height(iter: impl IntoIterator<Item = Self>) -> Self {
        let mut iter = iter.into_iter();

        let Some(first) = iter.next() else {
            return Self::Unknown;
        };

        iter.fold(first, |acc, h| acc.zip_with(h))
    }
}

#[recursive::recursive]
pub fn aexpr_projection_height_rec(
    ae_node: Node,
    expr_arena: &Arena<AExpr>,
    stack: &mut Vec<Node>,
    inputs_stack: &mut Vec<ExprProjectionHeight>,
) -> ExprProjectionHeight {
    let ae = expr_arena.get(ae_node);

    let base_stack_len = stack.len();
    let base_inputs_stack_len = inputs_stack.len();
    ae.inputs(stack);
    let num_inputs = stack.len() - base_stack_len;

    for i in base_stack_len..stack.len() {
        let h = aexpr_projection_height_rec(stack[i], expr_arena, stack, inputs_stack);
        inputs_stack.push(h);
    }

    assert_eq!(stack.len(), base_stack_len + num_inputs);
    stack.truncate(base_stack_len);

    assert_eq!(inputs_stack.len(), base_inputs_stack_len + num_inputs);
    let mut h = ExprProjectionHeight::Unknown;
    aexpr_projection_height(
        expr_arena.get(ae_node),
        &inputs_stack[base_inputs_stack_len..],
        &mut h,
    );
    inputs_stack.truncate(base_inputs_stack_len);

    h
}

pub fn aexpr_projection_height(
    aexpr: &AExpr,
    input_heights: &[ExprProjectionHeight],
    output_height: &mut ExprProjectionHeight,
) {
    use AExpr::*;
    use ExprProjectionHeight as H;

    *output_height = match aexpr {
        Column(_) => H::Column,

        // FIXME
        // Not technically correct but we rely on this to determine if an eval expr
        // is length-preserving (which looks for "column-length").
        Element => H::Column,
        #[cfg(feature = "dtype-struct")]
        StructField(_) => H::Unknown,
        Literal(lv) => {
            if lv.is_scalar() {
                H::Scalar
            } else {
                H::Unknown
            }
        },

        Eval { .. } => input_heights[0],
        #[cfg(feature = "dtype-struct")]
        StructEval { .. } => input_heights[0],

        Filter { .. } | Slice { .. } | Explode { .. } => H::Unknown,

        Agg(_) | AnonymousAgg { .. } => H::Scalar,
        Len => H::Scalar,

        BinaryExpr { .. } => {
            let [l, r] = input_heights.try_into().unwrap();
            l.zip_with(r)
        },
        Ternary { .. } => {
            let [pred, truthy, falsy] = input_heights.try_into().unwrap();
            pred.zip_with(truthy).zip_with(falsy)
        },

        Cast { .. } | Sort { .. } => {
            let [h] = input_heights.try_into().unwrap();
            h
        },

        SortBy { .. } => input_heights[0],

        Gather { returns_scalar, .. } => {
            if *returns_scalar {
                H::Scalar
            } else {
                H::Unknown
            }
        },

        AExpr::Function { options, .. } | AExpr::AnonymousFunction { options, .. } => {
            if options.flags.returns_scalar() {
                H::Scalar
            } else if options.flags.is_elementwise() || options.flags.is_length_preserving() {
                H::zipped_projection_height(input_heights.iter().copied())
            } else {
                H::Unknown
            }
        },

        #[cfg(feature = "dynamic_group_by")]
        Rolling { .. } => H::Column,
        Over { mapping, .. } => {
            if matches!(mapping, WindowMapping::Explode) {
                H::Unknown
            } else {
                H::Column
            }
        },
    };
}
