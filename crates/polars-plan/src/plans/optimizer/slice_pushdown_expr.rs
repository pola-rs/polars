use std::ops::ControlFlow;

use polars_core::chunked_array::cast::CastOptions;
use polars_utils::UnitVec;
use polars_utils::collection::{Collection, CollectionWrap, MappedCollection};
use polars_utils::slice_enum::Slice;

use super::*;
use crate::plans::projection_height::{ExprProjectionHeight, aexpr_projection_height};
use crate::traversal::visitor::{FnVisitors, SubtreeVisit};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExprSlice {
    Extracted(Slice),
    Opaque { offset: Node, len: Node },
}

impl ExprSlice {
    pub(crate) fn from_nodes(offset: Node, len: Node, arena: &Arena<AExpr>) -> Self {
        if let (AExpr::Literal(offset), AExpr::Literal(len)) = (arena.get(offset), arena.get(len))
            && offset.is_scalar()
            && len.is_scalar()
            && let (LiteralValue::Scalar(offset), LiteralValue::Scalar(len)) =
                (offset.clone().materialize(), len.clone().materialize())
            && let Ok(offset) = offset.cast_with_options(&DataType::Int64, CastOptions::NonStrict)
            && let AnyValue::Int64(offset) = offset.as_any_value()
            && let Some(len) = match len.as_any_value() {
                AnyValue::Null => Some(None),
                len => len
                    .extract::<u64>()
                    .and_then(|len| usize::try_from(len).ok())
                    .map(Some),
            }
            && let Some(slice) = Slice::new(offset, len)
            && slice.is_within_i64()
        {
            return Self::Extracted(slice);
        };

        ExprSlice::Opaque { offset, len }
    }

    fn combine_with_inner_slice(&self, inner_slice: &Self) -> Option<Self> {
        let (Self::Extracted(outer), Self::Extracted(inner)) = (self, inner_slice) else {
            return None;
        };

        Some(Self::Extracted(inner.reslice(*outer)?))
    }

    pub(crate) fn slice_arena_node(&self, ae_node: Node, expr_arena: &mut Arena<AExpr>) {
        let (offset, length) = match self {
            Self::Extracted(slice) => {
                let (offset, len) = slice.to_signed_offset_len();
                let len = match len {
                    Some(len) => AnyValue::UInt64(len as u64),
                    None => AnyValue::Null,
                };
                (
                    expr_arena.add(AExpr::Literal(LiteralValue::Scalar(Scalar::new(
                        DataType::Int64,
                        AnyValue::Int64(offset),
                    )))),
                    expr_arena.add(AExpr::Literal(LiteralValue::Scalar(Scalar::new(
                        DataType::UInt64,
                        len,
                    )))),
                )
            },
            Self::Opaque { offset, len } => (*offset, *len),
        };

        let ae_node_copy = expr_arena.add(expr_arena.get(ae_node).clone());

        expr_arena.replace(
            ae_node,
            AExpr::Slice {
                input: ae_node_copy,
                offset,
                length,
            },
        );
    }

    fn to_extracted_slice(&self) -> Option<Slice> {
        match self {
            Self::Extracted(v) => Some(*v),
            Self::Opaque { .. } => None,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct State {
    candidate_push_locations: UnitVec<Node>,
    height: ExprProjectionHeight,
}

impl SlicePushDown {
    pub(crate) fn aexpr_slice_pushdown_rec(
        &mut self,
        current_ae_node: Node,
        expr_arena: &mut Arena<AExpr>,
        common_col_limit: &mut Option<Slice>,
        col_hit_count: &mut Option<usize>,
        maintain_errors: bool,
    ) {
        let mut visitor = FnVisitors::new(
            State::default,
            |_, _, _| ControlFlow::Continue(SubtreeVisit::Visit),
            |ae_node, arena, edges| {
                edges.outputs()[0] = aexpr_slice_pushdown_top(
                    ae_node,
                    &mut *edges.inputs(),
                    arena,
                    common_col_limit,
                    col_hit_count,
                    maintain_errors,
                );

                ControlFlow::<()>::Continue(())
            },
        );

        aexpr_tree_traversal(
            current_ae_node,
            expr_arena,
            self.ae_nodes_scratch.get(),
            self.ae_slice_pd_state_scratch.get(),
            &mut visitor,
        )
        .continue_value()
        .unwrap();
    }
}

/// Flips a comparison operator so that `n <op> len()` becomes `len() <flip(op)> n`.
fn flip_comparison(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        other => other,
    }
}

/// If `ae` is a comparison between `len()` and a non-negative integer literal, returns the
/// leading-rows slice that needs to be materialized to answer it.
fn len_cmp_head_slice(ae: &AExpr, expr_arena: &Arena<AExpr>) -> Option<Slice> {
    let AExpr::BinaryExpr { left, op, right } = ae else {
        return None;
    };

    let (op, literal_node) = if matches!(expr_arena.get(*left), AExpr::Len) {
        (*op, *right)
    } else if matches!(expr_arena.get(*right), AExpr::Len) {
        (flip_comparison(*op), *left)
    } else {
        return None;
    };

    let AExpr::Literal(lv) = expr_arena.get(literal_node) else {
        return None;
    };
    let n: u64 = lv.extract_i64().ok()?.try_into().ok()?;

    let head_len = match op {
        // `len() < n` doesn't need the extra row: `head(n).len() < n` is
        // already equivalent to `len() < n`.
        Operator::Lt => n,
        Operator::Eq | Operator::NotEq | Operator::LtEq | Operator::Gt | Operator::GtEq => {
            n.saturating_add(1)
        },
        _ => return None,
    };

    Some(Slice::head(usize::try_from(head_len).ok()?))
}

/// The inputs of `ae` that a slice of `ae` can be pushed into. For an elementwise `ae` these are
/// the inputs [`is_elementwise`] reports: row i of the output depends only on row i of each of
/// these inputs.
fn slice_pushdown_inputs(ae: &AExpr, expr_arena: &Arena<AExpr>) -> UnitVec<Node> {
    let mut inputs = UnitVec::new();
    match ae {
        // The offset and length are parameters, the rows come from `input`.
        AExpr::Slice { input, .. }
        | AExpr::Agg(IRAggExpr::First(input) | IRAggExpr::Last(input)) => inputs.push(*input),
        // A field that is not elementwise can read all rows of the struct input through
        // `pl.field`.
        #[cfg(feature = "dtype-struct")]
        AExpr::StructEval { evaluation, .. } if !all_elementwise(evaluation, expr_arena) => {},
        _ => _ = is_elementwise(&mut inputs, ae, expr_arena),
    }
    inputs
}

fn aexpr_slice_pushdown_top(
    current_ae_node: Node,
    input_states: &mut dyn Collection<State>,
    expr_arena: &mut Arena<AExpr>,
    common_col_limit: &mut Option<Slice>,
    col_hit_count: &mut Option<usize>,
    maintain_errors: bool,
) -> State {
    use ExprProjectionHeight as H;

    let mut input_states = CollectionWrap::new(input_states);
    let ae = expr_arena.get(current_ae_node);

    // We propagate upwards the position of the last non-elementwise node(s), these represent
    // the deepest position that a slice can be pushed to from the current node.
    //
    // E.g.
    //
    //                ▲
    //                │
    //            H::Column
    //                │
    //        ┌──────►==◄─────┐  (current expr)
    //        │                │
    //        │           H::Co│umn
    //        │                │
    //        │                │
    //    H::Column     ┌────►+◄──┐
    //        │         │          │
    //        │         H::Column  H::Scalar
    //    ┌──────┐   ┌──────┐      │
    //    │col(A)│   │col(B)│      1
    //    └──────┘   └──────┘
    //
    // Push candidates are `col(A)` and `col(B)`.
    let slice_inputs = slice_pushdown_inputs(ae, expr_arena);
    let mut inputs = UnitVec::new();
    ae.inputs(&mut inputs);
    let mut slice_input_idxs = (0..inputs.len()).filter(|&i| slice_inputs.contains(&inputs[i]));
    let mut state = slice_input_idxs
        .find(|&i| input_states[i].height != H::Scalar)
        .map_or(State::default(), |i| {
            let s = &mut input_states[i];
            State {
                candidate_push_locations: std::mem::take(&mut s.candidate_push_locations),
                height: s.height,
            }
        });

    for remainder in slice_input_idxs.map(|i| &input_states[i]) {
        match (state.height, remainder.height) {
            (_, H::Scalar) => {},
            (H::Column, H::Column) => state
                .candidate_push_locations
                .extend(remainder.candidate_push_locations.iter().copied()),
            _ => {
                state.candidate_push_locations.clear();
                break;
            },
        }
    }

    state.height = aexpr_projection_height(
        ae,
        Some(&mut MappedCollection::new(
            &mut *input_states,
            |s| &s.height,
            |s| &mut s.height,
        )),
    )
    .unwrap();

    match ae {
        AExpr::Column(_) | AExpr::Len => *col_hit_count = col_hit_count.map(|x| x + 1),
        // A window runs its function per group and keeps the frame height, so it needs all
        // rows of the frame.
        AExpr::Over { .. } => *col_hit_count = None,
        _ => {},
    }

    let len_cmp_slice = len_cmp_head_slice(ae, expr_arena)
        .map(|slice| (expr_arena.add(AExpr::Column(PlSmallStr::EMPTY)), slice));
    if state.candidate_push_locations.is_empty()
        && let Some((dummy_node, _)) = len_cmp_slice
    {
        state.candidate_push_locations.push(dummy_node);
    }

    'pushdown_current_slice: {
        if state.candidate_push_locations.is_empty() {
            break 'pushdown_current_slice;
        }

        let (current_input_node, current_slice) = if let Some((dummy_node, slice)) = len_cmp_slice {
            (dummy_node, ExprSlice::Extracted(slice))
        } else {
            match expr_arena.get(current_ae_node) {
                AExpr::Slice {
                    input,
                    offset,
                    length,
                } => (*input, ExprSlice::from_nodes(*offset, *length, expr_arena)),
                AExpr::Agg(IRAggExpr::First(input)) => {
                    (*input, ExprSlice::Extracted(Slice::head(1)))
                },
                AExpr::Agg(IRAggExpr::Last(input)) => {
                    (*input, ExprSlice::Extracted(Slice::tail(1)))
                },
                _ => break 'pushdown_current_slice,
            }
        };

        for candidate_node in state.candidate_push_locations.iter().copied() {
            let mut extracted_slice: Option<Slice> = current_slice.to_extracted_slice();
            // Set to 0 if updating an existing slice.
            let mut hit_count_contribution: usize = 1;
            let mut sliced_at_candidate = false;

            match expr_arena.get(candidate_node) {
                AExpr::Slice {
                    input: inner_input,
                    offset: inner_offset,
                    length: inner_length,
                } => {
                    hit_count_contribution = 0;
                    let inner_slice =
                        ExprSlice::from_nodes(*inner_offset, *inner_length, expr_arena);

                    if let Some(combined) = current_slice.combine_with_inner_slice(&inner_slice) {
                        // Update existing slice
                        let inner_input = *inner_input;

                        extracted_slice = combined.to_extracted_slice();

                        if combined != inner_slice {
                            expr_arena.replace(candidate_node, expr_arena.get(inner_input).clone());
                            combined.slice_arena_node(candidate_node, expr_arena);
                        }

                        sliced_at_candidate = true;
                    } else {
                        if candidate_node != current_input_node {
                            current_slice.slice_arena_node(candidate_node, expr_arena);
                            sliced_at_candidate = true;
                        }
                    }
                },
                _ => {
                    if candidate_node != current_input_node {
                        current_slice.slice_arena_node(candidate_node, expr_arena);
                        sliced_at_candidate = true;
                    }
                },
            };

            if sliced_at_candidate && matches!(expr_arena.get(current_ae_node), AExpr::Slice { .. })
            {
                expr_arena.replace(current_ae_node, expr_arena.get(current_input_node).clone());
            }

            let slice_input = if sliced_at_candidate {
                let AExpr::Slice { input, .. } = expr_arena.get(candidate_node) else {
                    unreachable!()
                };
                *input
            } else {
                current_input_node
            };

            'update_common_slice: {
                let AExpr::Column(_) = expr_arena.get(slice_input) else {
                    break 'update_common_slice;
                };

                *col_hit_count = col_hit_count.and_then(|x| x.checked_sub(hit_count_contribution));

                let Some(extracted_slice) = extracted_slice else {
                    *col_hit_count = None;
                    break 'update_common_slice;
                };

                if let Some(existing) = common_col_limit {
                    let Some(new) = Slice::try_new_enclosing(*existing, extracted_slice) else {
                        *col_hit_count = None;
                        break 'update_common_slice;
                    };
                    *existing = new;
                } else {
                    *common_col_limit = Some(extracted_slice)
                }
            }
        }
    }

    let current_ae = expr_arena.get(current_ae_node);

    if !current_ae.is_elementwise_top_level()
        || (maintain_errors && current_ae.is_fallible_top_level_ignoring_overflow(expr_arena))
    {
        state.candidate_push_locations.clear();
    }

    if state.candidate_push_locations.is_empty() {
        state.candidate_push_locations.push(current_ae_node);
    }

    state
}
