use std::ops::ControlFlow;

use polars_core::chunked_array::cast::CastOptions;
use polars_utils::UnitVec;
use polars_utils::collection::{Collection, CollectionWrap, MappedCollection};

use super::*;
use crate::plans::optimizer::slice_pushdown_lp::{
    State as ExtractedSlice, combine_outer_inner_slice,
};
use crate::plans::projection_height::{ExprProjectionHeight, aexpr_projection_height};
use crate::traversal::visitor::{FnVisitors, SubtreeVisit};

#[derive(Debug, Clone, Copy)]
pub(crate) enum CommonSlice {
    Positive {
        /// Invariant: >=0
        min_offset: i64,
        /// `max(s.offset + s.len) for s in slices`
        /// Invariant: >=0
        max_end: IdxSize,
    },
    Negative {
        /// Invariant: <0
        min_offset: i64,
        /// Invariant: <=0
        max_end: i64,
    },
}

impl CommonSlice {
    pub(crate) fn common_limit_with(self, other: Self) -> Option<Self> {
        Some(match (self, other) {
            (
                CommonSlice::Positive {
                    min_offset,
                    max_end,
                },
                CommonSlice::Positive {
                    min_offset: r_min_offset,
                    max_end: r_max_end,
                },
            ) => CommonSlice::Positive {
                min_offset: i64::min(min_offset, r_min_offset),
                max_end: IdxSize::max(max_end, r_max_end),
            },
            (
                CommonSlice::Negative {
                    min_offset,
                    max_end,
                },
                CommonSlice::Negative {
                    min_offset: r_min_offset,
                    max_end: r_max_end,
                },
            ) => CommonSlice::Negative {
                min_offset: i64::min(min_offset, r_min_offset),
                max_end: i64::max(max_end, r_max_end),
            },
            _ => return None,
        })
    }

    pub(crate) fn from_slice(slice: ExtractedSlice) -> Self {
        if slice.offset >= 0 {
            CommonSlice::Positive {
                min_offset: slice.offset,
                max_end: slice
                    .offset
                    .try_into()
                    .unwrap_or(IdxSize::MAX)
                    .saturating_add(slice.len),
            }
        } else {
            CommonSlice::Negative {
                min_offset: slice.offset,
                #[cfg_attr(
                    not(feature = "bigidx"),
                    expect(clippy::unnecessary_fallible_conversions)
                )]
                max_end: i64::min(
                    0,
                    slice
                        .offset
                        .saturating_add(slice.len.try_into().unwrap_or(i64::MAX)),
                ),
            }
        }
    }

    /// Compute the IR slice that can be inserted to apply before the projections. Also returns
    /// an offset correction to be subtracted from the direct column slices in the expression tree
    /// due to the added IR slice.
    pub(crate) fn to_slice_with_correction(self) -> (ExtractedSlice, i64) {
        match self {
            Self::Positive {
                min_offset,
                max_end,
            } => {
                assert!(min_offset >= 0);
                let len = max_end - min_offset.try_into().unwrap_or(IdxSize::MAX);

                (
                    ExtractedSlice {
                        offset: min_offset,
                        len,
                    },
                    min_offset,
                )
            },
            Self::Negative {
                min_offset,
                max_end,
            } => {
                assert!(min_offset < 0);
                assert!(max_end <= 0);

                let len = min_offset.abs_diff(max_end);
                #[cfg_attr(feature = "bigidx", expect(clippy::useless_conversion))]
                let len: IdxSize = len.try_into().unwrap_or(IdxSize::MAX);

                (
                    ExtractedSlice {
                        offset: min_offset,
                        len,
                    },
                    max_end,
                )
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Slice {
    Extracted(ExtractedSlice),
    Opaque { offset: Node, len: Node },
}

impl Slice {
    pub(crate) fn from_nodes(offset: Node, len: Node, arena: &Arena<AExpr>) -> Self {
        if let (AExpr::Literal(offset), AExpr::Literal(len)) = (arena.get(offset), arena.get(len))
            && offset.is_scalar()
            && len.is_scalar()
            && let (LiteralValue::Scalar(offset), LiteralValue::Scalar(len)) =
                (offset.clone().materialize(), len.clone().materialize())
            && let (Ok(offset), Ok(len)) = (
                offset.cast_with_options(&DataType::Int64, CastOptions::NonStrict),
                len.cast_with_options(&DataType::IDX_DTYPE, CastOptions::NonStrict),
            )
            && let AnyValue::Int64(offset) = offset.as_any_value()
            && let Some(len) = match len.as_any_value() {
                AnyValue::Null => {
                    if offset < 0 {
                        (-offset).try_into().ok()
                    } else {
                        Some(IdxSize::MAX)
                    }
                },
                #[cfg(feature = "bigidx")]
                AnyValue::UInt64(len) => Some(len),
                #[cfg(not(feature = "bigidx"))]
                AnyValue::UInt32(len) => Some(len),
                _ => None,
            }
        {
            return Self::Extracted(ExtractedSlice { offset, len });
        };

        Slice::Opaque { offset, len }
    }

    fn combine_with_inner_slice(&self, inner_slice: &Self) -> Option<Self> {
        use slice_pushdown_lp::State;

        let Self::Extracted(ExtractedSlice { offset, len }) = self else {
            return None;
        };
        let outer = State {
            offset: *offset,
            len: *len,
        };

        let Self::Extracted(ExtractedSlice { offset, len }) = inner_slice else {
            return None;
        };
        let inner = State {
            offset: *offset,
            len: *len,
        };

        let State { offset, len } = combine_outer_inner_slice(outer, inner)?;

        Some(Self::Extracted(ExtractedSlice { offset, len }))
    }

    pub(crate) fn slice_arena_node(&self, ae_node: Node, expr_arena: &mut Arena<AExpr>) {
        let (offset, length) = match self {
            Self::Extracted(ExtractedSlice { offset, len }) => (
                expr_arena.add(AExpr::Literal(LiteralValue::Scalar(Scalar::new(
                    DataType::Int64,
                    AnyValue::Int64(*offset),
                )))),
                expr_arena.add(AExpr::Literal(LiteralValue::Scalar(Scalar::new(
                    DataType::IDX_DTYPE,
                    {
                        #[cfg(not(feature = "bigidx"))]
                        {
                            AnyValue::UInt32(*len)
                        }
                        #[cfg(feature = "bigidx")]
                        {
                            AnyValue::UInt64(*len)
                        }
                    },
                )))),
            ),
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

    fn to_extracted_slice(&self) -> Option<ExtractedSlice> {
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
        common_col_limit: &mut Option<CommonSlice>,
        col_hit_count: &mut Option<usize>,
        all_slice_ae_nodes_with_direct_col_input: &mut PlHashSet<Node>,
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
                    all_slice_ae_nodes_with_direct_col_input,
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

fn aexpr_slice_pushdown_top(
    current_ae_node: Node,
    input_states: &mut dyn Collection<State>,
    expr_arena: &mut Arena<AExpr>,
    common_col_limit: &mut Option<CommonSlice>,
    col_hit_count: &mut Option<usize>,
    all_slice_ae_nodes_with_direct_col_input: &mut PlHashSet<Node>,
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
    let idx = input_states
        .iter()
        .position(|s| s.height != H::Scalar)
        .unwrap_or(input_states.len());
    let mut state = input_states
        .get_mut(idx)
        .map_or(State::default(), |s| State {
            candidate_push_locations: std::mem::take(&mut s.candidate_push_locations),
            height: s.height,
        });

    for remainder in (idx + 1..input_states.len()).map(|i| &input_states[i]) {
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

    if let AExpr::Column(_) = ae {
        *col_hit_count = col_hit_count.map(|x| x + 1);
    }

    'pushdown_current_slice: {
        if state.candidate_push_locations.is_empty() {
            break 'pushdown_current_slice;
        }

        let (current_input_node, current_slice) = match ae {
            AExpr::Slice {
                input,
                offset,
                length,
            } => (*input, Slice::from_nodes(*offset, *length, expr_arena)),
            AExpr::Agg(IRAggExpr::First(input)) => (
                *input,
                Slice::Extracted(ExtractedSlice { offset: 0, len: 1 }),
            ),
            AExpr::Agg(IRAggExpr::Last(input)) => (
                *input,
                Slice::Extracted(ExtractedSlice { offset: -1, len: 1 }),
            ),
            _ => break 'pushdown_current_slice,
        };

        for candidate_node in state.candidate_push_locations.iter().copied() {
            let mut extracted_slice: Option<ExtractedSlice> = current_slice.to_extracted_slice();
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
                    let inner_slice = Slice::from_nodes(*inner_offset, *inner_length, expr_arena);

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

                if sliced_at_candidate {
                    all_slice_ae_nodes_with_direct_col_input.insert(candidate_node);
                } else if matches!(expr_arena.get(current_ae_node), AExpr::Slice { .. }) {
                    all_slice_ae_nodes_with_direct_col_input.insert(current_ae_node);
                }

                *col_hit_count = col_hit_count.and_then(|x| x.checked_sub(hit_count_contribution));

                let Some(extracted_slice) = extracted_slice else {
                    *col_hit_count = None;
                    break 'update_common_slice;
                };

                let extracted_limit = CommonSlice::from_slice(extracted_slice);

                if let Some(existing) = common_col_limit {
                    let Some(new) = extracted_limit.common_limit_with(*existing) else {
                        *col_hit_count = None;
                        break 'update_common_slice;
                    };
                    *existing = new;
                } else {
                    *common_col_limit = Some(extracted_limit)
                }
            }
        }
    }

    let current_ae = expr_arena.get(current_ae_node);

    if !current_ae.is_elementwise_top_level()
        || (maintain_errors && current_ae.is_fallible_top_level(expr_arena))
    {
        state.candidate_push_locations.clear();
    }

    if state.candidate_push_locations.is_empty() {
        state.candidate_push_locations.push(current_ae_node);
    }

    state
}
