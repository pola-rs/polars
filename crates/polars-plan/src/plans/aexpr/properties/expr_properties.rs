use std::ops::ControlFlow;

use polars_utils::aliases::PlIndexMap;
use polars_utils::idx_vec::UnitVec;
use polars_utils::scratch_vec::ScratchVec;
use polars_utils::unitvec;

use super::super::*;
use super::general::is_input_independent_scalar_top_level;
use crate::plans::projection_height::{ExprHeightVisitor, ExprProjectionHeight};
use crate::traversal::edge_provider::NodeEdgesProvider;
use crate::traversal::visitor::{NodeVisitor, SubtreeVisit};

/// Memoized properties of the expressions in an `Arena<AExpr>`.
///
/// A `Node` must keep referring to the same expression for as long as `ExprProperties` is used.
#[derive(Default)]
pub struct ExprProperties {
    is_elementwise: PlIndexMap<Node, bool>,
    is_row_separable: PlIndexMap<Node, bool>,
    is_input_independent: PlIndexMap<Node, bool>,
    is_input_independent_scalar: PlIndexMap<Node, bool>,
    is_frame_dependent: PlIndexMap<Node, bool>,
    is_length_preserving: PlIndexMap<Node, bool>,
    node_scratch: ScratchVec<Node>,
    height_scratch: ScratchVec<ExprProjectionHeight>,
}

impl ExprProperties {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            is_elementwise: PlIndexMap::with_capacity(capacity),
            is_row_separable: PlIndexMap::with_capacity(capacity),
            is_input_independent: PlIndexMap::with_capacity(capacity),
            is_input_independent_scalar: PlIndexMap::with_capacity(capacity),
            is_frame_dependent: PlIndexMap::with_capacity(capacity),
            is_length_preserving: PlIndexMap::with_capacity(capacity),
            node_scratch: ScratchVec::default(),
            height_scratch: ScratchVec::default(),
        }
    }

    /// Return whether the expression at `node` is elementwise, as [`is_elementwise_rec`].
    #[recursive::recursive]
    pub fn is_elementwise(&mut self, node: Node, expr_arena: &Arena<AExpr>) -> bool {
        if let Some(&ret) = self.is_elementwise.get(&node) {
            return ret;
        }

        // `is_elementwise` skips the single-literal right-hand side of `is_in`.
        let mut inputs = unitvec![];
        let ret = is_elementwise(&mut inputs, expr_arena.get(node), expr_arena)
            && inputs.iter().all(|&n| self.is_elementwise(n, expr_arena));

        self.is_elementwise.insert(node, ret);
        ret
    }

    /// Return whether the expression at `node` is row-separable: evaluating it on batches of rows
    /// and concatenating the results gives the same output as evaluating it on the whole frame,
    /// where a scalar result is broadcast to the frame's height.
    ///
    /// Every node must be row-separable, and the inputs it zips must line up within a batch (see
    /// [`Self::zipped_inputs_line_up`]). A node that isn't elementwise must also have an input that
    /// reads the frame: `lit([1, 2]).explode()` gives two rows per batch.
    #[recursive::recursive]
    pub fn is_row_separable(&mut self, node: Node, expr_arena: &Arena<AExpr>) -> bool {
        if let Some(&ret) = self.is_row_separable.get(&node) {
            return ret;
        }

        let ae = expr_arena.get(node);
        let mut inputs = unitvec![];
        ae.inputs_rev(&mut inputs);
        let ret = ae.is_row_separable_top_level()
            && inputs.iter().all(|&n| self.is_row_separable(n, expr_arena))
            && self.zipped_inputs_line_up(&inputs, expr_arena)
            && (ae.is_elementwise_top_level()
                || inputs
                    .iter()
                    .any(|&n| self.is_frame_dependent(n, expr_arena)));

        self.is_row_separable.insert(node, ret);
        ret
    }

    /// Return whether `inputs`, zipped within a batch of rows, line up as they do over the whole
    /// frame: all are elementwise, or all but one are input-independent scalars.
    pub fn zipped_inputs_line_up(&mut self, inputs: &[Node], expr_arena: &Arena<AExpr>) -> bool {
        let non_scalars: UnitVec<Node> = inputs
            .iter()
            .copied()
            .filter(|&n| !self.is_input_independent_scalar(n, expr_arena))
            .collect();
        non_scalars.len() <= 1
            || non_scalars
                .iter()
                .all(|&n| self.is_elementwise(n, expr_arena))
    }

    /// Return whether the expression at `node` reads a column or the frame's length.
    #[recursive::recursive]
    fn is_frame_dependent(&mut self, node: Node, expr_arena: &Arena<AExpr>) -> bool {
        if let Some(&ret) = self.is_frame_dependent.get(&node) {
            return ret;
        }

        let ae = expr_arena.get(node);
        let ret = matches!(ae, AExpr::Column(_) | AExpr::Len) || {
            let mut inputs: UnitVec<Node> = unitvec![];
            ae.inputs_rev(&mut inputs);
            inputs
                .iter()
                .any(|&n| self.is_frame_dependent(n, expr_arena))
        };

        self.is_frame_dependent.insert(node, ret);
        ret
    }

    /// Return whether the expression at `node` can be evaluated without the input frame. Unlike
    /// [`Self::is_input_independent_scalar`], the result need not be a scalar.
    #[recursive::recursive]
    pub fn is_input_independent(&mut self, node: Node, expr_arena: &Arena<AExpr>) -> bool {
        if let Some(&ret) = self.is_input_independent.get(&node) {
            return ret;
        }

        let ret = match expr_arena.get(node) {
            // Handled separately in `Eval`.
            AExpr::Element | AExpr::EvalColumn(_) => unreachable!(),
            #[cfg(feature = "dtype-struct")]
            AExpr::StructField(_) => false,
            AExpr::Explode { expr: inner, .. }
            | AExpr::Cast {
                expr: inner,
                dtype: _,
                options: _,
            }
            | AExpr::Sort {
                expr: inner,
                options: _,
            } => self.is_input_independent(*inner, expr_arena),
            AExpr::Column(_) => false,

            AExpr::Literal(_) => true,
            AExpr::BinaryExpr { left, op: _, right } => {
                self.is_input_independent(*left, expr_arena)
                    && self.is_input_independent(*right, expr_arena)
            },
            AExpr::Gather {
                expr,
                idx,
                returns_scalar: _,
                null_on_oob: _,
            } => {
                self.is_input_independent(*expr, expr_arena)
                    && self.is_input_independent(*idx, expr_arena)
            },
            AExpr::SortBy {
                expr,
                by,
                sort_options: _,
            } => {
                self.is_input_independent(*expr, expr_arena)
                    && by
                        .iter()
                        .all(|expr| self.is_input_independent(*expr, expr_arena))
            },
            AExpr::Filter { input, by } => {
                self.is_input_independent(*input, expr_arena)
                    && self.is_input_independent(*by, expr_arena)
            },
            AExpr::Agg(agg_expr) => match agg_expr.get_input() {
                NodeInputs::Leaf => true,
                NodeInputs::Single(expr) => self.is_input_independent(expr, expr_arena),
                NodeInputs::Many(exprs) => exprs
                    .iter()
                    .all(|expr| self.is_input_independent(*expr, expr_arena)),
            },
            AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                self.is_input_independent(*predicate, expr_arena)
                    && self.is_input_independent(*truthy, expr_arena)
                    && self.is_input_independent(*falsy, expr_arena)
            },
            AExpr::AnonymousFunction {
                input,
                function: _,
                options: _,
                fmt_str: _,
            }
            | AExpr::AnonymousAgg {
                input,
                function: _,
                fmt_str: _,
            }
            | AExpr::Function {
                input,
                function: _,
                options: _,
            } => input
                .iter()
                .all(|expr| self.is_input_independent(expr.node(), expr_arena)),
            AExpr::Eval {
                expr,
                evaluation: _,
                variant: _,
                columns,
            } => {
                self.is_input_independent(*expr, expr_arena)
                    && columns
                        .iter()
                        .all(|c| self.is_input_independent(c.node(), expr_arena))
            },
            #[cfg(feature = "dtype-struct")]
            AExpr::StructEval {
                expr,
                evaluation,
                variant: _,
            } => {
                // The `StructField`s of `evaluation` read the struct of `expr`, not the input. The
                // bodies of `Eval`s are skipped (`inputs_rev`): a `len()` there is a list's length.
                self.is_input_independent(*expr, expr_arena)
                    && evaluation.iter().all(|expr| {
                        let mut stack = vec![expr.node()];
                        while let Some(node) = stack.pop() {
                            let ae = expr_arena.get(node);
                            if matches!(ae, AExpr::Column(_) | AExpr::Len) {
                                return false;
                            }
                            ae.inputs_rev(&mut stack);
                        }
                        true
                    })
            },
            AExpr::Slice {
                input,
                offset,
                length,
            } => {
                self.is_input_independent(*input, expr_arena)
                    && self.is_input_independent(*offset, expr_arena)
                    && self.is_input_independent(*length, expr_arena)
            },
            AExpr::Len => false,
            // The length of a window's output depends on the input's height.
            #[cfg(feature = "dynamic_group_by")]
            AExpr::Rolling { .. } => false,
            AExpr::Over { .. } => false,
        };

        self.is_input_independent.insert(node, ret);
        ret
    }

    /// Return whether the expression at `node` is an input-independent scalar, as
    /// [`is_input_independent_scalar_rec`].
    #[recursive::recursive]
    pub fn is_input_independent_scalar(&mut self, node: Node, expr_arena: &Arena<AExpr>) -> bool {
        if let Some(&ret) = self.is_input_independent_scalar.get(&node) {
            return ret;
        }

        let ae = expr_arena.get(node);
        let ret = is_input_independent_scalar_top_level(ae) && {
            let mut inputs: UnitVec<Node> = unitvec![];
            ae.inputs_rev(&mut inputs);
            inputs
                .iter()
                .all(|&n| self.is_input_independent_scalar(n, expr_arena))
        };

        self.is_input_independent_scalar.insert(node, ret);
        ret
    }

    /// Return whether the expression at `node` has the height of the frame.
    pub fn is_length_preserving(&mut self, node: Node, expr_arena: &Arena<AExpr>) -> bool {
        let mut visitor = CachedLengthPreservingVisitor {
            cache: &mut self.is_length_preserving,
            height_resolver: &mut Default::default(),
        };

        let height = aexpr_tree_traversal(
            node,
            &mut &*expr_arena,
            self.node_scratch.get(),
            self.height_scratch.get(),
            &mut visitor,
        )
        .continue_value()
        .unwrap();

        matches!(height, ExprProjectionHeight::Column)
    }
}

struct CachedLengthPreservingVisitor<'a> {
    height_resolver: &'a mut ExprHeightVisitor<'a>,
    cache: &'a mut PlIndexMap<Node, bool>,
}

impl<'a> NodeVisitor for CachedLengthPreservingVisitor<'a> {
    type Edge = ExprProjectionHeight;
    type Key = Node;
    type Storage = &'a Arena<AExpr>;
    type BreakValue = ();

    fn default_edge(
        &mut self,
        key: Self::Key,
        parent_key_and_port: Option<(Self::Key, usize)>,
    ) -> Self::Edge {
        self.height_resolver.default_edge(key, parent_key_and_port)
    }

    fn pre_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue, SubtreeVisit> {
        use ControlFlow as CF;

        if let Some(length_preserving) = self.cache.get(&key) {
            // TODO: A cached `false` comes back as `Unknown`, not the `Scalar` or `Range` it was,
            // so whether a parent counts as length-preserving depends on what was queried before.
            edges.outputs()[0] = if *length_preserving {
                ExprProjectionHeight::Column
            } else {
                ExprProjectionHeight::Unknown
            };

            CF::Continue(SubtreeVisit::Skip)
        } else {
            self.height_resolver.pre_visit(key, storage, edges)
        }
    }

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue> {
        let control_flow = self.height_resolver.post_visit(key, storage, edges);

        let length_preserving = matches!(edges.outputs()[0], ExprProjectionHeight::Column);

        self.cache.insert(key, length_preserving);

        control_flow
    }
}
