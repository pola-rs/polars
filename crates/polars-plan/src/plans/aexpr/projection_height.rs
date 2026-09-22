use std::marker::PhantomData;
use std::ops::ControlFlow;

use polars_utils::arena::{Arena, Node};
use polars_utils::collection::{Collection, CollectionWrap};
use polars_utils::scratch_vec::ScratchVec;

use crate::dsl::WindowMapping;
use crate::plans::{AExpr, aexpr_tree_traversal};
use crate::traversal::visitor::{NodeVisitor, SubtreeVisit};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ExprProjectionHeight {
    Column,
    Scalar,
    Range,
    #[default]
    Unknown,
}

impl ExprProjectionHeight {
    pub fn zip_with(self, other: Self) -> Self {
        use ExprProjectionHeight::*;

        match (self, other) {
            (Scalar, v) | (v, Scalar) => v,
            (Range, Column) | (Column, Range) => Column,
            (Range, _) | (_, Range) => Range,
            (Unknown, _) | (_, Unknown) => Unknown,
            (Column, Column) => Column,
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
    stack: &mut ScratchVec<Node>,
    edges_stack: &mut ScratchVec<ExprProjectionHeight>,
) -> ExprProjectionHeight {
    aexpr_projection_height_rec_with(
        ae_node,
        expr_arena,
        stack,
        edges_stack,
        ExprHeightOptions::default(),
    )
}

/// As [`aexpr_projection_height_rec`], but with the resolution of nodes whose height cannot be
/// derived from the expression alone under the caller's control. See [`ExprHeightOptions`].
#[recursive::recursive]
pub fn aexpr_projection_height_rec_with(
    ae_node: Node,
    mut expr_arena: &Arena<AExpr>,
    stack: &mut ScratchVec<Node>,
    edges_stack: &mut ScratchVec<ExprProjectionHeight>,
    options: ExprHeightOptions,
) -> ExprProjectionHeight {
    let mut visitor = ExprHeightVisitor {
        options,
        _phantom: PhantomData,
    };

    aexpr_tree_traversal(
        ae_node,
        &mut expr_arena,
        stack.get(),
        edges_stack.get(),
        &mut visitor,
    )
    .continue_value()
    .unwrap()
}

/// Controls how [`ExprHeightVisitor`] resolves the nodes whose height cannot be derived from the
/// expression alone.
#[derive(Debug, Clone, Copy)]
pub struct ExprHeightOptions {
    /// The height reported for a `StructField` reference.
    ///
    /// A `StructField` has the height of the struct it is evaluated against, which is unrelated to
    /// the height of the frame, hence [`ExprProjectionHeight::Unknown`] by default. Within the
    /// `evaluation` of an `AExpr::StructEval` that struct *is* the reference height, so there it
    /// is [`ExprProjectionHeight::Column`].
    pub struct_field: ExprProjectionHeight,

    /// If set, only the nodes that change the height *structurally* report
    /// [`ExprProjectionHeight::Unknown`]; every other node that cannot be resolved reports the
    /// zipped height of its inputs.
    ///
    /// Opaque expressions such as `map_batches`, `fold` and `search_sorted` have a height that is
    /// not statically known, which is not the same as them changing the height -- they typically
    /// do produce one value per input row. Callers that turn `Unknown` into a user-facing error
    /// should set this so that those are not rejected, and leave the rest to the runtime shape
    /// checks.
    pub structural_unknowns_only: bool,
}

impl Default for ExprHeightOptions {
    fn default() -> Self {
        Self {
            struct_field: ExprProjectionHeight::Unknown,
            structural_unknowns_only: false,
        }
    }
}

/// Returns whether `aexpr` produces a number of rows that is unrelated to the number of rows of
/// its input, no matter what the data is.
fn changes_height_structurally(aexpr: &AExpr) -> bool {
    match aexpr {
        AExpr::Explode { .. } | AExpr::Filter { .. } | AExpr::Slice { .. } => true,
        AExpr::Over { mapping, .. } => matches!(mapping, WindowMapping::Explode),
        _ => false,
    }
}

pub struct ExprHeightVisitor<'a> {
    options: ExprHeightOptions,
    _phantom: PhantomData<&'a ()>,
}

impl Default for ExprHeightVisitor<'_> {
    fn default() -> Self {
        Self {
            options: ExprHeightOptions::default(),
            _phantom: PhantomData,
        }
    }
}

impl<'a> NodeVisitor for ExprHeightVisitor<'a> {
    type Key = Node;
    type Edge = ExprProjectionHeight;
    type Storage = &'a Arena<AExpr>;
    type BreakValue = ();

    #[inline]
    fn default_edge(
        &mut self,
        _key: Self::Key,
        _parent_key_and_port: Option<(Self::Key, usize)>,
    ) -> Self::Edge {
        ExprProjectionHeight::Unknown
    }

    fn pre_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn crate::traversal::edge_provider::NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue, SubtreeVisit> {
        ControlFlow::Continue(if let Some(height) = self.height(storage.get(key), None) {
            edges.outputs()[0] = height;
            SubtreeVisit::Skip
        } else {
            SubtreeVisit::Visit
        })
    }

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn crate::traversal::edge_provider::NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue> {
        // @NOTE. `post_visit` also runs for nodes whose subtree `pre_visit` skipped, so the
        // `StructField` override has to live in `height()` rather than in `pre_visit`.
        edges.outputs()[0] = self
            .height(storage.get(key), Some(&mut *edges.inputs()))
            .unwrap();
        ControlFlow::Continue(())
    }
}

impl ExprHeightVisitor<'_> {
    fn height(
        &self,
        aexpr: &AExpr,
        input_heights: Option<&mut dyn Collection<ExprProjectionHeight>>,
    ) -> Option<ExprProjectionHeight> {
        #[cfg(feature = "dtype-struct")]
        if matches!(aexpr, AExpr::StructField(_)) {
            return Some(self.options.struct_field);
        }

        if !self.options.structural_unknowns_only {
            return aexpr_projection_height(aexpr, input_heights);
        }

        if changes_height_structurally(aexpr) {
            return Some(ExprProjectionHeight::Unknown);
        }

        let Some(input_heights) = input_heights else {
            // Returning `None` makes the traversal descend into the subtree and ask again from
            // `post_visit`, where the input heights are available.
            return aexpr_projection_height(aexpr, None)
                .filter(|h| !matches!(h, ExprProjectionHeight::Unknown));
        };

        let heights = || (0..input_heights.len()).map(|i| *input_heights.get(i).unwrap());
        let zipped = ExprProjectionHeight::zipped_projection_height(heights());
        // Only an `Unknown` that this node introduces itself may be assumed away; one that is
        // inherited from an input has to keep propagating, or a structural height change deeper
        // in the expression would be laundered here.
        let inherited_unknown = heights().any(|h| matches!(h, ExprProjectionHeight::Unknown));

        Some(match aexpr_projection_height(aexpr, Some(input_heights)) {
            Some(ExprProjectionHeight::Unknown) | None if !inherited_unknown => zipped,
            Some(height) => height,
            None => ExprProjectionHeight::Unknown,
        })
    }
}

/// # Returns
/// Returns `None` if the output height is dependent on input heights and input heights were not
/// provided.
pub fn aexpr_projection_height(
    aexpr: &AExpr,
    input_heights: Option<&mut dyn Collection<ExprProjectionHeight>>,
) -> Option<ExprProjectionHeight> {
    use AExpr::*;
    use ExprProjectionHeight as H;

    let input_heights = input_heights.map(CollectionWrap::<ExprProjectionHeight, _>::new);

    Some(match aexpr {
        Column(_) => H::Column,

        Element => H::Column,
        #[cfg(feature = "dtype-struct")]
        StructField(_) => H::Unknown,
        Literal(lv) => {
            if lv.is_scalar() {
                H::Scalar
            } else {
                H::Range
            }
        },

        Eval { .. } => input_heights?[0],
        #[cfg(feature = "dtype-struct")]
        StructEval { .. } => input_heights?[0],

        Filter { .. } | Slice { .. } | Explode { .. } => H::Unknown,

        Agg(_) | AnonymousAgg { .. } => H::Scalar,
        Len => H::Scalar,

        BinaryExpr { .. } => {
            let [l, r] = input_heights?.try_into().unwrap();
            l.zip_with(r)
        },
        Ternary { .. } => {
            let [truthy, falsy, pred] = input_heights?.try_into().unwrap();
            truthy.zip_with(falsy).zip_with(pred)
        },

        Cast { .. } | Sort { .. } => {
            let [h] = input_heights?.try_into().unwrap();
            h
        },

        SortBy { .. } => H::zipped_projection_height(input_heights?.iter().copied()),

        Gather { returns_scalar, .. } => {
            if *returns_scalar {
                // This is `get()` from the API
                H::Scalar
            } else {
                let indices_height = input_heights?[1];

                match indices_height {
                    H::Column => H::Column,
                    H::Scalar | H::Unknown => H::Unknown,
                    H::Range => H::Range,
                }
            }
        },

        AExpr::Function { options, .. } | AExpr::AnonymousFunction { options, .. } => {
            if options.flags.returns_scalar() {
                H::Scalar
            } else if options.flags.is_elementwise() || options.flags.is_length_preserving() {
                H::zipped_projection_height(input_heights?.iter().copied())
            } else if options.flags.is_range() {
                H::Range
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
    })
}
