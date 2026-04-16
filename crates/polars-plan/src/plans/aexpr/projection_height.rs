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
    #[default]
    Unknown,
}

impl ExprProjectionHeight {
    pub fn zip_with(self, other: Self) -> Self {
        use ExprProjectionHeight::*;

        match (self, other) {
            (Scalar, v) | (v, Scalar) => v,
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
    mut expr_arena: &Arena<AExpr>,
    stack: &mut ScratchVec<Node>,
    edges_stack: &mut ScratchVec<ExprProjectionHeight>,
) -> ExprProjectionHeight {
    let mut visitor = ExprHeightVisitor::default();

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

#[derive(Default)]
pub struct ExprHeightVisitor<'a>(PhantomData<&'a ()>);

impl<'a> NodeVisitor for ExprHeightVisitor<'a> {
    type Key = Node;
    type Edge = ExprProjectionHeight;
    type Storage = &'a Arena<AExpr>;
    type BreakValue = ();

    fn default_edge(&mut self) -> Self::Edge {
        ExprProjectionHeight::Unknown
    }

    fn pre_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn crate::traversal::edge_provider::NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue, SubtreeVisit> {
        ControlFlow::Continue(
            if let Some(height) = aexpr_projection_height(storage.get(key), None) {
                edges.outputs()[0] = height;
                SubtreeVisit::Skip
            } else {
                SubtreeVisit::Visit
            },
        )
    }

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn crate::traversal::edge_provider::NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue> {
        edges.outputs()[0] =
            aexpr_projection_height(storage.get(key), Some(&mut *edges.inputs())).unwrap();
        ControlFlow::Continue(())
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
                H::Unknown
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
            let [pred, truthy, falsy] = input_heights?.try_into().unwrap();
            pred.zip_with(truthy).zip_with(falsy)
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
                }
            }
        },

        AExpr::Function { options, .. } | AExpr::AnonymousFunction { options, .. } => {
            if options.flags.returns_scalar() {
                H::Scalar
            } else if options.flags.is_elementwise() || options.flags.is_length_preserving() {
                H::zipped_projection_height(input_heights?.iter().copied())
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
