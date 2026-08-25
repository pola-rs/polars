//! Finding the parts of a plan whose joins may be reordered.
//!
//! A *cluster* is a maximal contiguous run of inner equi-joins. Its *leaves* are the
//! subtrees hanging off that run, which are opaque here: reordering permutes the
//! leaves and never looks inside one.
//!
//! Anything not known to be safe to reorder across ends the cluster instead.

use std::sync::Arc;

use polars_core::prelude::PlIndexSet;
use polars_core::schema::SchemaRef;
use polars_ops::frame::JoinValidation;
use polars_utils::arena::{Arena, Node};
use recursive::recursive;

use super::stats::{LeafStats, leaf_stats};
use crate::plans::{AExpr, ExprIR, IR, JoinOptionsIR, JoinTypeOptionsIR, aexpr_to_leaf_names_iter};
use crate::prelude::{JoinArgs, JoinType, MaintainOrderJoin};

/// With two leaves there is only one order, so a cluster needs at least three.
const MIN_LEAVES: usize = 3;

pub(super) struct Leaf {
    pub(super) node: Node,
    pub(super) schema: SchemaRef,
    pub(super) stats: LeafStats,
}

/// One equi-key pair, resolved to the leaves it connects.
///
/// One join can contribute several, and they need not touch the same pair of leaves:
/// in `(a ⋈ b) ⋈ c ON a.x = c.x AND b.y = c.y` the outer join bridges `a`–`c` and
/// `b`–`c`.
pub(super) struct Edge {
    pub(super) left_leaf: usize,
    pub(super) right_leaf: usize,
    pub(super) left_key: ExprIR,
    pub(super) right_key: ExprIR,
}

/// An edge oriented against the leaves joined so far.
pub(super) struct Bridge<'a> {
    /// The already-joined leaf this edge reaches back to.
    pub(super) placed_leaf: usize,
    /// Key belonging to the accumulated (left) side.
    pub(super) placed_key: &'a ExprIR,
    /// Key belonging to the candidate (right) side.
    pub(super) candidate_key: &'a ExprIR,
}

pub(super) struct Cluster {
    pub(super) leaves: Vec<Leaf>,
    pub(super) edges: Vec<Edge>,
    /// Schema of the cluster root before reordering. The rebuilt plan is projected
    /// back to it.
    pub(super) output_schema: SchemaRef,
    /// Options used for every rebuilt join. [`same_settings`] guarantees all joins
    /// in the cluster agree on everything but their keys.
    pub(super) options: Arc<JoinOptionsIR>,
}

impl Cluster {
    /// Edges bridging `candidate` to anything already placed, oriented so the
    /// accumulated side is `placed_key` and `candidate` is `candidate_key`.
    ///
    /// An empty iterator means the candidate is unconnected, so joining it now would
    /// be a cross product. Ordering and key emission both use this, so they agree on
    /// what "connected" means.
    pub(super) fn bridging<'a>(
        &'a self,
        is_placed: &'a [bool],
        candidate: usize,
    ) -> impl Iterator<Item = Bridge<'a>> + 'a {
        self.edges.iter().filter_map(move |edge| {
            if edge.right_leaf == candidate && is_placed[edge.left_leaf] {
                Some(Bridge {
                    placed_leaf: edge.left_leaf,
                    placed_key: &edge.left_key,
                    candidate_key: &edge.right_key,
                })
            } else if edge.left_leaf == candidate && is_placed[edge.right_leaf] {
                Some(Bridge {
                    placed_leaf: edge.right_leaf,
                    placed_key: &edge.right_key,
                    candidate_key: &edge.left_key,
                })
            } else {
                None
            }
        })
    }
}

/// Whether a join node may be reordered against its neighbours.
///
/// Coalescing joins are excluded: they fold the two key columns into one and keep the
/// left name, so swapping the inputs of `a.join(b, left_on="xk", right_on="yk")`
/// would rename the output column from `xk` to `yk`.
fn reorderable(options: &JoinOptionsIR) -> bool {
    let args = &options.args;

    matches!(args.how, JoinType::Inner)
        && !args.should_coalesce()
        && args.slice.is_none()
        && matches!(args.maintain_order, MaintainOrderJoin::None)
        // Validation checks a named side for uniqueness; reordering would point it
        // at a different relation.
        && matches!(args.validation, JoinValidation::ManyToMany)
        // A forced build side refers to this specific join, so leave it alone.
        && args.build_side.is_none()
        && matches!(&options.options, JoinTypeOptionsIR::Equi { on } if !on.is_empty())
}

/// Extract the cluster rooted at `root`, or `None` if it cannot be reordered.
pub(super) fn extract(
    root: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Option<Cluster> {
    let IR::Join { options, .. } = ir_arena.get(root) else {
        return None;
    };
    if !reorderable(options) {
        return None;
    }
    let options = options.clone();

    let mut leaf_nodes = Vec::new();
    let mut key_pairs = Vec::new();
    collect(root, ir_arena, &options, &mut leaf_nodes, &mut key_pairs);

    if leaf_nodes.len() < MIN_LEAVES {
        return None;
    }

    // Every leaf needs an estimate. Ordering on partial information would order by
    // which leaves happened to be measurable.
    let mut leaves = Vec::with_capacity(leaf_nodes.len());
    for node in leaf_nodes {
        let stats = leaf_stats(node, ir_arena, expr_arena)?;
        let schema = ir_arena.get(node).schema(ir_arena).into_owned();
        leaves.push(Leaf {
            node,
            schema,
            stats,
        });
    }

    // Every column name must belong to exactly one leaf. A shared name is suffixed
    // on collision, and which side gets the suffix depends on which ends up left, so
    // reordering could rename columns.
    if !column_names_are_disjoint(&leaves) {
        return None;
    }

    let mut edges = Vec::with_capacity(key_pairs.len());
    for (left_key, right_key) in key_pairs {
        let left_leaf = owning_leaf(&left_key, &leaves, expr_arena)?;
        let right_leaf = owning_leaf(&right_key, &leaves, expr_arena)?;
        // A key pair inside one leaf is a filter, not a join condition.
        if left_leaf == right_leaf {
            return None;
        }
        edges.push(Edge {
            left_leaf,
            right_leaf,
            left_key,
            right_key,
        });
    }

    let output_schema = ir_arena.get(root).schema(ir_arena).into_owned();

    Some(Cluster {
        leaves,
        edges,
        output_schema,
        options,
    })
}

/// Walk the contiguous run of reorderable joins, collecting leaves and raw key pairs.
///
/// A join configured differently from the root becomes a leaf instead of being folded
/// in. Rebuilt joins inherit the root's settings, so folding in a join that disagreed
/// on, say, `nulls_equal` would change its meaning.
#[recursive]
fn collect(
    node: Node,
    ir_arena: &Arena<IR>,
    root_options: &JoinOptionsIR,
    leaves: &mut Vec<Node>,
    key_pairs: &mut Vec<(ExprIR, ExprIR)>,
) {
    // Column-narrowing projections commonly sit between joins. They preserve rows
    // and rename nothing, so look past them for the join underneath; otherwise
    // almost every join is its own cluster.
    let peeled = peel_projections(node, ir_arena);

    match ir_arena.get(peeled) {
        IR::Join {
            input_left,
            input_right,
            options,
            ..
        } if reorderable(options) && same_settings(options, root_options) => {
            if let Some(on) = options.options.key_pairs() {
                key_pairs.extend(on.iter().cloned());
            }
            collect(*input_left, ir_arena, root_options, leaves, key_pairs);
            collect(*input_right, ir_arena, root_options, leaves, key_pairs);
        },
        // Keep the unpeeled node: a projection on a leaf still narrows it.
        _ => leaves.push(node),
    }
}

/// Strip any chain of [`IR::SimpleProjection`] to reach the node beneath.
///
/// Dropping an interior projection widens the rows flowing through the rebuilt joins.
/// Projection pushdown runs after this pass and narrows them again against the new
/// order, and the cluster is projected back to its original schema, so the extra
/// columns are not observable.
fn peel_projections(mut node: Node, ir_arena: &Arena<IR>) -> Node {
    while let IR::SimpleProjection { input, .. } = ir_arena.get(node) {
        node = *input;
    }
    node
}

/// Whether two joins agree on everything that survives being rebuilt.
///
/// The suffix is excluded because the SQL frontend names it after the right-hand
/// table, so no two joins agree on it. This is only sound while
/// [`column_names_are_disjoint`] holds, since a suffix only applies on a collision.
///
/// Destructured so that a new `JoinArgs` field is a compile error here.
fn same_settings(a: &JoinOptionsIR, b: &JoinOptionsIR) -> bool {
    let JoinArgs {
        how,
        validation,
        suffix: _,
        slice,
        nulls_equal,
        coalesce,
        maintain_order,
        build_side,
    } = &a.args;

    *how == b.args.how
        && *validation == b.args.validation
        && *slice == b.args.slice
        && *nulls_equal == b.args.nulls_equal
        && *coalesce == b.args.coalesce
        && *maintain_order == b.args.maintain_order
        && *build_side == b.args.build_side
        && a.allow_parallel == b.allow_parallel
        && a.force_parallel == b.force_parallel
}

fn column_names_are_disjoint(leaves: &[Leaf]) -> bool {
    let total: usize = leaves.iter().map(|l| l.schema.len()).sum();
    let mut seen = PlIndexSet::with_capacity_and_hasher(total, Default::default());
    leaves
        .iter()
        .flat_map(|l| l.schema.iter_names())
        .all(|name| seen.insert(name.as_str()))
}

/// Which leaf a key expression reads from, or `None` if that is not exactly one leaf.
fn owning_leaf(key: &ExprIR, leaves: &[Leaf], expr_arena: &Arena<AExpr>) -> Option<usize> {
    let mut owner = None;
    for name in aexpr_to_leaf_names_iter(key.node(), expr_arena) {
        let found = leaves
            .iter()
            .position(|leaf| leaf.schema.contains(name.as_str()))?;
        match owner {
            None => owner = Some(found),
            // A key spanning two leaves cannot be attributed to one side.
            Some(existing) if existing != found => return None,
            Some(_) => {},
        }
    }
    owner
}
