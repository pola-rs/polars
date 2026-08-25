//! Finding the parts of a plan whose joins may be reordered.
//!
//! A *cluster* is a maximal contiguous run of inner equi-joins. Its *leaves* are the
//! subtrees hanging off that run, which are opaque here: reordering permutes the
//! leaves and never looks inside one.
//!
//! Anything not known to be safe to reorder across ends the cluster instead.

use std::ops::Range;
use std::sync::Arc;

use polars_core::prelude::{PlIndexMap, PlIndexSet};
use polars_core::schema::SchemaRef;
use polars_ops::frame::JoinValidation;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
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
/// Coalescing joins pass here but are constrained further in [`coalesce_keys`]:
/// coalescing folds a key pair into one column under the left name, so only pairs of
/// identically named columns survive the inputs being swapped.
fn reorderable(options: &JoinOptionsIR) -> bool {
    let args = &options.args;

    matches!(args.how, JoinType::Inner)
        && args.slice.is_none()
        && matches!(args.maintain_order, MaintainOrderJoin::None)
        // Validation checks a named side for uniqueness; reordering would point it
        // at a different relation.
        && matches!(args.validation, JoinValidation::ManyToMany)
        // A forced build side refers to this specific join, so leave it alone.
        && args.build_side.is_none()
        && matches!(&options.options, JoinTypeOptionsIR::Equi { on } if !on.is_empty())
}

/// A key pair as written, together with the leaves either side of its join.
///
/// A key expression is resolved by name against its own input, so the sides have to
/// be looked up separately; the same name can occur on both.
struct RawKey {
    left_key: ExprIR,
    right_key: ExprIR,
    left_leaves: Range<usize>,
    right_leaves: Range<usize>,
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
    let mut raw_keys = Vec::new();
    collect(
        root,
        ir_arena,
        expr_arena,
        &options,
        &mut leaf_nodes,
        &mut raw_keys,
    );

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

    let mut edges = Vec::with_capacity(raw_keys.len());
    for raw in raw_keys {
        let left_leaf = owning_leaf(&raw.left_key, &leaves, raw.left_leaves, expr_arena)?;
        let right_leaf = owning_leaf(&raw.right_key, &leaves, raw.right_leaves, expr_arena)?;
        edges.push(Edge {
            left_leaf,
            right_leaf,
            left_key: raw.left_key,
            right_key: raw.right_key,
        });
    }

    let coalesced = if options.args.should_coalesce() {
        let (names, closed) = coalesce_keys(&leaves, &edges, expr_arena)?;
        edges = closed;
        names
    } else {
        PlIndexSet::default()
    };

    if !column_names_are_unambiguous(&leaves, &coalesced) {
        return None;
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
    expr_arena: &Arena<AExpr>,
    root_options: &JoinOptionsIR,
    leaves: &mut Vec<Node>,
    key_pairs: &mut Vec<RawKey>,
) {
    // Column-narrowing projections commonly sit between joins. They preserve rows
    // and rename nothing, so look past them for the join underneath; otherwise
    // almost every join is its own cluster.
    let peeled = peel_projections(node, ir_arena, expr_arena);

    match ir_arena.get(peeled) {
        IR::Join {
            input_left,
            input_right,
            options,
            ..
        } if reorderable(options) && same_settings(options, root_options) => {
            // Each side's leaves land in one contiguous run, which is the range the
            // keys of that side resolve against.
            let start = leaves.len();
            collect(
                *input_left,
                ir_arena,
                expr_arena,
                root_options,
                leaves,
                key_pairs,
            );
            let mid = leaves.len();
            collect(
                *input_right,
                ir_arena,
                expr_arena,
                root_options,
                leaves,
                key_pairs,
            );
            let end = leaves.len();

            if let Some(on) = options.options.key_pairs() {
                key_pairs.extend(on.iter().map(|(left_key, right_key)| RawKey {
                    left_key: left_key.clone(),
                    right_key: right_key.clone(),
                    left_leaves: start..mid,
                    right_leaves: mid..end,
                }));
            }
        },
        // Keep the unpeeled node: a projection on a leaf still narrows it.
        _ => leaves.push(node),
    }
}

/// Strip any chain of column-narrowing projections to reach the node beneath.
///
/// Dropping an interior projection widens the rows flowing through the rebuilt joins.
/// Projection pushdown runs after this pass and narrows them again against the new
/// order, and the cluster is projected back to its original schema, so the extra
/// columns are not observable.
///
/// That argument only holds for a projection which selects existing columns and
/// nothing more: one that computes or renames a column cannot be dropped, because the
/// restoring projection can only pick columns out of what the joins produce.
fn peel_projections(mut node: Node, ir_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Node {
    loop {
        node = match ir_arena.get(node) {
            IR::SimpleProjection { input, .. } => *input,
            // A `select` of plain columns is the same thing before `fast_projection`
            // (which runs after this pass) rewrites it into one.
            IR::Select { input, expr, .. }
                if expr.iter().all(|e| e.plain_column(expr_arena).is_some()) =>
            {
                *input
            },
            _ => return node,
        };
    }
}

/// Whether two joins agree on everything that survives being rebuilt.
///
/// The suffix is excluded because the SQL frontend names it after the right-hand
/// table, so no two joins agree on it. This is only sound while
/// [`column_names_are_unambiguous`] holds, since a suffix only applies on a collision.
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

/// Whether every column name in the cluster identifies exactly one output column.
///
/// A name held by two leaves is suffixed on collision, and which side gets the suffix
/// depends on which ends up left, so reordering could rename columns. Coalesced key
/// names are the exception: they are folded into one column rather than suffixed.
fn column_names_are_unambiguous(leaves: &[Leaf], coalesced: &PlIndexSet<PlSmallStr>) -> bool {
    let total: usize = leaves.iter().map(|l| l.schema.len()).sum();
    let mut seen = PlIndexSet::with_capacity_and_hasher(total, Default::default());
    leaves
        .iter()
        .flat_map(|l| l.schema.iter_names())
        .all(|name| coalesced.contains(name.as_str()) || seen.insert(name.as_str()))
}

/// The names a coalescing cluster folds away, and the edges closed over them, or
/// `None` if the cluster cannot be reordered.
///
/// Coalescing keeps the left key's column and drops the right one, so a pair naming
/// different columns would rename the output when the inputs swap. Only pairs of
/// identically named plain columns are accepted.
///
/// Equality is transitive across a run of inner joins, so the edges on a name are
/// replaced by the clique over every leaf holding it. Without those implied edges an
/// order could join two holders over some other key and leave both columns behind.
fn coalesce_keys(
    leaves: &[Leaf],
    edges: &[Edge],
    expr_arena: &Arena<AExpr>,
) -> Option<(PlIndexSet<PlSmallStr>, Vec<Edge>)> {
    let mut by_name: PlIndexMap<PlSmallStr, Vec<&Edge>> = PlIndexMap::default();
    for edge in edges {
        let name = edge.left_key.plain_column(expr_arena)?;
        if edge.right_key.plain_column(expr_arena) != Some(name) {
            return None;
        }
        by_name.entry(name.clone()).or_default().push(edge);
    }

    let mut closed = Vec::with_capacity(edges.len());
    for (name, on_name) in &by_name {
        let holders: Vec<usize> = (0..leaves.len())
            .filter(|&i| leaves[i].schema.contains(name.as_str()))
            .collect();
        if !folds_into_one_column(name, &holders, on_name, leaves) {
            return None;
        }

        // Every edge on `name` reads the same column on both sides, so any of them
        // supplies the key expressions for the whole clique.
        let template = on_name[0];
        for (nth, &left_leaf) in holders.iter().enumerate() {
            for &right_leaf in &holders[nth + 1..] {
                closed.push(Edge {
                    left_leaf,
                    right_leaf,
                    left_key: template.left_key.clone(),
                    right_key: template.right_key.clone(),
                });
            }
        }
    }

    Some((by_name.into_keys().collect(), closed))
}

/// Whether every leaf holding `name` collapses into a single column of that name.
///
/// The dtypes have to agree because coalescing keeps the left column, so the output
/// dtype would otherwise depend on the order. Reachability over the edges on `name`
/// is what makes the clique sound: joining two leaves that were not already connected
/// by this name would drop rows the original query kept.
fn folds_into_one_column(
    name: &PlSmallStr,
    holders: &[usize],
    edges: &[&Edge],
    leaves: &[Leaf],
) -> bool {
    let mut parent: Vec<usize> = (0..leaves.len()).collect();
    for edge in edges {
        let (left, right) = (
            find(&mut parent, edge.left_leaf),
            find(&mut parent, edge.right_leaf),
        );
        parent[left] = right;
    }

    let Some((&first, rest)) = holders.split_first() else {
        return false;
    };
    let root = find(&mut parent, first);
    let dtype = leaves[first].schema.get(name.as_str());

    rest.iter()
        .all(|&i| find(&mut parent, i) == root && leaves[i].schema.get(name.as_str()) == dtype)
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Which leaf of `range` a key expression reads from, or `None` if that is not
/// exactly one leaf.
fn owning_leaf(
    key: &ExprIR,
    leaves: &[Leaf],
    range: Range<usize>,
    expr_arena: &Arena<AExpr>,
) -> Option<usize> {
    let mut owner = None;
    for name in aexpr_to_leaf_names_iter(key.node(), expr_arena) {
        let found = leaves[range.clone()]
            .iter()
            .position(|leaf| leaf.schema.contains(name.as_str()))?
            + range.start;
        match owner {
            None => owner = Some(found),
            // A key spanning two leaves cannot be attributed to one side.
            Some(existing) if existing != found => return None,
            Some(_) => {},
        }
    }
    owner
}
